import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from td3_trainer import collect_finetune_rollout, td3_finetune, _quick_evaluate
from critic_guided_updater import inject_critic_field_direction
from archive_io_utils import mask_fingerprint
from archive_saver import build_training_detail
from memory import Memory, MixedMemory
from mask_config import element_level_sparsity
from behavior_descriptor import (
    GRID_BINS_PER_DIM,
    aggregate_bd_from_episodes,
    resolve_control_dt,
)
from policy_identity import entry_stable_id

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Budget:
    total_update_steps: int
    update_frequency: int
    eval_interval: int
    eval_episodes: int
    warmup_episodes: int
    online_rollout_interval: int
    online_rollout_steps: int
    grad_clip_norm: float
    critic_dir_inject: bool
    critic_dir_alpha: float
    critic_dir_interval: int


class RetrainingScheduler:
    """Schedules per-iter retraining of NBC-selected archive agents."""

    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config

    def _ensure_private_memory(self, buffer, buffer_id: Optional[str], local_memory_size: int):
        """Ensure finetune network has a Memory private replay buffer."""
        if isinstance(buffer, Memory):
            mem_buffer = buffer
        else:
            mem_buffer = Memory(
                memory_size=int(local_memory_size),
                state_dim=self.optimizer.input_dim,
                action_dim=self.optimizer.output_dim,
            )
            if buffer is not None and hasattr(self.optimizer.sparse_optimizer, '_copy_replay_buffer'):
                try:
                    self.optimizer.sparse_optimizer._copy_replay_buffer(buffer, mem_buffer)
                except Exception as exc:
                    logger.debug("[scheduler] failed to copy replay into Memory for %s: %s",
                                 buffer_id, exc)
        if buffer_id and hasattr(self.optimizer.sparse_optimizer, 'register_sparse_memory'):
            self.optimizer.sparse_optimizer.register_sparse_memory(buffer_id, mem_buffer)
        return mem_buffer

    @staticmethod
    def _memory_size(buffer) -> int:
        try:
            return int(buffer.size()) if buffer is not None and hasattr(buffer, 'size') else 0
        except Exception:
            return 0

    def _private_replay_target_steps(self, sparsity: float) -> int:
        batch_size = int(getattr(self.config, 'batch_size', 256))
        target = max(
            batch_size,
            int(getattr(self.config, 'finetune_private_replay_min_steps', batch_size)),
        )
        if bool(getattr(self.config, 'finetune_skip_high_sparse_low_buffer', True)) and sparsity >= 0.7:
            target = max(
                target,
                int(getattr(self.config, 'finetune_min_buffer_for_high_sparse', 1000)),
            )
        return int(max(1, target))

    def _ensure_private_replay_ready(self, actor, env, buffer, buffer_id: str,
                                     sparsity: float) -> Dict[str, Any]:
        """Top up the private replay with the current sparse actor's rollout."""
        target_steps = self._private_replay_target_steps(float(sparsity))
        size_before = self._memory_size(buffer)
        episodes_budget = int(getattr(self.config, 'finetune_fallback_on_policy_episodes', 2))
        max_episode_steps = int(getattr(self.config, 'max_episode_steps', 1000))
        max_action = float(getattr(self.optimizer, 'max_action', 1.0))
        noise = float(getattr(
            self.config,
            'finetune_private_replay_noise',
            getattr(self.config, 'sparse_policy_noise', getattr(self.config, 'policy_noise', 0.1)),
        ))

        collected = 0
        episodes_used = 0
        actor = actor.to(device)
        while self._memory_size(buffer) < target_steps and episodes_used < episodes_budget:
            remaining = target_steps - self._memory_size(buffer)
            rollout_steps = min(max_episode_steps, max(1, remaining))
            got = collect_finetune_rollout(
                actor=actor, env=env, buffer=buffer,
                max_steps=rollout_steps, exploration_std=noise * max_action,
                max_action=max_action, config=self.config,
            )
            collected += int(got)
            episodes_used += 1
            if got <= 0:
                break

        size_after = self._memory_size(buffer)
        ready = size_after >= target_steps
        if buffer_id and hasattr(self.optimizer.sparse_optimizer, 'register_sparse_memory'):
            self.optimizer.sparse_optimizer.register_sparse_memory(buffer_id, buffer)
        if not ready:
            logger.info(
                "[scheduler] private replay for %s remains small: %d/%d after %d on-policy steps",
                buffer_id, size_after, target_steps, collected,
            )
        return {
            'private_replay_size_before': int(size_before),
            'private_replay_size_after': int(size_after),
            'private_replay_target_steps': int(target_steps),
            'private_replay_prefill_steps': int(collected),
            'private_replay_prefill_episodes': int(episodes_used),
            'private_replay_ready': bool(ready),
        }

    def _build_critic_sample_memory(self, sparse_buffer, dense_memory):
        """Sparse on-policy buffer + dense ES replay, weighted by config."""
        dense_ratio = float(getattr(self.config, 'finetune_critic_sample_dense_ratio', 0.3))
        sparse_ratio = float(getattr(self.config, 'finetune_critic_sample_sparse_ratio', 0.7))
        if dense_ratio >= sparse_ratio:
            dense_ratio, sparse_ratio = 0.3, 0.7

        sparse_ok = (
            sparse_buffer is not None
            and hasattr(sparse_buffer, 'size')
            and hasattr(sparse_buffer, 'sample')
        )
        if not sparse_ok:
            return dense_memory

        return MixedMemory(
            buffers=[dense_memory, sparse_buffer],
            ratio=[dense_ratio, sparse_ratio],
        )

    @staticmethod
    def fixed_budget(config, current_reward=None, best_reward=None,
                     archive_idx=None, archive=None, critic_bank=None,
                     q_std_grid=None, sparsity=0.0, buffer_size=0,
                     is_sparse=True) -> Budget:
        """Diversity-aware budget: total_mult = reward_mult × diversity_mult
        × critic_std_mult, capped at max_mult.
        """
        reward_mult = 1.0
        if (current_reward is not None and best_reward is not None
                and best_reward > 0
                and bool(getattr(config, 'ft_tiered_enabled', True))):
            if bool(getattr(config, 'ft_budget_by_headroom', False)):
                # Headroom allocates more refinement budget to low-return archive cells.
                span = max(1e-9, float(best_reward))
                headroom = 1.0 - min(max(float(current_reward) / span, 0.0), 1.0)
                lo = float(getattr(config, 'ft_headroom_zero_mult', 0.7))
                hi = float(getattr(config, 'ft_headroom_full_mult', 2.0))
                reward_mult = lo + (hi - lo) * headroom
            else:
                ratio = float(current_reward) / float(best_reward)
                if ratio >= 0.9:
                    reward_mult = 0.7
                elif ratio >= 0.5:
                    reward_mult = 1.0
                else:
                    reward_mult = 1.2

        diversity_mult = 1.0
        if archive_idx is not None and archive is not None and 0 <= archive_idx < len(archive):
            agent = archive[archive_idx]
            bd = agent[2] if len(agent) > 2 else None
            from adaptive_archive_admission import gait_bd_to_cell
            n_bins = int(getattr(config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM))
            cell = gait_bd_to_cell(bd, n_bins) if bd is not None else None
            if cell is not None:
                radius = int(getattr(config, 'finetune_diversity_radius', 2))
                low_thr = int(getattr(config, 'finetune_diversity_low_density_threshold', 2))
                high_thr = int(getattr(config, 'finetune_diversity_high_density_threshold', 5))
                n_neighbors = 0
                for i, other in enumerate(archive):
                    if i == archive_idx:
                        continue
                    if other is None or len(other) < 3:
                        continue
                    other_cell = gait_bd_to_cell(other[2], n_bins)
                    if other_cell is None:
                        continue
                    if abs(other_cell[0] - cell[0]) + abs(other_cell[1] - cell[1]) <= radius:
                        n_neighbors += 1
                if n_neighbors <= low_thr:
                    diversity_mult = float(getattr(config, 'finetune_diversity_mult_isolated', 2.0))
                elif n_neighbors <= high_thr:
                    diversity_mult = float(getattr(config, 'finetune_diversity_mult_sparse', 1.5))
                else:
                    diversity_mult = 1.0

        critic_std_mult = 1.0
        if (critic_bank is not None
                and bool(getattr(config, 'finetune_use_critic_std_bonus', True))
                and q_std_grid is not None
                and archive_idx is not None
                and archive is not None
                and 0 <= archive_idx < len(archive)):
            from adaptive_archive_admission import gait_bd_to_cell
            n_bins = int(getattr(config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM))
            agent = archive[archive_idx]
            bd = agent[2] if len(agent) > 2 else None
            cell = gait_bd_to_cell(bd, n_bins) if bd is not None else None
            if cell is not None and cell in q_std_grid:
                std_at_cell = float(q_std_grid[cell])
                ref_percentile = float(getattr(config, 'finetune_critic_std_ref_percentile', 90))
                vals = sorted(q_std_grid.values())
                ref = float(np.percentile(vals, ref_percentile)) if vals else 1.0
                ref = max(ref, 1e-6)
                norm = min(max(std_at_cell / ref, 0.0), 1.0)
                critic_std_mult = 1.0 + 0.5 * norm  # up to ×1.5
            else:
                critic_std_mult = 1.0

        max_mult = float(getattr(config, 'finetune_budget_max_mult', 2.0))
        total_mult = min(reward_mult * diversity_mult * critic_std_mult, max_mult)

        # Protect sparse actors with little replay.
        if is_sparse and bool(getattr(config, 'finetune_skip_high_sparse_low_buffer', True)):
            _min_buf = int(getattr(config, 'finetune_min_buffer_for_high_sparse', 1000))
            if sparsity >= 0.7 and buffer_size < _min_buf:
                total_mult *= float(getattr(config, 'finetune_high_sparse_low_buffer_mult', 0.3))

        if is_sparse:
            base_steps = int(getattr(config, 'finetune_td3_steps', 5000))
            eval_episodes_attr = 'finetune_eval_episodes_sparse'
            warmup_attr = 'finetune_warmup_episodes_sparse'
            warmup_default = int(getattr(config, 'finetune_prefill_episodes', 8))
        else:
            base_steps = int(getattr(config, 'finetune_td3_steps_dense', 8000))
            eval_episodes_attr = 'finetune_eval_episodes_dense'
            warmup_attr = 'finetune_warmup_episodes_dense'
            warmup_default = 0

        return Budget(
            total_update_steps=max(0, int(round(base_steps * total_mult))),
            update_frequency=int(getattr(config, 'policy_delay', 2)),
            eval_interval=1000,
            eval_episodes=int(getattr(config, eval_episodes_attr, 3 if not is_sparse else 2)),
            warmup_episodes=int(getattr(config, warmup_attr, warmup_default)),
            online_rollout_interval=500,
            online_rollout_steps=500,
            grad_clip_norm=0.5,
            critic_dir_inject=bool(getattr(config, 'retraining_critic_dir_inject', True)),
            critic_dir_alpha=float(getattr(config, 'retraining_critic_dir_alpha', 0.05)),
            critic_dir_interval=int(getattr(config, 'retraining_critic_dir_interval', 250)),
        )

    def _admission_like_seeds(self, n_episodes: int, salt: int = 0) -> List[int]:
        """Return checkpoint-comparable reset seeds independent of episode length."""
        base = int(getattr(self.config, 'seed', 1)) * 7919 + 13 * int(salt)
        return [base + i for i in range(max(1, int(n_episodes)))]

    def _evaluate_with_bd(self, actor, env, n_episodes, seeds, max_episode_steps,
                          env_name=None):
        """Evaluate `actor` and return (mean_reward, rewards, bd, steps)."""
        from evaluation import evaluate_actor
        rewards_arr, _rollouts, _gait, x_traj_list, xpos_traj_list = evaluate_actor(
            actor=actor, env=env, max_action=float(self.optimizer.max_action),
            memory=None, n_episodes=max(1, int(n_episodes)),
            max_episode_steps=max_episode_steps,
            return_rollout=False, return_bd_traj=True,
            env_name=env_name or self.optimizer.env_name,
            bc_type='gait', ant_bc_mode=str(getattr(self.config, 'ant_bc_mode', '2d')),
            episode_seeds=seeds,
        )
        rewards_arr = np.asarray(rewards_arr, dtype=np.float32).reshape(-1)
        steps = int(sum(max(0, len(t) - 1) for t in (x_traj_list or [])))
        if not x_traj_list or not xpos_traj_list:
            return (float(np.mean(rewards_arr)) if rewards_arr.size else 0.0,
                    rewards_arr, None, steps)
        bd = aggregate_bd_from_episodes(
            x_traj_list, xpos_traj_list,
            env_name or self.optimizer.env_name,
            dt=resolve_control_dt(env, env_name or self.optimizer.env_name),
        )
        bd = np.asarray(bd, dtype=np.float32).reshape(-1)
        return (float(np.mean(rewards_arr)) if rewards_arr.size else 0.0,
                rewards_arr, bd, steps)

    def _archive_cells_and_bests(self, ctx):
        """Occupied cells and per-cell best reward of the live archive."""
        from adaptive_archive_admission import gait_bd_to_cell
        n_bins = int(getattr(self.config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM))
        occupied = set()
        bests = {}
        for ind in (ctx.get('archive_ref') or []):
            try:
                bd = ind[2] if ind is not None and len(ind) > 2 else None
                cell = gait_bd_to_cell(bd, n_bins) if bd is not None else None
                if cell is None:
                    continue
                occupied.add(cell)
                r = float(ind[1]) if len(ind) > 1 else float('-inf')
                if cell not in bests or r > bests[cell]:
                    bests[cell] = r
            except Exception:
                continue
        return occupied, bests

    def _build_qd_probe_fn(self, ctx, budget, probe_state, keep_cells):
        """Keep intermediate policies that add or improve a cell, independent of parent return."""
        config = self.config
        if not bool(getattr(config, 'finetune_qd_candidate_enabled', False)):
            return None
        max_new = max(0, int(getattr(config, 'finetune_qd_candidate_max', 2)))
        probe_eps = max(1, int(getattr(config, 'finetune_qd_probe_episodes', 3)))
        env = ctx['env']
        max_episode_steps = ctx['max_episode_steps']
        from adaptive_archive_admission import gait_bd_to_cell
        n_bins = int(getattr(config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM))
        salt_base = int(ctx.get('agent_idx', 0)) * 101 + int(probe_state['counter'])

        def _probe(actor, update_step):
            probe_state['counter'] += 1
            seeds = self._admission_like_seeds(probe_eps, salt=salt_base)
            reward, rewards, bd, steps = self._evaluate_with_bd(
                actor, env, probe_eps, seeds, max_episode_steps)
            out = {'reward': reward, 'eval_steps': steps, 'n_evals': 1,
                   'obs_descriptor': bd, 'keep': False, 'reason': ''}
            if len(keep_cells) >= max_new:
                out['reason'] = 'candidate_budget_exhausted'
                return out
            if bd is None:
                out['reason'] = 'no_bd'
                return out
            cell = gait_bd_to_cell(bd, n_bins)
            if cell is None:
                out['reason'] = 'invalid_bd'
                return out
            if cell in keep_cells:
                out['reason'] = 'duplicate_cell_in_batch'
                return out
            occupied, bests = probe_state.get('archive_view', lambda: self._archive_cells_and_bests(ctx))()
            if cell not in occupied:
                out['reason'] = 'fills_empty_cell'
            elif float(reward) > float(bests.get(cell, float('-inf'))):
                out['reason'] = 'improves_occupied_cell'
            else:
                out['reason'] = 'no_archive_contribution'
                return out
            keep_cells.add(cell)
            out['keep'] = True
            return out

        return _probe

    def _build_direction_inject_fn(self, ctx, budget, probe_state, keep_cells,
                                   stats):
        """Factory for `td3_finetune`'s `direction_inject_fn`.

        The training trajectory compares reward with the running accepted-
        proposal reference, initialized from Stage A. Ordinary actor updates
        and periodic evaluations do not refresh that reference. The proposed
        actor is separately probed as a QD candidate before possible rollback.
        """
        config = self.config
        env = ctx['env']
        max_action = ctx['max_action']
        max_episode_steps = ctx['max_episode_steps']
        seed_offset_func = ctx['seed_offset_func']
        masks = ctx.get('masks')
        archive_ref = ctx.get('archive_ref')
        agent_idx = ctx.get('agent_idx')
        if not budget.critic_dir_inject or masks is None:
            return None

        def _inject(actor, critic, critic_target, update_step):
            reward_before = stats['last_eval_reward']
            dirs_added = inject_critic_field_direction(
                actor=actor, critic=critic,
                critic_bank=ctx.get('critic_bank'),
                memory=ctx.get('memory'), mask=masks,
                archive=archive_ref, agent_idx=agent_idx,
                n_neighbors=3, alpha=budget.critic_dir_alpha,
                noise_std=0.0, batch_size=256, grad_clip_norm=1.0,
            )
            stats['n_dirs_injected'] += int(dirs_added)
            if int(dirs_added) <= 0:
                return {'n_dirs': 0}
            try:
                injected_reward, injected_steps = _quick_evaluate(
                    actor, env, max_action,
                    int(getattr(config, 'seed', 0)),
                    seed_offset_func,
                    max_episode_steps,
                    n_episodes=budget.eval_episodes,
                )
            except Exception as exc:
                logger.warning('[scheduler] direction-inject eval failed: %s', exc)
                stats['n_dirs_reverted'] += int(dirs_added)
                return {'n_dirs': int(dirs_added), 'accepted': False,
                        'error': type(exc).__name__}
            stats['n_inject_evals'] += 1
            n_iters_so_far = float(getattr(self, '_global_iter_count', 0))
            margin = -min(0.5, 0.025 * n_iters_so_far)
            accept = float(injected_reward) >= float(reward_before) + margin
            if accept:
                stats['n_dirs_accepted'] += int(dirs_added)
                stats['last_eval_reward'] = float(injected_reward)
            else:
                stats['n_dirs_reverted'] += int(dirs_added)
            probe_out = {'n_dirs': 0}
            probe_fn = probe_state.get('qd_probe_fn')
            if probe_fn is not None and bool(
                    getattr(config, 'finetune_qd_probe_direction_inject', True)):
                probe_out = probe_fn(actor, update_step) or {}
            return {
                'n_dirs': int(dirs_added),
                'eval_steps': int(injected_steps),
                'n_evals': 1,
                'accepted': bool(accept),
                'margin': float(margin),
                'injected_reward': float(injected_reward),
                'probe': probe_out,
            }

        return _inject

    def apply_tiered_training(self, actor, critic, memory, env,
                               budget, ctx) -> Dict[str, Any]:
        """Calibrate the critic in Stage A, then refine the actor in continuous Stage B.
        Optional chunked mode carries executed-step offsets and sums all chunk costs."""
        config = ctx['config']
        max_action = ctx['max_action']
        seed_offset_func = ctx['seed_offset_func']
        max_episode_steps = ctx['max_episode_steps']
        base_reward = ctx['base_reward']
        agent_idx = ctx['agent_idx']
        critic_sample = ctx['critic_sample']
        actor_replay = ctx.get('actor_replay') or memory
        masks = ctx.get('masks')
        sparsity = float(ctx.get('sparsity', 0.0))
        archive_ref = ctx.get('archive_ref')

        steps_a = max(50, budget.total_update_steps // 5)
        t0 = time.time()
        try:
            res_a = td3_finetune(
                actor=actor, env=env, memory=memory,
                config=config, max_action=max_action,
                seed_offset_func=seed_offset_func,
                max_episode_steps=max_episode_steps,
                total_update_steps=steps_a,
                update_frequency=budget.update_frequency,
                sample_memory=critic_sample, write_memory=memory,
                critic_td3=critic,
                warmup_episodes=budget.warmup_episodes,
                base_reward=base_reward,
                sparsity=sparsity,
                eval_interval=max(50, budget.eval_interval // 4),
                eval_episodes=budget.eval_episodes,
                grad_clip_norm=budget.grad_clip_norm,
                online_rollout_interval=budget.online_rollout_interval,
                online_rollout_steps=budget.online_rollout_steps,
                conservative_actor_update=not bool(
                    getattr(config, 'ablation_plain_actor_update', False)),
                enable_actor_update=False,
                actor_sample_memory=actor_replay,
                actor_min_replay_steps=max(
                    int(getattr(config, 'batch_size', 256)),
                    int(getattr(config, 'finetune_actor_min_replay_steps',
                                getattr(config, 'batch_size', 256))),
                ),
            )
        except Exception as e:
            logger.warning("[scheduler] Stage A td3_finetune failed for agent %s: %s", agent_idx, e)
            res_a = self._fallback_result(actor, critic, base_reward, exception=str(e))
            res_a['wall_time_sec'] = float(time.time() - t0)
            return self._finalize(
                res_a, stages=['A'],
                early_exit=f'stage_a_failed:{type(e).__name__}',
                n_dirs_injected=0, ctx=ctx,
            )
        res_a['wall_time_sec'] = float(time.time() - t0)
        actor = res_a['actor']
        critic = res_a['critic']

        steps_b = max(0, budget.total_update_steps - steps_a)
        if steps_b <= 0:
            out = self._finalize(res_a, stages=['A'], early_exit=None,
                                 n_dirs_injected=0, ctx=ctx)
            out['stage_b_update_steps'] = 0
            out['qd_checkpoints'] = list(res_a.get('qd_checkpoints', []) or [])
            return out

        keep_cells: set = set()
        probe_state = {
            'counter': 0,
            'archive_view': lambda: self._archive_cells_and_bests(ctx),
        }
        stats = {
            'n_dirs_injected': 0,
            'n_dirs_accepted': 0,
            'n_dirs_reverted': 0,
            'n_inject_evals': 0,
            'last_eval_reward': float(res_a.get('reward', base_reward)),
        }
        qd_probe_fn = self._build_qd_probe_fn(ctx, budget, probe_state, keep_cells)
        probe_state['qd_probe_fn'] = qd_probe_fn
        direction_inject_fn = self._build_direction_inject_fn(
            ctx, budget, probe_state, keep_cells, stats)

        # Sum all chunk costs; res_b contains only the final chunk.
        stage_b_totals: Dict[str, int] = {}

        if bool(getattr(config, 'finetune_stage_b_chunked', False)):
            res_b, chunk_stats = self._run_stage_b_chunked(
                actor=actor, critic=critic, memory=memory, env=env,
                budget=budget, ctx=ctx, steps_b=steps_b,
                direction_inject_fn=direction_inject_fn,
                qd_probe_fn=qd_probe_fn,
                qd_probe_interval=int(getattr(config, 'finetune_qd_probe_interval', 0)),
            )
            for key, value in chunk_stats.items():
                stats[key] = stats.get(key, 0) + int(value)
            for _k in ('total_steps', 'grad_updates', 'n_evals',
                       'online_rollout_events',
                       'online_rollout_steps_collected'):
                if _k in chunk_stats:
                    stage_b_totals[_k] = int(chunk_stats[_k])
        else:
            t_b = time.time()
            try:
                res_b = td3_finetune(
                    actor=actor, env=env, memory=memory,
                    config=config, max_action=max_action,
                    seed_offset_func=seed_offset_func,
                    max_episode_steps=max_episode_steps,
                    total_update_steps=steps_b,
                    update_frequency=budget.update_frequency,
                    sample_memory=critic_sample, write_memory=memory,
                    critic_td3=critic,
                    warmup_episodes=0,    # Stage A already did warmup
                    base_reward=base_reward,
                    sparsity=sparsity,
                    eval_interval=budget.eval_interval,
                    eval_episodes=budget.eval_episodes,
                    grad_clip_norm=budget.grad_clip_norm,
                    online_rollout_interval=budget.online_rollout_interval,
                    online_rollout_steps=budget.online_rollout_steps,
                    enable_actor_update=True,
                    conservative_actor_update=not bool(
                        getattr(config, 'ablation_plain_actor_update', False)),
                    actor_sample_memory=actor_replay,
                    actor_min_replay_steps=max(
                        int(getattr(config, 'batch_size', 256)),
                        int(getattr(config, 'finetune_actor_min_replay_steps',
                                    getattr(config, 'batch_size', 256))),
                    ),
                    direction_inject_fn=direction_inject_fn,
                    qd_probe_fn=qd_probe_fn,
                    qd_probe_interval=int(getattr(config, 'finetune_qd_probe_interval', 0)),
                )
            except Exception as e:
                logger.warning(f"[scheduler] Stage B td3_finetune failed for agent {agent_idx}: {e}")
                res_b = self._fallback_result(actor, critic, base_reward, exception=str(e))
            res_b['wall_time_sec'] = float(time.time() - t_b)

        def _sb(key, default=0):
            if key in stage_b_totals:
                return int(stage_b_totals[key])
            return int(res_b.get(key, default) or 0)

        total_steps = int(res_a.get('total_steps', 0)) + _sb('total_steps')
        grad_updates = int(res_a.get('grad_updates', 0)) + _sb('grad_updates')
        online_events = int(res_a.get('online_rollout_events', 0)) + \
            _sb('online_rollout_events')
        online_steps = int(res_a.get('online_rollout_steps_collected', 0)) + \
            _sb('online_rollout_steps_collected')
        n_evals = int(res_a.get('n_evals', 0)) + _sb('n_evals')
        qd_checkpoints = list(res_a.get('qd_checkpoints', []) or []) + \
            list(res_b.get('qd_checkpoints', []) or [])

        out = self._finalize(res_b, stages=['A', 'B'], early_exit=None,
                             n_dirs_injected=stats['n_dirs_injected'], ctx=ctx)
        out['total_steps'] = total_steps
        out['grad_updates'] = grad_updates
        out['online_rollout_events'] = online_events
        out['online_rollout_steps_collected'] = online_steps
        out['n_evals'] = n_evals
        out['n_dirs_injected'] = int(stats['n_dirs_injected'])
        out['n_dirs_accepted'] = int(stats['n_dirs_accepted'])
        out['n_dirs_reverted'] = int(stats['n_dirs_reverted'])
        out['stage_a_update_steps'] = int(steps_a)
        out['stage_b_update_steps'] = int(steps_b)
        out['stage_b_chunked'] = bool(getattr(config, 'finetune_stage_b_chunked', False))
        out['critic_calibrated_not_rolled_back'] = bool(
            res_a.get('critic_calibrated_not_rolled_back', False))
        out['qd_checkpoints'] = qd_checkpoints
        return out

    def _run_stage_b_chunked(self, actor, critic, memory, env, budget, ctx,
                             steps_b, direction_inject_fn, qd_probe_fn,
                             qd_probe_interval) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Run chunked refinement with cumulative executed-step offsets and summed costs."""
        config = ctx['config']
        chunk_size = max(1, int(budget.critic_dir_interval)) \
            if budget.critic_dir_interval > 0 else steps_b
        totals = {
            'total_steps': 0, 'n_evals': 0, 'grad_updates': 0,
            'online_rollout_events': 0, 'online_rollout_steps_collected': 0,
            'n_dirs_injected': 0, 'n_dirs_accepted': 0, 'n_dirs_reverted': 0,
        }
        checkpoints: List[Dict[str, Any]] = []
        done = 0
        last = None
        update_offset = 0
        while done < steps_b:
            chunk = min(chunk_size, steps_b - done)
            try:
                res_chunk = td3_finetune(
                    actor=actor, env=env, memory=memory,
                    config=config, max_action=ctx['max_action'],
                    seed_offset_func=ctx['seed_offset_func'],
                    max_episode_steps=ctx['max_episode_steps'],
                    total_update_steps=chunk,
                    update_frequency=budget.update_frequency,
                    sample_memory=ctx['critic_sample'], write_memory=memory,
                    critic_td3=critic,
                    warmup_episodes=0,
                    base_reward=ctx['base_reward'],
                    sparsity=float(ctx.get('sparsity', 0.0)),
                    eval_interval=chunk,
                    eval_episodes=budget.eval_episodes,
                    grad_clip_norm=budget.grad_clip_norm,
                    online_rollout_interval=budget.online_rollout_interval,
                    online_rollout_steps=budget.online_rollout_steps,
                    enable_actor_update=True,
                    conservative_actor_update=not bool(
                        getattr(config, 'ablation_plain_actor_update', False)),
                    actor_sample_memory=ctx.get('actor_replay') or memory,
                    update_step_offset=update_offset,
                    direction_inject_fn=direction_inject_fn,
                    qd_probe_fn=qd_probe_fn,
                    qd_probe_interval=qd_probe_interval,
                )
            except Exception as e:
                logger.warning("[scheduler] Stage B chunk failed: %s", e)
                res_chunk = self._fallback_result(actor, critic, ctx['base_reward'],
                                                  exception=str(e))
            for key in ('total_steps', 'n_evals', 'grad_updates',
                        'online_rollout_events', 'online_rollout_steps_collected'):
                totals[key] += int(res_chunk.get(key, 0) or 0)
            totals['n_dirs_injected'] += int(res_chunk.get('n_dirs_injected', 0) or 0)
            totals['n_dirs_accepted'] += int(res_chunk.get('n_dirs_accepted', 0) or 0)
            totals['n_dirs_reverted'] += int(res_chunk.get('n_dirs_reverted', 0) or 0)
            checkpoints.extend(res_chunk.get('qd_checkpoints', []) or [])
            actor = res_chunk['actor'].to(device)
            critic = res_chunk['critic'].to(device)
            # Advance callback offsets by executed critic steps, including short or empty calls.
            update_offset += int(res_chunk.get('update_steps_used', 0) or 0)
            done += chunk
            last = res_chunk
        last = dict(last or {})
        last['qd_checkpoints'] = checkpoints
        return last, totals

    def train_one(self, agent, env, history: Dict[int, Dict[str, Any]],
                  agent_idx: int, iter_idx: int) -> Tuple:
        """Full retraining pipeline for one archive agent.

        Returns the 6-tuple consumed by main.py:
            (finetuned_actor, finetuned_reward, new_buffer_id,
             finetuned_critic_cpu, finetuned_critic_target_cpu, detail)
        """
        self._global_iter_count = int(iter_idx)
        base_network = agent[0]
        masks = agent[3] if len(agent) > 3 else None
        archive_critic = agent[6] if len(agent) > 6 and isinstance(agent[6], nn.Module) else None
        base_reward = float(agent[1]) if len(agent) > 1 else float('-inf')
        buffer_id = agent[8] if len(agent) > 8 else None
        archive_sparsity = float(agent[4]) if len(agent) > 4 and agent[4] is not None else 0.0
        is_sparse = masks is not None and archive_sparsity >= 0.10

        from networks import safe_deepcopy

        if archive_critic is None:
            raise ValueError(
                "Finetune requires archive_critic, but found None. "
                "Cannot proceed without archive critic."
            )

        actor_for_update = safe_deepcopy(base_network, device_target=device)
        critic_for_update = safe_deepcopy(archive_critic, device_target=device)

        local_memory_size = int(getattr(self.config, 'local_memory_size', 100000))

        if is_sparse:
            finetune_buffer = None
            if buffer_id and hasattr(self.optimizer.sparse_optimizer, 'get_sparse_memory'):
                finetune_buffer = self.optimizer.sparse_optimizer.get_sparse_memory(
                    buffer_id, create_if_missing=True,
                )
            if finetune_buffer is None:
                logger.warning(
                    "[scheduler] sparse archive buffer missing for %s; "
                    "creating empty private Memory and relying on on-policy prefill",
                    buffer_id,
                )
                finetune_buffer = Memory(
                    memory_size=local_memory_size,
                    state_dim=self.optimizer.input_dim,
                    action_dim=self.optimizer.output_dim,
                )
                if buffer_id and hasattr(self.optimizer.sparse_optimizer, 'register_sparse_memory'):
                    self.optimizer.sparse_optimizer.register_sparse_memory(buffer_id, finetune_buffer)
            finetune_buffer = self._ensure_private_memory(
                finetune_buffer, buffer_id, local_memory_size,
            )
            try:
                if hasattr(finetune_buffer, 'size') and finetune_buffer.size() > local_memory_size:
                    if hasattr(finetune_buffer, 'trim_to'):
                        finetune_buffer.trim_to(local_memory_size)
            except Exception:
                pass
            new_buffer_id = buffer_id or "sparse_unknown"

            replay_info = self._ensure_private_replay_ready(
                actor=actor_for_update, env=env,
                buffer=finetune_buffer, buffer_id=new_buffer_id,
                sparsity=float(archive_sparsity) if archive_sparsity is not None else 0.0,
            )

            sparsity_now = float(archive_sparsity) if archive_sparsity is not None else 0.0
            cur_buf_size = 0
            try:
                cur_buf_size = int(finetune_buffer.size()) if hasattr(finetune_buffer, 'size') else 0
            except Exception:
                cur_buf_size = 0
            if bool(getattr(self.config, 'finetune_skip_high_sparse_low_buffer', True)):
                min_buf_for_high_sp = int(getattr(self.config, 'finetune_min_buffer_for_high_sparse', 1000))
                if sparsity_now >= 0.7 and cur_buf_size < min_buf_for_high_sp:
                    logger.info(
                        "[scheduler] high-sparsity (%.1f%%) + small buffer (%d steps) agent %s: "
                        "shorten training to avoid regression",
                        sparsity_now * 100, cur_buf_size, agent_idx,
                    )

            critic_sample = self._build_critic_sample_memory(
                sparse_buffer=finetune_buffer,
                dense_memory=self.optimizer.sparse_optimizer.memory_dense,
            )
        else:
            finetune_buffer = self.optimizer.sparse_optimizer.memory_dense
            # Dense agents share replay; do not register per-mask ownership for that buffer.
            new_buffer_id = None
            cur_buf_size = finetune_buffer.size() if hasattr(finetune_buffer, 'size') else 0
            replay_info = {
                'private_replay_size_before': int(cur_buf_size),
                'private_replay_size_after': int(cur_buf_size),
                'private_replay_target_steps': 0,
                'private_replay_prefill_steps': 0,
                'private_replay_prefill_episodes': 0,
                'private_replay_ready': True,
            }
            critic_sample = finetune_buffer

        type_label = "稀疏" if is_sparse else "密集"
        mem_size = finetune_buffer.size() if hasattr(finetune_buffer, 'size') else 0
        dense_size = self.optimizer.sparse_optimizer.memory_dense.size()
        ready_note = "ready" if replay_info.get('private_replay_ready') else "small"
        if is_sparse:
            dense_ratio = float(getattr(self.config, 'finetune_critic_sample_dense_ratio', 0.3))
            sparse_ratio = float(getattr(self.config, 'finetune_critic_sample_sparse_ratio', 0.7))
            if dense_ratio >= sparse_ratio:
                dense_ratio, sparse_ratio = 0.3, 0.7
            ratio_total = max(1e-12, dense_ratio + sparse_ratio)
            dense_pct = int(round(100.0 * dense_ratio / ratio_total))
            sparse_pct = int(round(100.0 * sparse_ratio / ratio_total))
            print(f'  [memory] {type_label} agent {agent_idx}: '
                  f'网络Memory (buffer_id={new_buffer_id}, {mem_size} steps/{ready_note}, '
                  f'prefill={replay_info.get("private_replay_prefill_steps", 0)} steps); '
                  f'微调混合采样=memory_dense ({dense_size} steps) {dense_pct}% + 私有 {sparse_pct}% '
                  f'[MixedMemory]')
        else:
            print(f'  [memory] {type_label} agent {agent_idx}: '
                  f'复用 memory_dense ({mem_size} steps, buffer_id={new_buffer_id}, '
                  f'无私有 buffer) [直接采样]')

        sparsity = 0.0
        if is_sparse and masks is not None:
            sparsity, _ = element_level_sparsity(masks)
        ctx = {
            'agent_idx': agent_idx,
            'archive_entry': agent,
            'masks': masks,
            'critic_bank': getattr(self.optimizer, 'critic_bank', None),
            'archive_ref': self.optimizer.archive,
            'seed_offset_func': self.optimizer.sparse_optimizer._compute_seed_offset,
            'max_action': self.optimizer.max_action,
            'max_episode_steps': self.config.max_episode_steps,
            'config': self.config,
            'base_reward': base_reward,
            'history': history,
            'critic_sample': critic_sample,
            # Direction proposals sample the same replay mixture as their critic.
            'memory': critic_sample,
            # Actor updates use private replay; critic updates may use a dense/sparse mixture.
            'actor_replay': finetune_buffer,
            'env': env,
            'sparsity': sparsity,
            'replay_info': replay_info,
        }

        best_reward = float(getattr(self.optimizer, 'best_global_reward', 0.0))
        if best_reward <= 0:
            try:
                best_reward = max((float(a[1]) for a in self.optimizer.archive if len(a) > 1), default=0.0)
            except Exception:
                best_reward = 0.0
        try:
            _cur_buf_size = int(finetune_buffer.size()) if hasattr(finetune_buffer, 'size') else 0
        except Exception:
            _cur_buf_size = 0
        budget = self.fixed_budget(self.config, current_reward=base_reward, best_reward=best_reward,
                                   archive_idx=agent_idx, archive=self.optimizer.archive,
                                   critic_bank=getattr(self.optimizer, 'critic_bank', None),
                                   q_std_grid=getattr(self.optimizer, '_last_q_std_grid', None),
                                   sparsity=float(archive_sparsity) if archive_sparsity is not None else 0.0,
                                   buffer_size=_cur_buf_size,
                                   is_sparse=is_sparse)

        start = time.time()
        result = self.apply_tiered_training(
            actor=actor_for_update, critic=critic_for_update,
            memory=finetune_buffer, env=env, budget=budget, ctx=ctx,
        )
        training_time = time.time() - start
        result['total_steps'] = int(result.get('total_steps', 0)) + int(
            replay_info.get('private_replay_prefill_steps', 0)
        )
        result.update(replay_info)

        finetuned_actor = result['actor']
        finetuned_reward = result['reward']
        from networks import safe_deepcopy as _sdc
        finetuned_critic_cpu = _sdc(result['critic'], device_target='cpu') if result['critic'] is not None else None
        finetuned_critic_target_cpu = _sdc(result['critic_target'], device_target='cpu') if result.get('critic_target') is not None else None

        # Intermediate archive candidates are detached snapshots of the training trajectory.
        qd_side_candidates = self._build_qd_side_candidates(
            result.get('qd_checkpoints') or [], finetuned_actor, masks,
            candidate_id=new_buffer_id, sparsity=float(sparsity),
            critic=result.get('critic'), critic_target=result.get('critic_target'),
            source_archive_idx=int(agent_idx), source_reward=float(base_reward),
        )

        # Stable policy IDs prevent replacement occupants from inheriting training history.
        stable_id = entry_stable_id(agent)
        hkey = f'sid:{stable_id}' if stable_id is not None else f'idx:{agent_idx}'
        h = history.setdefault(hkey, {
            'n_retrains': 0, 'last_iter': -10**6,
            'best_observed_reward': float('-inf'), 'recent_rewards': [],
            'stable_id': stable_id, 'last_archive_idx': int(agent_idx),
        })
        h['n_retrains'] += 1
        h['last_iter'] = int(iter_idx)
        h['last_archive_idx'] = int(agent_idx)
        h['best_observed_reward'] = max(float(h['best_observed_reward']), float(finetuned_reward))
        h['recent_rewards'].append(float(finetuned_reward))
        if len(h['recent_rewards']) > 10:
            h['recent_rewards'] = h['recent_rewards'][-10:]

        try:
            detail = build_training_detail(
                archive_idx=agent_idx,
                network_type='sparse' if is_sparse else 'dense',
                sparsity=sparsity,
                base_reward=base_reward,
                finetuned_reward=finetuned_reward,
                warmup_episodes=budget.warmup_episodes,
                training_time_sec=training_time,
                td3_result=result,
            )
            if isinstance(detail, dict):
                detail['stages_completed'] = result.get('stages_completed', [])
                detail['early_exit_reason'] = result.get('early_exit_reason', None)
                detail['n_dirs_injected'] = int(result.get('n_dirs_injected', 0))
                detail['n_dirs_accepted'] = int(result.get('n_dirs_accepted', 0))
                detail['n_dirs_reverted'] = int(result.get('n_dirs_reverted', 0))
                detail['actor_update_enabled'] = bool(result.get('actor_update_enabled', True))
                detail['actor_anchor_weight'] = float(result.get('actor_anchor_weight', 0.0))
                detail['actor_bc_weight'] = float(result.get('actor_bc_weight', 0.0))
                detail['actor_q_weight'] = float(result.get('actor_q_weight', 0.0))
                detail['actual_actor_lr'] = float(result.get('actual_actor_lr', 0.0))
                detail['actual_critic_lr'] = float(result.get('actual_critic_lr', 0.0))
                detail['best_eval'] = float(result.get('best_eval', 0.0))
                detail['final_eval'] = float(result.get('final_eval', 0.0))
                detail['actor_update_skipped_insufficient_replay'] = int(
                    result.get('actor_update_skipped_insufficient_replay', 0)
                )
                detail['network_memory_type'] = type(finetune_buffer).__name__
                detail['critic_sample_memory_type'] = type(critic_sample).__name__
                detail.update(replay_info)
        except Exception as e:
            logger.debug(f"build_training_detail failed, fallback: {e}")
            detail = {
                'archive_idx': int(agent_idx),
                'network_type': 'sparse' if is_sparse else 'dense',
                'sparsity': float(sparsity),
                'base_reward': float(base_reward),
                'finetuned_reward': float(finetuned_reward),
                'improvement': float(finetuned_reward - base_reward),
                'training_time_sec': float(training_time),
                'warmup_episodes': int(budget.warmup_episodes),
                'stages_completed': result.get('stages_completed', []),
                'early_exit_reason': result.get('early_exit_reason'),
                'n_dirs_injected': int(result.get('n_dirs_injected', 0)),
                'n_dirs_accepted': int(result.get('n_dirs_accepted', 0)),
                'n_dirs_reverted': int(result.get('n_dirs_reverted', 0)),
                'n_evals': int(result.get('n_evals', 0)),
                'total_steps': int(result.get('total_steps', 0)),
                'network_memory_type': type(finetune_buffer).__name__,
                'critic_sample_memory_type': type(critic_sample).__name__,
            }
            detail.update(replay_info)

        detail['qd_side_candidates'] = int(len(qd_side_candidates))
        detail['actor_best_restored'] = bool(result.get('actor_best_restored', False))
        detail['critic_calibrated_not_rolled_back'] = bool(
            result.get('critic_calibrated_not_rolled_back', False))
        detail['online_rollout_events'] = int(result.get('online_rollout_events', 0))
        detail['online_rollout_steps_collected'] = int(
            result.get('online_rollout_steps_collected', 0))
        detail['grad_updates'] = int(result.get('grad_updates', 0))
        detail['stage_b_chunked'] = bool(result.get('stage_b_chunked', False))
        detail['stage_a_update_steps'] = int(result.get('stage_a_update_steps', 0))
        detail['stage_b_update_steps'] = int(result.get('stage_b_update_steps', 0))
        detail['total_steps'] = int(result.get('total_steps', 0))
        detail['history_key'] = hkey

        return (finetuned_actor, float(finetuned_reward), new_buffer_id,
                finetuned_critic_cpu, finetuned_critic_target_cpu, detail,
                qd_side_candidates)

    def _build_qd_side_candidates(self, checkpoints, fallback_actor, masks,
                                  candidate_id, sparsity, critic, critic_target,
                                  source_archive_idx, source_reward) -> List[Dict[str, Any]]:
        """Build intermediate candidates for full admission reevaluation; probe returns are provisional."""
        from networks import safe_deepcopy
        out: List[Dict[str, Any]] = []
        base = safe_deepcopy(fallback_actor, device_target='cpu')
        for ck in checkpoints or []:
            try:
                net = safe_deepcopy(base, device_target='cpu')
                net.load_state_dict(ck['actor_state'])
            except Exception as exc:
                logger.debug('[scheduler] skipping unusable QD checkpoint: %s', exc)
                continue
            out.append({
                'network': net,
                'critic': safe_deepcopy(critic, device_target='cpu') if critic is not None else None,
                'critic_target': safe_deepcopy(critic_target, device_target='cpu')
                if critic_target is not None else None,
                'masks': masks,
                'candidate_id': candidate_id,
                'memory_id': candidate_id if isinstance(candidate_id, str) else None,
                'source_sparsity': float(sparsity),
                'source_reward': float(source_reward),
                'source': 'finetune',
                'avg_reward': None,
                'obs_descriptor': None,
                '_force_reeval': True,
                'n_evals': 1,
                '_source_archive_idx': int(source_archive_idx),
                '_qd_side_candidate': True,
                '_qd_side_reason': str(ck.get('reason', '')),
                '_qd_side_update_step': int(ck.get('update_step', 0)),
                '_qd_side_probe_reward': float(ck.get('reward', float('-inf'))),
            })
        return out

    def _env_budget_exhausted(self) -> bool:
        """Whether the run's environment-step cap has been reached."""
        cap = int(getattr(self.config, 'max_env_steps', 0) or 0)
        if cap <= 0:
            return False
        opt = getattr(self, 'optimizer', None)
        try:
            return int(getattr(opt, 'total_env_steps', 0) or 0) >= cap
        except Exception:
            return False

    def run_for_iter(self, archive, env, history, iter_idx, nbc_selected_indices) -> List[Dict[str, Any]]:
        """Train each NBC-selected agent and return their training_details."""
        if bool(getattr(self.config, "disable_td3_finetune", False)) or bool(
            getattr(self.config, "ablation_no_td3", False)
        ):
            label = "w/o TD3" if bool(getattr(self.config, "ablation_no_td3", False)) else "disable_td3_finetune"
            print(f"[retraining] {label}=True — TD3 fine-tune 跳过")
            return []
        training_details: List[Dict[str, Any]] = []
        fine_tuned_candidates: List[Dict[str, Any]] = []
        pending_details: List[Dict[str, Any]] = []

        for i, agent_idx in enumerate(nbc_selected_indices):
            if agent_idx < 0 or agent_idx >= len(archive):
                continue
            # Check the interaction cap before selecting the next representative.
            if self._env_budget_exhausted():
                print('  [budget] env-step cap reached; stopping fine-tune '
                      'for the remaining agents')
                break
            self.optimizer.current_agent_idx = i
            agent = archive[agent_idx]

            agent_sparsity = float(agent[4]) if len(agent) > 4 and agent[4] is not None else 0.0
            agent_masks = agent[3] if len(agent) > 3 else None
            is_sparse = agent_masks is not None and agent_sparsity >= 0.10
            type_label = "稀疏" if is_sparse else "密集"
            agent_reward = float(agent[1]) if len(agent) > 1 else 0.0
            print(f'  训练 agent {agent_idx}: 类型={type_label}, '
                  f'稀疏度={agent_sparsity:.1%}, reward={agent_reward:.2f}')

            try:
                (updated_network, updated_score, used_memory_id,
                 updated_critic, updated_critic_target,
                 detail, qd_side_candidates) = self.train_one(
                    agent, env, history, agent_idx=agent_idx, iter_idx=iter_idx)
            except Exception as e:
                logger.warning(f"[scheduler] train_one failed for agent {agent_idx}: {e}")
                continue

            ft_n_evals = int(detail.get('n_evals', 0)) if detail else 0
            training_details.append(detail)

            improvement = float(updated_score) - agent_reward
            print(f'  [训练前后] {type_label} agent {agent_idx}: '
                  f'{agent_reward:>8.2f}  →  {float(updated_score):>8.2f}  '
                  f'(Δ{improvement:+.2f})')

            if isinstance(detail, dict):
                detail['submitted_to_archive'] = True
                detail['archive_submit_policy'] = 'always_evaluate_after_finetune'

            candidate_id = used_memory_id if isinstance(used_memory_id, str) else None
            if not candidate_id and agent_masks is not None:
                candidate_id = mask_fingerprint(agent_masks)
            pending_details.append(detail)
            fine_tuned_candidates.append({
                'network': updated_network,
                'critic': updated_critic,
                'critic_target': updated_critic_target,
                'masks': agent_masks,
                'candidate_id': candidate_id,
                'memory_id': used_memory_id if isinstance(used_memory_id, str) else None,
                'source_sparsity': agent_sparsity,
                'source_reward': agent_reward,
                'source': 'finetune',
                'avg_reward': None,        # force re-eval of the post-finetune actor
                'obs_descriptor': None,
                '_force_reeval': True,
                'n_evals': 1 + ft_n_evals,
                '_source_archive_idx': int(agent_idx),
            })
            # Intermediate candidates use the same cell competition and admission rules.
            for side in qd_side_candidates or []:
                fine_tuned_candidates.append(side)
                pending_details.append({
                    'archive_idx': int(agent_idx),
                    'network_type': 'sparse' if is_sparse else 'dense',
                    'sparsity': float(agent_sparsity),
                    'base_reward': float(agent_reward),
                    'finetuned_reward': float(agent_reward),
                    'qd_side_candidate': True,
                    'qd_side_reason': side.get('_qd_side_reason', ''),
                    'qd_side_probe_reward': side.get('_qd_side_probe_reward'),
                    'submitted_to_archive': True,
                    'archive_submit_policy': 'qd_side_candidate',
                    'n_evals': 0,
                    'total_steps': 0,
                })
                training_details.append(pending_details[-1])

        if fine_tuned_candidates:
            source_idx_by_candidate = []
            for ft_cand in fine_tuned_candidates:
                src = None
                if isinstance(ft_cand, dict):
                    src = ft_cand.get('_source_archive_idx')
                source_idx_by_candidate.append(int(src) if src is not None else -1)

            admission_results = self.optimizer._evaluate_and_archive_candidates(
                es_candidates=None,
                mask_candidates=fine_tuned_candidates,
                env=env,
                return_results=True,
            )
            for i, (detail, admission) in enumerate(
                zip(pending_details, admission_results or [])
            ):
                if not isinstance(detail, dict) or not isinstance(admission, dict):
                    continue
                admission_reward = admission.get('avg_reward')
                source_reward = float(detail.get('base_reward', 0.0))
                detail.update({
                    'admission_status': admission.get('evaluation_status', 'unknown'),
                    'admission_action': admission.get('action', 'unknown'),
                    'admitted_to_archive': bool(admission.get('admitted', False)),
                    'admission_archive_index': int(admission.get('archive_index', -1)),
                    'admission_reason': admission.get('reason', admission.get('action', 'unknown')),
                    'admission_cell': admission.get('cell'),
                    'admission_reward': (
                        float(admission_reward)
                        if admission_reward is not None and np.isfinite(float(admission_reward))
                        else None
                    ),
                })
                if detail['admission_reward'] is not None:
                    detail['admission_improvement'] = float(detail['admission_reward'] - source_reward)

                if detail['admitted_to_archive']:
                    self._record_source_lineage_after_finetune(
                        source_idx=source_idx_by_candidate[i] if i < len(source_idx_by_candidate) else -1,
                        admission_action=admission.get('action', ''),
                        admission_archive_index=detail['admission_archive_index'],
                        detail=detail,
                    )
                parent_note = (
                    detail.get('source_evict_reason', 'parent_kept')
                    if detail['admitted_to_archive'] else 'parent_kept_rejected_child'
                )
                reward_text = (
                    f"{detail['admission_reward']:.2f}"
                    if detail.get('admission_reward') is not None else 'nan'
                )
                print(
                    f"  [入档结果] agent {source_idx_by_candidate[i] if i < len(source_idx_by_candidate) else -1}: "
                    f"{detail['admission_action']} reward={reward_text} "
                    f"reason={detail['admission_reason']} parent={parent_note}"
                )

        return training_details

    @staticmethod
    def _record_source_lineage_after_finetune(
        source_idx, admission_action, admission_archive_index, detail,
    ) -> None:
        """Record MAP-Elites lineage without deleting the parent entry."""
        detail['source_evicted'] = False
        detail['source_archive_idx'] = int(source_idx)
        if admission_action == 'replace' and admission_archive_index == source_idx:
            detail['source_evict_reason'] = 'in_place_upgrade_by_admission'
            return
        detail['source_evict_reason'] = 'parent_kept_child_admitted'

    @staticmethod
    def _finalize(res, stages, early_exit, n_dirs_injected, ctx):
        out = dict(res)
        out['stages_completed'] = list(stages)
        out['early_exit_reason'] = early_exit
        out['n_dirs_injected'] = int(n_dirs_injected)
        out['agent_idx'] = int(ctx.get('agent_idx', -1))
        out['reward'] = float(res.get('reward', float('-inf')))
        out['total_steps'] = int(res.get('total_steps', 0))
        return out

    @staticmethod
    def _fallback_result(actor, critic, base_reward, exception="") -> Dict[str, Any]:
        """Training-failure fallback that preserves the input actor/critic."""
        from networks import safe_deepcopy
        return {
            'actor': safe_deepcopy(actor, device_target='cpu'),
            'critic': safe_deepcopy(critic, device_target='cpu'),
            'critic_target': safe_deepcopy(critic, device_target='cpu'),
            'reward': float(base_reward),
            'wall_time_sec': 0.0,
            'total_steps': 0,
            'n_evals': 0,
            'initial_eval': float(base_reward),
            'final_eval': float(base_reward),
            'best_eval': float(base_reward),
            'stopped_reason': f'fallback: {exception}',
            'update_steps_used': 0,
        }
