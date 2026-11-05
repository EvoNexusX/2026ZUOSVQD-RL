"""TD3 training: evaluate / td3_finetune / collect_finetune_rollout."""
import os
import numpy as np
import torch
import logging
import torch.nn.functional as F
from typing import Any, Dict
from copy import deepcopy

from experiment_utils import seed_env
from evaluation import evaluate_actor as _evaluate_actor
from replay_flags import replay_terminal, unpack_step

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA_GRAPH_IN_ROLLOUT = os.environ.get("FT_CUDA_GRAPH_ROLLOUT", "1") == "1"


def evaluate_actor_with_td3_update(
    actor,
    env,
    memory,
    config,
    max_action,
    seed_offset_func,
    max_episode_steps=1000,
    update_steps=10,
    update_frequency=2,
    sample_memory=None,
    write_memory=None,
    critic_td3=None,
    return_updated_critics=False,
    warmup_episodes=None,
    return_bd=False,
    env_name=None,
    eval_episodes=1,
    enable_actor_update=True,
    conservative_actor_update=False,
    actor_sample_memory=None,
    actor_min_replay_steps=None,
    batch_size_override=None,
    grad_clip_norm=0.5,
    warmup_max_steps=None,
    lambda_explore=0.0,
    exploration_noise_std=None,
    collect_eval_rollout_to_memory=False,
):
    """Run a brief TD3 finetune then evaluate. Optionally return BD alongside
    the reward so the caller can skip a redundant fair eval.
    """
    if sample_memory is None:
        sample_memory = memory
    if write_memory is None:
        write_memory = memory
    if actor_sample_memory is None:
        actor_sample_memory = sample_memory

    c = critic_td3
    if c is None:
        raise ValueError("critic_td3 is required for TD3 finetune")
    c = c.to(device)
    total_steps = 0
    from networks import safe_deepcopy
    actor_target = safe_deepcopy(actor, device_target=device)
    actor.train()
    critic_target = safe_deepcopy(c, device_target=device)
    anchor_actor = safe_deepcopy(actor, device_target=device)
    anchor_actor.eval()
    for p in anchor_actor.parameters():
        p.requires_grad_(False)

    if hasattr(env, 'n_envs') and getattr(env, 'n_envs', 1) > 1 and hasattr(env, 'single_env_view'):
        env = env.single_env_view(0)

    if warmup_episodes is None:
        warmup_episodes = int(getattr(config, 'warmup_episodes', 4))
    warmup_episodes = int(warmup_episodes)
    if exploration_noise_std is None:
        if exploration_noise_std := getattr(config, 'sparse_policy_noise', None):
            exploration_noise_std = float(exploration_noise_std or getattr(config, 'policy_noise', 0.2))
        else:
            exploration_noise_std = float(getattr(config, 'policy_noise', 0.2))
    else:
        exploration_noise_std = float(exploration_noise_std)
    bs = int(batch_size_override or getattr(config, 'batch_size', 256))
    bs = max(1, bs)
    actor_min_replay_steps = max(
        bs,
        int(actor_min_replay_steps if actor_min_replay_steps is not None else bs),
    )
    warmup_budget = (
        int(warmup_max_steps)
        if warmup_max_steps is not None and int(warmup_max_steps) > 0
        else warmup_episodes * max_episode_steps
    )
    warmup_used = 0

    def _maybe_actor_update(batch_size):
        if not enable_actor_update:
            return
        if conservative_actor_update:
            try:
                actor_memory_size = (
                    int(actor_sample_memory.size())
                    if actor_sample_memory is not None and hasattr(actor_sample_memory, 'size')
                    else 0
                )
            except Exception:
                actor_memory_size = 0
            if actor_memory_size < actor_min_replay_steps:
                return
            _conservative_actor_update(
                actor=actor, critic=c, memory=actor_sample_memory,
                batch_size=min(batch_size, actor_memory_size),
                anchor_actor=anchor_actor,
                grad_clip_norm=float(grad_clip_norm),
                bc_weight=float(getattr(config, 'finetune_actor_bc_weight', 0.05)),
                anchor_weight=float(getattr(config, 'finetune_actor_anchor_weight_sparse', 0.5)),
                q_weight=float(getattr(config, 'finetune_actor_q_weight', 1.0)),
            )
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(actor.tau * param.data + (1 - actor.tau) * target_param.data)
            return
        actor.update(sample_memory, batch_size, c, actor_target, lambda_explore=lambda_explore)

    for episode in range(warmup_episodes):
        if warmup_used >= warmup_budget:
            break
        offset = seed_offset_func(episode_idx=episode)
        state = seed_env(env, int(getattr(config, 'seed', 0)), offset)
        if isinstance(state, tuple):
            state = state[0]
        episode_limit = min(max_episode_steps, warmup_budget - warmup_used)
        for step in range(episode_limit):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()[0]
                noise_std = exploration_noise_std * max_action
                exploration_noise = np.random.normal(0, noise_std, size=action.shape)
                action = np.clip(action + exploration_noise, -max_action, max_action)
            next_state, reward, done, truncated, info = env.step(action)
            # Bootstrap through time-limit truncation; only termination zeros the TD mask.
            write_memory.add((state, next_state, action, reward, float(done)))
            state = next_state
            total_steps += 1
            warmup_used += 1
            if step % 20 == 0:
                batch_size = bs
                if sample_memory is not None and int(sample_memory.size()) >= batch_size:
                    c.update(sample_memory, batch_size, actor_t=actor_target, critic_t=critic_target)
                    _maybe_actor_update(batch_size)
            if done or truncated:
                break

    sampler_for_update = sample_memory
    if sampler_for_update is not None and int(sampler_for_update.size()) >= max(bs, 1):
        batch_size = bs * 2
        for update_step in range(int(update_steps)):
            if int(sampler_for_update.size()) < batch_size:
                break
            c.update(sampler_for_update, batch_size, actor_t=actor_target, critic_t=critic_target)
            if (update_step + 1) % int(max(1, update_frequency)) == 0:
                _maybe_actor_update(batch_size)

    seed_val = int(getattr(config, 'seed', 0))
    offset = seed_offset_func(episode_idx=0)
    obs_bd = None
    eval_episodes = int(max(1, eval_episodes))
    _eval_memory = write_memory if collect_eval_rollout_to_memory else None
    if return_bd:
        try:
            _res = _evaluate_actor(
                actor, env, max_action=max_action,
                memory=_eval_memory, n_episodes=eval_episodes, base_seed=seed_val, step_offset=offset,
                max_episode_steps=max_episode_steps,
                return_rollout=True, return_bd_traj=True,
                env_name=env_name or getattr(config, 'env_name', None),
                bc_type=getattr(config, 'BC', 'gait'),
                ant_bc_mode=getattr(config, 'ant_bc_mode', '2d'),
            )
            if isinstance(_res, tuple) and len(_res) >= 5:
                _rew_arr, _rollout_datas, _, _x_traj, _xpos_traj = _res
                average_reward = float(np.mean(_rew_arr)) if np.size(_rew_arr) > 0 else 0.0
                eval_steps = (
                    int(sum(len(ep_rollout) for ep_rollout in (_rollout_datas or [])))
                    if _rollout_datas else int(np.size(_rew_arr)) * max_episode_steps
                )
                if _x_traj and _xpos_traj:
                    from behavior_descriptor import (
                        aggregate_bd_from_episodes, resolve_control_dt,
                    )
                    try:
                        _bd_env_name = env_name or getattr(config, 'env_name', 'Hopper-v5')
                        obs_bd = aggregate_bd_from_episodes(
                            _x_traj, _xpos_traj, _bd_env_name,
                            dt=resolve_control_dt(env, _bd_env_name),
                        ).astype(np.float32)
                    except Exception:
                        obs_bd = None
            elif isinstance(_res, tuple) and len(_res) >= 2:
                average_reward, eval_steps = _res[0], _res[1]
            else:
                average_reward, eval_steps = float(_res), max_episode_steps
        except Exception:
            average_reward, eval_steps = _evaluate_actor(
                actor, env, max_action,
                memory=_eval_memory,
                n_episodes=eval_episodes, base_seed=seed_val, step_offset=offset,
                max_episode_steps=max_episode_steps,
            )
    else:
        average_reward, eval_steps = _evaluate_actor(
            actor, env, max_action,
            memory=_eval_memory,
            n_episodes=eval_episodes,
            base_seed=seed_val,
            step_offset=offset,
            max_episode_steps=max_episode_steps,
        )
    total_steps += eval_steps

    if return_updated_critics and return_bd:
        return average_reward, total_steps, c, critic_target, obs_bd
    if return_updated_critics:
        return average_reward, total_steps, c, critic_target
    if return_bd:
        return average_reward, total_steps, obs_bd
    return average_reward, total_steps


def _quick_evaluate(actor, env, max_action, seed, seed_offset_func,
                    max_episode_steps, n_episodes=3):
    offset = seed_offset_func(episode_idx=0)
    mean_score, eval_steps = _evaluate_actor(
        actor, env, max_action,
        n_episodes=n_episodes, base_seed=seed, step_offset=offset,
        max_episode_steps=max_episode_steps,
    )
    return mean_score, eval_steps


def _conservative_actor_update(
    actor, critic, memory, batch_size, anchor_actor,
    grad_clip_norm, bc_weight, anchor_weight, q_weight,
):
    """Conservative policy-improvement step.

    Maximizes critic advantage over the frozen pre-finetune anchor while
    keeping the actor close to both replay actions and the original actor,
    preventing pure Q maximization from exploiting unsupported replay regions.
    """
    states, _, actions, _, _ = memory.sample(batch_size)
    pred_actions = actor(states)

    q1, q2 = critic(states, pred_actions)
    q_pred = torch.min(q1, q2)

    with torch.no_grad():
        anchor_actions = anchor_actor(states)
        bq1, bq2 = critic(states, anchor_actions)
        q_anchor = torch.min(bq1, bq2)
        q_scale = q_anchor.abs().mean().clamp(min=1.0)

    q_advantage = (q_pred - q_anchor).mean() / q_scale
    bc_loss = F.mse_loss(pred_actions, actions)
    anchor_loss = F.mse_loss(pred_actions, anchor_actions)
    actor_loss = (
        -float(q_weight) * q_advantage
        + float(bc_weight) * bc_loss
        + float(anchor_weight) * anchor_loss
    )

    actor.optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), float(grad_clip_norm))
    actor.optimizer.step()

    return {
        'actor_loss': float(actor_loss.item()),
        'q_advantage': float(q_advantage.item()),
        'bc_loss': float(bc_loss.item()),
        'anchor_loss': float(anchor_loss.item()),
    }


def _plain_actor_update(actor, critic, memory, batch_size, grad_clip_norm):
    """Maximize min(Q1, Q2) without conservative penalties for the actor-update ablation."""
    states, _n_states, actions, _r, _d = memory.sample(batch_size)
    q1, q2 = critic(states, actor(states))
    q_min = torch.min(q1, q2)
    actor_loss = -q_min.mean()

    actor.optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), float(grad_clip_norm))
    actor.optimizer.step()

    return {
        'actor_loss': float(actor_loss.item()),
        'q_advantage': 0.0,
        'bc_loss': float('nan'),
        'anchor_loss': float('nan'),
    }


def _lr_scale_for_sparsity(sparsity, config):
    if sparsity < 0.2:
        return 0.10
    if sparsity < 0.5:
        return float(getattr(config, 'finetune_lr_scale_low', 0.20))
    if sparsity < 0.7:
        return float(getattr(config, 'finetune_lr_scale_mid', 0.15))
    if sparsity < 0.85:
        return float(getattr(config, 'finetune_lr_scale_high', 0.10))
    return float(getattr(config, 'finetune_lr_scale_ultra', 0.05))


def _reset_rng(seed):
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def td3_finetune(
    actor, env, memory, config, max_action, seed_offset_func,
    max_episode_steps=1000, total_update_steps=5000,
    update_frequency=1, sample_memory=None, write_memory=None,
    critic_td3=None, warmup_episodes=2,
    base_reward=float('-inf'), sparsity=0.0,
    eval_interval=500, eval_episodes=3,
    grad_clip_norm=0.5, lr_scale=None,
    online_rollout_interval=100, online_rollout_steps=200,
    exploration_noise_std=None,
    enable_actor_update=True,
    critic_warmup_steps_override=None,
    actor_sample_memory=None,
    conservative_actor_update=True,
    update_step_offset=0,
    direction_inject_fn=None,
    qd_probe_fn=None,
    qd_probe_interval=0,
    actor_min_replay_steps=None,
) -> Dict[str, Any]:
    """Refine with a fixed anchor, shared target lifetime and matching best-state snapshots.
    Executed update offsets preserve callback schedules across optional chunks."""
    if sample_memory is None:
        sample_memory = memory
    if write_memory is None:
        write_memory = memory
    if actor_sample_memory is None:
        actor_sample_memory = sample_memory

    c = critic_td3
    if c is None:
        raise ValueError("critic_td3 is required for TD3 finetune")
    c = c.to(device)

    if hasattr(env, 'n_envs') and getattr(env, 'n_envs', 1) > 1 and hasattr(env, 'single_env_view'):
        env = env.single_env_view(0)

    base_actor_lr = float(getattr(config, 'actor_lr', 1e-3))
    base_critic_lr = float(getattr(config, 'critic_lr', 1e-3))

    if lr_scale is None:
        lr_scale = _lr_scale_for_sparsity(float(sparsity), config)

    actual_actor_lr = base_actor_lr * lr_scale
    # Scale critic learning rates too for small sparse replay buffers.
    actual_critic_lr = base_critic_lr * min(lr_scale, 1.0)

    for pg in actor.optimizer.param_groups:
        pg['lr'] = actual_actor_lr
    for pg in c.optimizer.param_groups:
        pg['lr'] = actual_critic_lr

    total_steps = 0
    actor.to(device)
    actor.train()

    n_internal_evals = 0

    from networks import safe_deepcopy
    actor_target = safe_deepcopy(actor, device_target=device)
    critic_target = safe_deepcopy(c, device_target=device)
    anchor_actor = safe_deepcopy(actor, device_target=device)
    anchor_actor.eval()
    for p in anchor_actor.parameters():
        p.requires_grad_(False)
    seed = int(getattr(config, 'seed', 0))
    bs = int(getattr(config, 'batch_size', 256))
    tau = float(getattr(config, 'tau', 0.005))

    warmup_episodes = int(warmup_episodes)
    if exploration_noise_std is None:
        exploration_noise_std = float(
            getattr(config, 'sparse_policy_noise', None) or
            getattr(config, 'policy_noise', 0.2)
        )
    exploration_noise_std = float(exploration_noise_std)

    _prefill_graph = None
    if USE_CUDA_GRAPH_IN_ROLLOUT and torch.cuda.is_available():
        from networks import get_actor_cuda_graph
        _prefill_graph = get_actor_cuda_graph(actor)

    prefill_episodes = max(0, int(warmup_episodes))
    actor.eval()
    for episode in range(prefill_episodes):
        offset = seed_offset_func(episode_idx=episode)
        state = seed_env(env, seed, offset)
        if isinstance(state, tuple):
            state = state[0]
        for step in range(max_episode_steps):
            if _prefill_graph is not None:
                action = _prefill_graph(state)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = actor(state_tensor).cpu().numpy()[0]
            noise_std = exploration_noise_std * max_action
            exploration_noise = np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action + exploration_noise, -max_action, max_action)
            next_state, reward, terminated, truncated, _info = unpack_step(
                env.step(action), env)
            # Store termination in replay; either termination or truncation ends collection.
            write_memory.add((state, next_state, action, reward,
                              replay_terminal(terminated, truncated)))
            state = next_state
            total_steps += 1
            if bool(terminated) or bool(truncated):
                break
    actor.train()

    critic_warmup_steps = int(
        critic_warmup_steps_override
        if critic_warmup_steps_override is not None
        else getattr(config, 'finetune_critic_warmup_steps', 5000)
    )
    if warmup_episodes <= 0 and bool(getattr(config, 'finetune_skip_repeated_critic_warmup', True)):
        critic_warmup_steps = 0
    for _ in range(critic_warmup_steps):
        if sample_memory is None or int(sample_memory.size()) < bs:
            break
        c.update(sample_memory, bs, actor_t=actor_target, critic_t=critic_target)
    for param, target_param in zip(c.parameters(), critic_target.parameters()):
        target_param.data.copy_(param.data)

    _reset_rng(int(getattr(config, 'seed', 0)))

    initial_eval, initial_eval_steps = _quick_evaluate(actor, env, max_action, seed, seed_offset_func, max_episode_steps, eval_episodes)
    total_steps += initial_eval_steps
    n_internal_evals += 1
    best_reward = initial_eval
    # Critic-only calibration keeps its latest critic instead of restoring a frozen-actor snapshot.
    restore_best_actor = bool(enable_actor_update)
    restore_best_optimizer_state = bool(
        getattr(config, 'finetune_restore_optimizer_state', True)
    )
    best_actor_state = deepcopy(actor.state_dict())
    best_actor_opt_state = deepcopy(actor.optimizer.state_dict()) \
        if restore_best_optimizer_state and getattr(actor, 'optimizer', None) is not None else None
    best_critic_state = deepcopy(c.state_dict())
    best_critic_target_state = deepcopy(critic_target.state_dict())
    best_critic_opt_state = deepcopy(c.optimizer.state_dict()) \
        if restore_best_optimizer_state and getattr(c, 'optimizer', None) is not None else None
    # Track whether evaluation refreshed the snapshot before allowing restoration.
    best_snapshot_refreshed = False

    stopped_reason = "completed"
    update_steps_used = 0
    actor_update_skipped_insufficient_replay = 0
    last_actor_diag: Dict[str, float] = {}
    online_rollout_events = 0
    online_rollout_steps_collected = 0
    direction_inject_events = 0
    qd_probe_events = 0
    qd_checkpoints: List[Dict[str, Any]] = []
    grad_updates = 0

    def _record_qd_probe(probe, update_step):
        """Count every completed probe and snapshot retained proposals before actor rollback."""
        nonlocal total_steps, n_internal_evals, qd_probe_events
        if not probe:
            return
        total_steps += int(probe.get('eval_steps', 0) or 0)
        n_internal_evals += int(probe.get('n_evals', 0) or 0)
        if probe.get('keep'):
            qd_probe_events += 1
            qd_checkpoints.append({
                'actor_state': deepcopy(actor.state_dict()),
                'reward': float(probe.get('reward', float('-inf'))),
                'obs_descriptor': probe.get('obs_descriptor'),
                'update_step': int(update_step),
                'reason': str(probe.get('reason', '')),
            })

    bc_weight = float(getattr(config, 'finetune_actor_bc_weight', 0.05))
    anchor_weight = float(getattr(config, 'finetune_actor_anchor_weight_sparse', 0.5))
    q_weight = float(getattr(config, 'finetune_actor_q_weight', 1.0))
    # Use the caller-selected actor replay buffer to determine its minimum size.
    actor_min_replay_steps = max(
        bs,
        int(actor_min_replay_steps
            if actor_min_replay_steps is not None
            else getattr(config, 'finetune_actor_min_replay_steps', bs)),
    )
    if bool(getattr(config, 'finetune_skip_high_sparse_low_buffer', True)) and float(sparsity) >= 0.7:
        actor_min_replay_steps = max(
            actor_min_replay_steps,
            int(getattr(config, 'finetune_min_buffer_for_high_sparse', 1000)),
        )

    update_frequency = max(1, int(update_frequency))
    eval_interval = max(1, int(eval_interval))
    online_rollout_interval = int(online_rollout_interval)
    # Carry executed update offsets across chunks to preserve callback intervals.
    update_step_base = max(0, int(update_step_offset))

    def _restore(module, state, opt_state):
        module.load_state_dict(state)
        if opt_state is not None and getattr(module, 'optimizer', None) is not None:
            try:
                module.optimizer.load_state_dict(opt_state)
            except Exception:
                # An optimizer-state mismatch must not prevent weight restoration.
                pass

    if sample_memory is not None and int(sample_memory.size()) >= bs:
        for update_step in range(int(total_update_steps)):
            if int(sample_memory.size()) < bs:
                break

            # Callbacks use the cumulative executed critic-step count.
            global_step = update_step_base + update_step

            c.update(sample_memory, bs, actor_t=actor_target, critic_t=critic_target)
            grad_updates += 1
            if enable_actor_update and (update_step + 1) % update_frequency == 0:
                actor_update_memory = actor_sample_memory
                try:
                    actor_memory_size = (
                        int(actor_update_memory.size())
                        if actor_update_memory is not None and hasattr(actor_update_memory, 'size')
                        else 0
                    )
                except Exception:
                    actor_memory_size = 0
                if actor_memory_size < actor_min_replay_steps:
                    # Insufficient actor replay must not skip the rollout that replenishes it.
                    actor_update_skipped_insufficient_replay += 1
                else:
                    if conservative_actor_update:
                        last_actor_diag = _conservative_actor_update(
                            actor=actor, critic=c, memory=actor_update_memory,
                            batch_size=bs, anchor_actor=anchor_actor,
                            grad_clip_norm=float(grad_clip_norm),
                            bc_weight=bc_weight, anchor_weight=anchor_weight,
                            q_weight=q_weight,
                        )
                    else:
                        last_actor_diag = _plain_actor_update(
                            actor=actor, critic=c, memory=actor_update_memory,
                            batch_size=bs, grad_clip_norm=float(grad_clip_norm),
                        )
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for param, target_param in zip(c.parameters(), critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            update_steps_used += 1

            if online_rollout_interval > 0 and (global_step + 1) % online_rollout_interval == 0:
                rollout_collected = collect_finetune_rollout(
                    actor=actor, env=env, buffer=write_memory,
                    max_steps=online_rollout_steps,
                    exploration_std=0.20 * max_action,
                    max_action=max_action, config=config,
                )
                total_steps += rollout_collected
                online_rollout_events += 1
                online_rollout_steps_collected += int(rollout_collected)

            if direction_inject_fn is not None and online_rollout_interval > 0 \
                    and (global_step + 1) % online_rollout_interval == 0:
                actor_state_before_inject = deepcopy(actor.state_dict())
                try:
                    outcome = direction_inject_fn(
                        actor=actor, critic=c, critic_target=critic_target,
                        update_step=global_step + 1,
                    ) or {}
                except Exception as exc:
                    actor.load_state_dict(actor_state_before_inject)
                    outcome = {'error': type(exc).__name__}
                    logger.warning('[td3] direction injection failed: %s', exc)
                n_dirs = int(outcome.get('n_dirs', 0) or 0)
                # Retain the proposal snapshot before any rejection restores the training actor.
                _record_qd_probe(outcome.get('probe'), global_step + 1)
                if n_dirs > 0:
                    direction_inject_events += 1
                    total_steps += int(outcome.get('eval_steps', 0) or 0)
                    n_internal_evals += int(outcome.get('n_evals', 0) or 0)
                    accepted = bool(outcome.get('accepted', True))
                    if not accepted:
                        actor.load_state_dict(actor_state_before_inject)

            if qd_probe_fn is not None and qd_probe_interval > 0 \
                    and (global_step + 1) % qd_probe_interval == 0:
                try:
                    probe = qd_probe_fn(actor, global_step + 1) or {}
                except Exception as exc:
                    probe = {'error': type(exc).__name__}
                    logger.warning('[td3] QD probe failed: %s', exc)
                _record_qd_probe(probe, global_step + 1)

            if (update_step + 1) % eval_interval == 0:
                eval_reward, eval_steps = _quick_evaluate(
                    actor, env, max_action, seed, seed_offset_func,
                    max_episode_steps, eval_episodes,
                )
                total_steps += eval_steps
                n_internal_evals += 1
                actor.train()
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    best_snapshot_refreshed = True
                    if restore_best_actor:
                        best_actor_state = deepcopy(actor.state_dict())
                        if best_actor_opt_state is not None:
                            best_actor_opt_state = deepcopy(actor.optimizer.state_dict())
                        # Store matching actor and critic snapshots.
                        best_critic_state = deepcopy(c.state_dict())
                        best_critic_target_state = deepcopy(critic_target.state_dict())
                        if best_critic_opt_state is not None:
                            best_critic_opt_state = deepcopy(c.optimizer.state_dict())

    if restore_best_actor and not best_snapshot_refreshed and update_steps_used > 0:
        # Evaluate short stages once before deciding whether to restore a best snapshot.
        tail_reward, tail_steps = _quick_evaluate(
            actor, env, max_action, seed, seed_offset_func,
            max_episode_steps, eval_episodes,
        )
        total_steps += tail_steps
        n_internal_evals += 1
        if tail_reward > best_reward:
            best_reward = tail_reward
            best_actor_state = deepcopy(actor.state_dict())
            if best_actor_opt_state is not None:
                best_actor_opt_state = deepcopy(actor.optimizer.state_dict())
            best_critic_state = deepcopy(c.state_dict())
            best_critic_target_state = deepcopy(critic_target.state_dict())
            if best_critic_opt_state is not None:
                best_critic_opt_state = deepcopy(c.optimizer.state_dict())

    if restore_best_actor:
        _restore(actor, best_actor_state, best_actor_opt_state)
        _restore(c, best_critic_state, best_critic_opt_state)
        critic_target.load_state_dict(best_critic_target_state)
    # Critic-only stages retain calibrated critics, targets and Adam moments.

    _reset_rng(int(getattr(config, 'seed', 0)))

    final_reward, final_eval_steps = _quick_evaluate(actor, env, max_action, seed, seed_offset_func, max_episode_steps, eval_episodes)
    total_steps += final_eval_steps
    n_internal_evals += 1
    final_eval_reward = max(initial_eval, final_reward, best_reward)

    return {
        'actor': safe_deepcopy(actor, device_target='cpu'),
        'reward': float(final_eval_reward),
        'critic': safe_deepcopy(c, device_target='cpu'),
        'critic_target': safe_deepcopy(critic_target, device_target='cpu'),
        'total_steps': total_steps,
        'update_steps_used': update_steps_used,
        'grad_updates': int(grad_updates),
        'update_step_offset': int(update_step_base),
        'stopped_reason': stopped_reason,
        'lr_scale': lr_scale,
        'actual_actor_lr': float(actual_actor_lr),
        'actual_critic_lr': float(actual_critic_lr),
        'initial_eval': initial_eval,
        'final_eval': final_reward,
        'best_eval': best_reward,
        'n_evals': int(n_internal_evals),
        'actor_update_enabled': bool(enable_actor_update),
        'conservative_actor_update': bool(conservative_actor_update),
        'critic_calibrated_not_rolled_back': bool(not restore_best_actor),
        'actor_best_restored': bool(restore_best_actor),
        'online_rollout_events': int(online_rollout_events),
        'online_rollout_steps_collected': int(online_rollout_steps_collected),
        'direction_inject_events': int(direction_inject_events),
        'qd_probe_events': int(qd_probe_events),
        'qd_checkpoints': qd_checkpoints,
        'actor_min_replay_steps': int(actor_min_replay_steps),
        'actor_update_skipped_insufficient_replay': int(actor_update_skipped_insufficient_replay),
        'actor_bc_weight': float(bc_weight),
        'actor_anchor_weight': float(anchor_weight),
        'actor_q_weight': float(q_weight),
        'last_actor_diag': last_actor_diag,
    }


def collect_finetune_rollout(
    actor, env, buffer, max_steps, exploration_std, max_action, config,
) -> int:
    """Collect on-policy rollout transitions with optional CUDA Graph caching."""
    if buffer is None or max_steps <= 0:
        return 0
    collected = 0
    base_seed = int(getattr(config, 'seed', 0))
    seed_offset = int(np.random.randint(0, 10_000_000))
    state = seed_env(env, base_seed, seed_offset)
    if isinstance(state, tuple):
        state = state[0]
    actor.eval()

    graph_runner = None
    if USE_CUDA_GRAPH_IN_ROLLOUT and torch.cuda.is_available():
        from networks import get_actor_cuda_graph
        graph_runner = get_actor_cuda_graph(actor)

    if graph_runner is not None:
        actor_forward = graph_runner
    else:
        def actor_forward(s):
            t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                return actor(t).cpu().numpy()[0]

    while collected < max_steps:
        action_np = actor_forward(state)
        if exploration_std > 0.0:
            noise = np.random.normal(0.0, exploration_std, size=action_np.shape)
            action_np = np.clip(action_np + noise, -max_action, max_action)
        next_state, reward, terminated, truncated, _info = unpack_step(
            env.step(action_np), env)
        # Pure time limits bootstrap; real termination wins if both flags are set.
        buffer.add((state, next_state, action_np, float(reward),
                    replay_terminal(terminated, truncated)))
        collected += 1
        state = next_state
        if bool(terminated) or bool(truncated):
            reset_out = env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    return collected
