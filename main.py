import os
import argparse
import json
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import logging
import time
from typing import Optional, List, Dict, Any

from ES import sepCEM
from samplers import IMSampler
from experiment_utils import set_global_seeds
from sparsity_metrics import measure_pruned_actor
from archive_io_utils import make_json_serializable, mask_fingerprint
from archive_saver import (
    init_results_dirs,
    save_archive_snapshot as external_save_archive_snapshot,
)

from networks import Actor
from archive_manager import update_archive_bcs, sync_archive_buffers
from nbc import nearest_better_clustering
from t import SparseNetworkOptimizationConfig, SparseNetworkOptimizer
from qd_metrics import compute_qd_metrics, print_qd_metrics
from behavior_descriptor import (
    BD_SPEC_ID,
    GRID_BINS_PER_DIM,
    _norm_env_name,
    aggregate_bd_from_episodes,
    get_bd_spec,
    get_env_behavior_info,
    resolve_control_dt,
)
from mask_config import binarize_masks
import logz as _logz

from adaptive_archive_admission import (
    AdaptiveCriticArchiveAdmission,
    canonicalize_gait_bd,
    gait_bd_to_cell,
)
from critic_bank import SharedCriticBank
from retraining_scheduler import RetrainingScheduler
from env_step_accounting import EnvStepAccountant, attach_counter
from policy_identity import (
    StableEntryId,
    entry_stable_id,
    policy_fingerprint,
)


def _make_env_direct(config, env_name_attr: str = 'env_name'):
    raw_env_name = getattr(config, env_name_attr)
    env_name = _norm_env_name(raw_env_name)
    setattr(config, env_name_attr, env_name)
    max_episode_steps = int(getattr(config, 'max_episode_steps', 1000))
    import gymnasium as gym
    env = gym.make(env_name, max_episode_steps=max_episode_steps)
    input_dim = int(env.observation_space.shape[0])
    output_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])
    return env, input_dim, output_dim, max_action

ABLATION_FLAG_LABELS = (
    ("ablation_no_mask", "w/o Mask"),
    ("ablation_no_td3", "w/o TD3"),
    ("ablation_no_nbc", "w/o NBC"),
    ("ablation_shared_critic", "w/o Branch Critic"),
    ("ablation_no_memory", "w/o Memory Matching"),
    ("ablation_no_es", "w/o ES Emitter"),
    ("ablation_plain_actor_update", "w/o Conservative Update"),
)


def resolve_ablation_label(args_or_config: Any) -> str:
    """Resolve the current ablation flag to a short human-readable label."""
    for flag, label in ABLATION_FLAG_LABELS:
        try:
            if bool(getattr(args_or_config, flag)):
                return label
        except AttributeError:
            continue
    return "baseline"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for _name in ("jax", "jax._src", "jaxlib"):
    logging.getLogger(_name).setLevel(logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use repeated evaluations where genuine termination makes returns variable.
_TERMINATING_ENVS = frozenset({
    'Hopper-v5', 'Hopper-v4', 'Walker2d-v5', 'Walker2d-v4',
    'Ant-v5', 'Ant-v4',
})


def _evaluation_record(source: str, status: str, *, avg_reward=None,
                       obs_descriptor=None, rewards_arr=None,
                       failed: bool = False, env_steps: int = 0,
                       n_evals: int = 0, cached: bool = False,
                       from_cache_episodes: int = 0) -> Dict[str, Any]:
    """Return uniform result keys and a float32 rewards array, empty on failure."""
    return {
        'avg_reward': float(avg_reward) if avg_reward is not None else None,
        'obs_descriptor': obs_descriptor,
        'rewards_arr': np.asarray(
            [] if rewards_arr is None else rewards_arr, dtype=np.float32),
        'evaluation_status': status,
        'evaluation_failed': bool(failed),
        'env_steps': int(env_steps),
        'n_evals': int(n_evals),
        'source': source,
        'from_cache': bool(cached),
        'from_cache_episodes': int(from_cache_episodes),
    }


class HybridSparseOptimizer:

    def __init__(self, config: SparseNetworkOptimizationConfig, input_dim: int,
                 output_dim: int, max_action: float = 1.0):
        self.config = config
        set_global_seeds(int(getattr(config, 'seed', 42)))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_action = max_action
        self.BC = getattr(config, 'BC', 'gait')
        self.env_name = _norm_env_name(getattr(config, 'env_name', 'Hopper-v5'))
        self.es_optimizer = None

        self.bd_config = get_env_behavior_info(self.env_name)
        self.bd_names = self.bd_config.get('bd_names', ['v_norm', 'foot_duty'])
        self.bd_dim = int(self.bd_config.get('descriptor_dim', len(self.bd_names)))
        self.env_config = self.bd_config

        self.archive = []
        self.window = 50000
        self.env = None

        self.current_hidden_layers = [config.hidden_layer_size] * 2

        self.qd_n_bins_per_dim = int(getattr(config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM))
        if self.qd_n_bins_per_dim != GRID_BINS_PER_DIM:
            raise ValueError(
                f"{BD_SPEC_ID} requires a fixed {GRID_BINS_PER_DIM}x{GRID_BINS_PER_DIM} "
                f"grid; got qd_n_bins_per_dim={self.qd_n_bins_per_dim}. Changing the "
                "grid changes archive admission and makes coverage incomparable."
            )
        self.bd_spec = get_bd_spec(self.env_name)
        self.enable_qd_metrics = bool(getattr(config, 'enable_qd_metrics', True))
        self.qd_metrics_history = []

        self.ablation_no_mask = bool(getattr(config, 'ablation_no_mask', False))
        self.ablation_no_td3 = bool(getattr(config, 'ablation_no_td3', False))
        self.ablation_no_nbc = bool(getattr(config, 'ablation_no_nbc', False))
        self.ablation_shared_critic = bool(getattr(config, 'ablation_shared_critic', False))
        self.ablation_no_memory = bool(getattr(config, 'ablation_no_memory', False))
        self.ablation_no_es = bool(getattr(config, 'ablation_no_es', False))
        self.ablation_label = resolve_ablation_label(config)
        self.shared_critic: Optional[Any] = None
        self.shared_memory: Optional[Any] = None

        self.admission_controller = AdaptiveCriticArchiveAdmission(
            n_bins=self.qd_n_bins_per_dim,
            bd_replace_radius=float(getattr(config, 'archive_bd_replace_radius', 0.0)),
            min_replace_margin=float(getattr(config, 'archive_min_replace_margin', 0.0)),
        )

        self.actor = self.create_initial_network()

        self.actor_t = Actor(
            state_dim=int(input_dim),
            action_dim=int(output_dim),
            max_action=float(max_action),
            hidden_layers=list(self.current_hidden_layers),
            args=self.config,
            masks=None,
        ).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.current_global_iter: int = 0

        self.total_evaluations: int = 0
        # Count interactions once at env.step, including training and evaluation.
        self.step_accountant = EnvStepAccountant()
        self.stable_entry_ids = StableEntryId(start=1)
        self._step_accounting_warned = False

        self.critic_bank = SharedCriticBank(
            state_dim=int(input_dim),
            action_dim=int(output_dim),
            max_action=float(max_action),
            config=config,
            device=str(device),
        )

        self._last_q_std_grid = {}
        self.sparse_optimizer = SparseNetworkOptimizer(config, input_dim, output_dim, max_action)
        self.sparse_optimizer.actor = self.actor
        self.sparse_optimizer.sampler = None
        self.sparse_optimizer.main_optimizer = self
        self.memory_dense = self.sparse_optimizer.memory_dense
        self.sparse_optimizer.external_archive = self.archive
        self.sparse_optimizer.critic_bank = self.critic_bank

        self.current_tabular = {}
        self.archive_save_models = bool(getattr(config, 'archive_save_models', True))
        self.archive_save_critics = bool(getattr(config, 'archive_save_critics', True))
        self.archive_save_masks_npz = bool(getattr(config, 'archive_save_masks_npz', True))
        self.retraining_history: Dict[int, Dict[str, Any]] = {}
        self.retraining_scheduler: Optional[RetrainingScheduler] = None

        paths = init_results_dirs(
            config=config,
            env_name=self.env_name,
            bc_type=self.BC,
            input_dim=int(self.input_dim),
            output_dim=int(self.output_dim),
            max_action=float(self.max_action),
            ablation_label=getattr(self, 'ablation_label', 'baseline'),
        )
        self.results_dir = paths["results_dir"]
        self.archive_dir = paths["archive_dir"]
        self.archive_manifest_path = paths["archive_manifest_path"]
        self.training_log_path = paths["training_log_path"]
        _logz.configure_output_dir(self.results_dir)

    def create_initial_network(self):
        return Actor(
            state_dim=self.input_dim,
            action_dim=self.output_dim,
            max_action=self.max_action,
            hidden_layers=self.current_hidden_layers,
            args=self.config,
        ).to(device)

    @property
    def total_env_steps(self) -> int:
        """Real `env.step` calls so far (authoritative; see env_step_accounting)."""
        return int(self.step_accountant.total_env_steps)

    @total_env_steps.setter
    def total_env_steps(self, value) -> None:
        # Preserve an explicitly supplied total as a live-accounting offset.
        try:
            self.step_accountant.base_offset = max(
                int(self.step_accountant.base_offset), int(value))
        except Exception:
            pass

    def attach_step_counters(self, *envs) -> None:
        """Route every env this optimizer can touch through the accountant."""
        for tag, env in envs:
            attach_counter(env, self.step_accountant, source=str(tag))

    def env_budget_exhausted(self) -> bool:
        cap = int(getattr(self.config, 'max_env_steps', 0) or 0)
        return bool(cap > 0 and self.total_env_steps >= cap)

    def evaluate_actor(self, actor, env, memory=None, n_episodes=1,
                      max_episode_steps=None, return_rollout=False, return_bd_traj=False,
                      env_name=None, bc_type=None, ant_bc_mode='2d',
                      base_seed=None, step_offset=0, episode_seeds=None):
        from evaluation import evaluate_actor
        self.attach_step_counters((f'eval:{self.env_name}', env))
        return evaluate_actor(
            actor=actor,
            env=env,
            max_action=self.max_action,
            memory=memory,
            n_episodes=n_episodes,
            max_episode_steps=max_episode_steps,
            return_rollout=return_rollout,
            return_bd_traj=return_bd_traj,
            base_seed=(int(getattr(self.config, 'seed', 42)) if base_seed is None
                       else int(base_seed)),
            step_offset=int(step_offset),
            env_name=env_name or self.env_name,
            bc_type=bc_type or self.BC,
            ant_bc_mode=ant_bc_mode,
            episode_seeds=episode_seeds,
            step_counter=self.step_accountant,
        )

    def save_archive_snapshot(self, global_iter: int, tag: str = "",
                              qd_metrics: Optional[Any] = None,
                              n_interactions: int = 0,
                              elapsed_time: float = 0.0,
                              ablation_label: Optional[str] = None) -> Optional[str]:
        return external_save_archive_snapshot(
            archive=self.archive,
            global_iter=global_iter,
            tag=tag,
            archive_dir=self.archive_dir,
            archive_manifest_path=self.archive_manifest_path,
            archive_save_models=self.archive_save_models,
            archive_save_critics=self.archive_save_critics,
            archive_save_masks_npz=self.archive_save_masks_npz,
            qd_metrics=qd_metrics,
            n_interactions=n_interactions,
            elapsed_time=elapsed_time,
            bd_spec=self.bd_spec,
            ablation_label=ablation_label if ablation_label is not None else getattr(self, 'ablation_label', 'baseline'),
        )


    def log_z(self, key, val=None):
        if isinstance(key, dict) and val is None:
            for k, v in key.items():
                self.current_tabular[k] = make_json_serializable(v)
                _logz.log_tabular(k, make_json_serializable(v))
        else:
            self.current_tabular[key] = make_json_serializable(val)
            _logz.log_tabular(key, make_json_serializable(val))

    def _append_training_log_iteration(self, record: Dict[str, Any]) -> None:
        path = getattr(self, 'training_log_path', '')
        if not path:
            return
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                payload = {"iterations": []}
            if not isinstance(payload, dict):
                payload = {"iterations": []}
            iterations = payload.setdefault("iterations", [])
            if not isinstance(iterations, list):
                payload["iterations"] = iterations = []
            iterations.append(make_json_serializable(record))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("[training_log] append iteration failed: %s", exc)

    @staticmethod
    def _summarize_finetune_details(training_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not training_details:
            return {
                'count': 0,
                'submitted_count': 0,
                'admitted_count': 0,
                'mean_internal_improvement': 0.0,
                'mean_admission_improvement': 0.0,
            }
        internal = [float(d.get('improvement', 0.0)) for d in training_details]
        admission = [
            float(d.get('admission_improvement', 0.0))
            for d in training_details
            if d.get('admission_improvement') is not None
        ]
        return {
            'count': int(len(training_details)),
            'submitted_count': int(sum(1 for d in training_details if d.get('submitted_to_archive'))),
            'admitted_count': int(sum(1 for d in training_details if d.get('admitted_to_archive'))),
            'mean_internal_improvement': float(np.mean(internal)) if internal else 0.0,
            'mean_admission_improvement': float(np.mean(admission)) if admission else 0.0,
            'insufficient_replay_skips': int(sum(
                int(d.get('actor_update_skipped_insufficient_replay', 0))
                for d in training_details
            )),
            'critic_dirs_accepted': int(sum(int(d.get('n_dirs_accepted', 0)) for d in training_details)),
            'critic_dirs_reverted': int(sum(int(d.get('n_dirs_reverted', 0)) for d in training_details)),
        }

    def _flush_tabular(self):
        if not self.current_tabular:
            return
        _logz.dump_tabular()
        self.current_tabular = {}

    def _write_metrics_csv_row(self, global_iter: int, qd_metrics, n_interactions: int,
                               iter_wall_time: float, cum_wall_time: float):
        import csv
        csv_path = os.path.join(self.results_dir, 'metrics.csv')
        is_new = not os.path.exists(csv_path)
        # n_evals counts candidate-evaluation calls, not episodes or interactions.
        n_evals = int(getattr(self, 'total_evaluations', 0))
        row = {
            'global_iter': int(global_iter + 1),
            'env_steps': int(n_interactions),
            'n_evals': n_evals,
            'wall_time_sec': round(cum_wall_time, 2),
            'iter_wall_time_sec': round(iter_wall_time, 2),
            'archive_size': len(self.archive),
        }
        if qd_metrics is not None:
            row.update({
                'qd_score': round(qd_metrics.qd_score, 4),
                'coverage': round(qd_metrics.coverage, 6),
                'coverage_pct': round(qd_metrics.coverage_percent, 4),
                'best_reward': round(qd_metrics.best_reward, 4),
                'mean_reward': round(qd_metrics.mean_reward, 4),
                'std_reward': round(qd_metrics.std_reward, 4),
                'mean_sparsity': round(qd_metrics.mean_sparsity, 6),
                'cells_occupied': int(qd_metrics.cells_occupied),
                'total_cells': int(qd_metrics.total_cells),
                'duplicate_cell_count': int(qd_metrics.duplicate_cell_count),
                'generation': qd_metrics.iteration,
                'iter': qd_metrics.iteration,
            })
        else:
            row.update({
                'qd_score': '', 'coverage': '', 'coverage_pct': '',
                'best_reward': '', 'mean_reward': '', 'std_reward': '',
                'mean_sparsity': '', 'cells_occupied': '', 'total_cells': '',
                'valid_bd_count': '', 'zero_bd_count': '', 'invalid_bd_count': '',
                'duplicate_cell_count': '', 'coverage_arithmetic_ok': '',
            })
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"写 metrics.csv 失败: {e}")

    def update_archive(self):
        """Trim excess elites, retaining cell winners first, then synchronize archive state."""
        cap = int(self.window)
        if cap > 0 and len(self.archive) > cap:
            self._trim_archive(cap)
            self.admission_controller.rebuild_cache(self.archive)

        update_archive_bcs(self.archive)
        sync_archive_buffers(self.archive, self.sparse_optimizer)
        self.critic_bank.sync_with_archive(self.archive)

    def _trim_archive(self, cap: int) -> None:
        """Keep the best elite per occupied cell, then top up by reward."""
        cell_best: dict = {}
        for i, ind in enumerate(self.archive):
            bd = ind[2] if len(ind) > 2 else None
            if bd is None:
                continue
            bd_arr = np.asarray(bd, dtype=np.float32).flatten()
            cell = gait_bd_to_cell(bd_arr, self.qd_n_bins_per_dim)
            if cell is None:
                continue
            r = float(ind[1]) if len(ind) > 1 else float('-inf')
            prev = cell_best.get(cell)
            if prev is None or r > prev[0]:
                cell_best[cell] = (r, i)
        must_keep = {idx for _, idx in cell_best.values()}

        others = sorted(
            [(float(ind[1]) if len(ind) > 1 else float('-inf'), i)
             for i, ind in enumerate(self.archive) if i not in must_keep],
            key=lambda x: x[0], reverse=True,
        )
        budget = max(0, cap - len(must_keep))
        keep_idxs = sorted(list(must_keep) + [i for _, i in others[:budget]])

        self.archive[:] = [self.archive[i] for i in keep_idxs]

    def _verify_incumbent(self, cell, idx, stored_reward, challenger_reward, env=None):
        """Return (comparison reward, verified) using the challenger reset seeds.
        Update the incumbent stored return when remeasured."""
        if not bool(getattr(self.config, 'admission_verify_incumbent', True)):
            return stored_reward, False, 0.0
        if not bool(getattr(self.config, 'admission_reeval_enabled', True)):
            return stored_reward, False, 0.0
        margin = float(getattr(self.config, 'admission_verify_margin', 0.0) or 0.0)
        if margin <= 0.0:
            # An explicit absolute margin takes precedence over the relative margin.
            rel = float(getattr(self.config, 'admission_verify_margin_rel', 0.0) or 0.0)
            margin = rel * abs(float(stored_reward))
        if margin > 0.0 and float(challenger_reward) < float(stored_reward) - margin:
            return stored_reward, False, 0.0
        if idx is None or not (0 <= int(idx) < len(self.archive)):
            return stored_reward, False, 0.0
        crn_enabled = bool(getattr(self.config, 'admission_crn_enabled', True))
        incumbent = self.archive[int(idx)]
        network = incumbent[0] if incumbent is not None and len(incumbent) > 0 else None
        if network is None:
            return stored_reward, False, 0.0
        try:
            rewards_arr, _rollouts, _gait, _xtraj, _xpos = self.evaluate_actor(
                actor=network,
                env=env if env is not None else self.env,
                memory=None,
                n_episodes=self._admission_eval_episodes(),
                max_episode_steps=int(getattr(self.config, 'max_episode_steps', 1000)),
                return_rollout=False,
                return_bd_traj=True,
                env_name=self.env_name,
                bc_type='gait',
                ant_bc_mode=str(getattr(self.config, 'ant_bc_mode', '2d')),
                # Match the challenger reset seeds for a paired comparison.
                base_seed=(self._admission_crn_seed() if crn_enabled else -1),
                step_offset=0,
            )
            measured = float(np.mean(np.asarray(rewards_arr, dtype=np.float64)))
            if not np.isfinite(measured):
                return stored_reward, False, 0.0
        except Exception as exc:
            logger.warning('[archive] incumbent re-verification failed for cell %s: %s',
                           cell, exc)
            return stored_reward, False, 0.0

        if abs(measured - stored_reward) > 1e-6 and isinstance(incumbent, list) and len(incumbent) > 1:
            incumbent[1] = float(measured)
            cache = self.admission_controller._cell_best.get(cell)
            if cache is not None and int(cache[1]) == int(idx):
                self.admission_controller._cell_best[cell] = (float(measured), int(idx))
            # Separate corrected incumbent estimates from actual training regressions.
            return measured, True, max(0.0, float(stored_reward) - measured)
        return measured, True, 0.0

    def _admission_eval_episodes(self) -> int:
        """Admission episode count, respecting the minimum for terminating environments."""
        n = getattr(self.config, 'admission_eval_episodes', None)
        n = 5 if n is None else int(n)
        if getattr(self, 'env_name', '') in _TERMINATING_ENVS:
            floor = getattr(self.config, 'admission_min_episodes_terminating', None)
            floor = 1 if floor is None else int(floor)
            n = max(n, floor)
        return max(1, n)

    def _admission_crn_seed(self) -> int:
        """First common reset seed, constant across admission comparisons in an iteration."""
        base = int(getattr(self.config, 'seed', 1))
        return 1_000_000 + base * 1_000 + int(getattr(self, 'current_global_iter', 0) or 0)

    def _admission_episode_seeds(self, n_episodes: int) -> List[int]:
        """Return explicit admission reset seeds; None entries request unseeded resets."""
        n = max(0, int(n_episodes))
        if not bool(getattr(self.config, 'admission_crn_enabled', True)):
            return [None] * n
        first = self._admission_crn_seed()
        return [int(first + k) for k in range(n)]

    def _effective_screen_margin(self, incumbent: float) -> float:
        """Use an explicit absolute margin, otherwise relative margin times |return|."""
        abs_margin = float(getattr(self.config, 'admission_screen_margin', 0.0) or 0.0)
        if abs_margin > 0.0:
            return float(abs_margin)
        rel_margin = float(getattr(self.config, 'admission_screen_margin_rel', 0.0) or 0.0)
        return float(max(0.0, rel_margin * abs(float(incumbent))))

    def _screen_needs_reeval(self, bd, reward) -> bool:
        """Confirm candidates for empty cells or within the incumbent screening margin."""
        if not bool(getattr(self.config, 'admission_reeval_enabled', True)):
            return False
        if bd is None or reward is None or not np.isfinite(float(reward)):
            return True
        cell = gait_bd_to_cell(bd, int(getattr(self.config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM)))
        if cell is None:
            return True
        current_best = self.admission_controller._cell_best.get(cell)
        if current_best is None:
            return True
        margin = self._effective_screen_margin(float(current_best[0]))
        return float(reward) >= float(current_best[0]) - margin

    def _cache_is_valid(self, payload, pre_bd_valid, eval_eps, seeds) -> bool:
        """Require matching actor weights, sufficient episodes and the requested seed list."""
        if payload.get('_force_reeval', False):
            return False
        if pre_bd_valid is None:
            return False
        network = payload.get('network')
        if network is None:
            return False
        recorded = payload.get('policy_fingerprint')
        if recorded is None:
            # Unattributed cached measurements cannot be reused.
            return False
        # Fingerprint compact actor weights only; structural masks are already applied.
        current = policy_fingerprint(network)
        if current is None or str(current) != str(recorded):
            return False
        cached_rewards = np.asarray(
            payload.get('rewards_arr') if payload.get('rewards_arr') is not None else [],
            dtype=np.float64,
        ).reshape(-1)
        if cached_rewards.size < int(eval_eps):
            return False
        if not np.all(np.isfinite(cached_rewards)):
            return False
        raw_cached = list(payload.get('eval_seeds') or [])
        cached_seeds = [int(s) if s is not None else None for s in raw_cached]
        wanted = [int(s) if s is not None else None for s in list(seeds or [])]
        # Compare seed positions without dropping unseeded (None) entries.
        n_cmp = min(len(cached_seeds), len(wanted))
        for k in range(n_cmp):
            if wanted[k] is None:
                continue
            if cached_seeds[k] is None or cached_seeds[k] != wanted[k]:
                return False
        for k in range(n_cmp, len(wanted)):
            if wanted[k] is not None:
                return False
        return True


    def _batch_evaluate_candidates(self, candidates, env, eval_eps, max_steps):
        """Evaluate on explicit reset seeds; reuse valid cached episodes and fill missing ones."""
        if not candidates:
            return []
        out = []
        eval_eps = max(1, int(eval_eps))
        max_steps = max(1, int(max_steps))
        seeds = self._admission_episode_seeds(eval_eps)

        for payload in candidates:
            source = str(payload.get('source', 'unknown'))
            pre_bd, pre_bd_status = canonicalize_gait_bd(payload.get('obs_descriptor'))
            cached_rewards = np.asarray(
                payload.get('rewards_arr') if payload.get('rewards_arr') is not None else [],
                dtype=np.float64,
            ).reshape(-1)

            if self._cache_is_valid(payload, pre_bd, eval_eps, seeds):
                cached = cached_rewards.astype(np.float32)
                payload['_cached_hit'] = True
                out.append(_evaluation_record(
                    source, f'{source}_cached',
                    avg_reward=float(np.mean(cached)),
                    obs_descriptor=pre_bd,
                    rewards_arr=cached,
                    env_steps=0,                       # already paid for
                    n_evals=int(cached.size),
                    cached=True,
                    from_cache_episodes=int(cached.size),
                ))
                continue

            network = payload.get('network', None)
            if network is None:
                out.append(_evaluation_record(
                    source, 'failed:no_network', failed=True,
                ))
                continue

            # Cache reuse requires trajectories to recompute reward and BD over the same episodes.
            reuse = []
            cached_x, cached_xpos = [], []
            if (bool(getattr(self.config, 'admission_reeval_enabled', True))
                    and pre_bd is not None
                    and payload.get('policy_fingerprint') is not None
                    and str(policy_fingerprint(network))
                    == str(payload.get('policy_fingerprint'))
                    and cached_rewards.size > 0
                    and np.all(np.isfinite(cached_rewards))):
                cached_seeds = [int(s) for s in (payload.get('eval_seeds') or [])
                                if s is not None]
                px = list(payload.get('x_traj_list') or [])
                pp = list(payload.get('xpos_traj_list') or [])
                usable = 0
                for k in range(min(int(cached_rewards.size), int(eval_eps))):
                    if seeds and cached_seeds and k < len(cached_seeds) \
                            and k < len(seeds) and seeds[k] is not None \
                            and cached_seeds[k] != int(seeds[k]):
                        break
                    if k >= len(px) or k >= len(pp):
                        break
                    usable += 1
                if usable > 0:
                    reuse = list(cached_rewards[:usable])
                    cached_x = px[:usable]
                    cached_xpos = pp[:usable]

            needed = max(0, int(eval_eps) - len(reuse))
            try:
                eval_out = self.evaluate_actor(
                    actor=network,
                    env=env,
                    memory=None,
                    n_episodes=needed if needed else eval_eps,
                    max_episode_steps=max_steps,
                    return_rollout=False,
                    return_bd_traj=True,
                    env_name=self.env_name,
                    bc_type='gait',
                    ant_bc_mode=str(getattr(self.config, 'ant_bc_mode', '2d')),
                    episode_seeds=(list(seeds[len(reuse):]) if needed else seeds),
                )
                if not isinstance(eval_out, tuple) or len(eval_out) != 5:
                    raise ValueError(f'unexpected evaluation result: {type(eval_out)!r}')
                new_rewards, _, _, x_traj_list, xpos_traj_list = eval_out
                new_rewards = np.asarray(new_rewards, dtype=np.float32).reshape(-1)
                if new_rewards.size == 0 or not np.all(np.isfinite(new_rewards)):
                    raise ValueError('evaluation returned no finite rewards')
                env_steps = int(sum(max(0, len(traj) - 1) for traj in (x_traj_list or [])))
                rewards_arr = np.concatenate(
                    [np.asarray(reuse, dtype=np.float32), new_rewards]) \
                    if reuse else new_rewards
                all_x = cached_x + list(x_traj_list or [])
                all_xpos = cached_xpos + list(xpos_traj_list or [])
                obs_descriptor = aggregate_bd_from_episodes(
                    all_x, all_xpos, self.env_name,
                    dt=resolve_control_dt(env, self.env_name),
                )
                canonical_bd, bd_status = canonicalize_gait_bd(obs_descriptor)
                status = ('reevaluated' if not reuse
                          else f'topup_cached_{len(reuse)}of{eval_eps}')
                self.total_evaluations += 1
                if canonical_bd is None:
                    out.append(_evaluation_record(
                        source, f'invalid_bd:{bd_status}',
                        avg_reward=float(np.mean(rewards_arr)),
                        obs_descriptor=None,
                        rewards_arr=rewards_arr,
                        env_steps=env_steps,
                        n_evals=int(rewards_arr.size),
                    ))
                    continue
                out.append(_evaluation_record(
                    source, status,
                    avg_reward=float(np.mean(rewards_arr)),
                    obs_descriptor=canonical_bd,
                    rewards_arr=rewards_arr,
                    env_steps=env_steps,
                    n_evals=int(rewards_arr.size),
                    from_cache_episodes=int(len(reuse)),
                ))
                payload['eval_seeds'] = [int(s) for s in seeds if s is not None]
                payload['policy_fingerprint'] = policy_fingerprint(network)
                payload['rewards_arr'] = rewards_arr
                payload['obs_descriptor'] = canonical_bd
                payload['avg_reward'] = float(np.mean(rewards_arr))
                payload['x_traj_list'] = all_x
                payload['xpos_traj_list'] = all_xpos
            except Exception as exc:
                logger.warning(
                    '[candidate_eval] source=%s candidate=%s reevaluation failed: %s',
                    source,
                    payload.get('candidate_id') or payload.get('memory_id') or payload.get('buffer_id'),
                    exc,
                )
                out.append(_evaluation_record(
                    source, f'failed:{type(exc).__name__}', failed=True,
                ))
        return out


    def _candidate_warrants_confirmation(self, bd, reward) -> bool:
        """Confirm candidates that open, improve or approach the return of an occupied cell."""
        if bd is None or reward is None or not np.isfinite(float(reward)):
            return False
        if self._screen_needs_reeval(bd, float(reward)):
            return True
        cell = gait_bd_to_cell(
            bd, int(getattr(self.config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM)))
        if cell is None:
            return False
        best = self.admission_controller._cell_best.get(cell)
        if best is None:
            return True
        margin = self._effective_screen_margin(float(best[0]))
        return float(reward) >= float(best[0]) - margin

    def _confirm_admission_candidates(self, candidates, screen_results, env,
                                       confirm_eps, max_steps):
        """Confirm promising policies on the full seed list, recomputing return and BD."""
        out = []
        for i, (payload, screen) in enumerate(zip(candidates, screen_results)):
            if screen.get('evaluation_failed', False):
                out.append(screen)
                continue
            bd, _status = canonicalize_gait_bd(screen.get('obs_descriptor'))
            reward = screen.get('avg_reward')
            if not self._candidate_warrants_confirmation(bd, reward):
                screen['evaluation_status'] = f"screen_reject:{screen.get('evaluation_status', 'n/a')}"
                screen['screen_only'] = True
                out.append(screen)
                continue
            confirm = self._batch_evaluate_candidates(
                [payload], env, confirm_eps, max_steps)[0]
            if confirm.get('evaluation_failed', False):
                out.append(screen)
                continue
            confirm['screen_avg_reward'] = (
                float(reward) if reward is not None and np.isfinite(float(reward)) else None)
            confirm['screen_cell'] = (
                list(gait_bd_to_cell(
                    bd, int(getattr(self.config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM))))
                if bd is not None and gait_bd_to_cell(
                    bd, int(getattr(self.config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM))) is not None
                else None)
            confirm['confirmed'] = True
            out.append(confirm)
        return out

    def _evaluate_and_archive_candidates(self, es_candidates, mask_candidates, env, return_results: bool = False):
        all_cands = []
        if isinstance(es_candidates, (list, tuple)) and len(es_candidates) > 0:
            for candidate in es_candidates:
                candidate.setdefault('source', 'es')
                all_cands.append(candidate)
        if isinstance(mask_candidates, (list, tuple)) and len(mask_candidates) > 0:
            for candidate in mask_candidates:
                candidate.setdefault('source', 'mask')
                all_cands.append(candidate)
        if not all_cands:
            return [] if return_results else None

        if self.ablation_shared_critic:
            shared = self._ensure_shared_critic()
            for payload in all_cands:
                if payload.get('source') == 'finetune':
                    continue
                shared_cpu = deepcopy(shared)
                payload['critic'] = shared_cpu
                payload['critic_target'] = shared_cpu

        sources = {str(c.get('source', 'unknown')) for c in all_cands}
        stage = sources.pop() if len(sources) == 1 else 'mixed'
        confirm_eps = self._admission_eval_episodes()
        max_steps = int(getattr(self.config, 'max_episode_steps', 1000))
        screen_eps = int(getattr(self.config, 'admission_screen_episodes', 0) or 0)
        screen_enabled = bool(
            getattr(self.config, 'admission_screen_enabled', True)
        ) and 0 < screen_eps < confirm_eps

        if screen_enabled:
            screen_results = self._batch_evaluate_candidates(
                all_cands, env, screen_eps, max_steps)
            eval_results = self._confirm_admission_candidates(
                all_cands, screen_results, env, confirm_eps, max_steps)
        else:
            eval_results = self._batch_evaluate_candidates(
                all_cands, env, confirm_eps, max_steps)

        results_list = [] if return_results else None

        def _ensure_critic_module(value, candidate_id):
            if value is None or isinstance(value, nn.Module):
                return value
            if isinstance(value, dict):
                from networks import CriticTD3 as _CriticTD3
                module = _CriticTD3(
                    int(self.input_dim), int(self.output_dim),
                    float(self.max_action), self.config,
                )
                try:
                    module.load_state_dict(value)
                    return module.to(device)
                except Exception as exc:
                    logger.warning('[archive] critic decode failed for %s: %s', candidate_id, exc)
                    return None
            return None

        def _reject(result_record, action, **extra):
            """Record the rejection and optional result fields; the caller must then continue."""
            result_record['action'] = action
            result_record.update(extra)
            if return_results:
                results_list.append(result_record)

        for idx, payload in enumerate(all_cands):
            candidate_id = (
                payload.get('memory_id') or payload.get('candidate_id')
                or payload.get('buffer_id')
            )
            result_record = {
                'source': str(payload.get('source', stage)),
                'candidate_id': candidate_id,
                'archive_index': -1,
                'action': 'reject_error',
                'admitted': False,
            }
            eval_result = eval_results[idx]
            result_record['evaluation_status'] = eval_result.get('evaluation_status', 'unknown')

            if eval_result.get('evaluation_failed', False):
                _reject(result_record, 'reject_error')
                continue

            avg_reward = eval_result.get('avg_reward')
            obs_descriptor, bd_status = canonicalize_gait_bd(eval_result.get('obs_descriptor'))
            if obs_descriptor is None:
                _reject(result_record, 'reject_invalid_bd')
                continue
            cand_cell = gait_bd_to_cell(obs_descriptor, self.qd_n_bins_per_dim)
            if cand_cell is None:
                _reject(result_record, 'reject_invalid_bd')
                continue

            reward_value = float(avg_reward)
            if not np.isfinite(reward_value):
                _reject(result_record, 'reject_reward')
                continue

            current_best = self.admission_controller._cell_best.get(cand_cell)

            # Optional admission floor: a quantile of the current finite elite returns.
            if current_best is None:
                floor_q = getattr(
                    self.config, 'admission_quality_floor_quantile', None)
                if floor_q is not None and len(self.archive) >= 8:
                    existing = np.asarray(
                        [float(a[1]) for a in self.archive
                         if a is not None and len(a) > 1],
                        dtype=np.float64)
                    # Ignore nonfinite elite returns when computing the floor.
                    if existing.size >= 8 and np.all(np.isfinite(existing)):
                        try:
                            floor_val = float(np.quantile(existing, float(floor_q)))
                        except (ValueError, TypeError) as exc:
                            raise ValueError(
                                'admission_quality_floor_quantile must be in '
                                f'[0, 1], got {floor_q!r}') from exc
                        if reward_value < floor_val:
                            _reject(
                                result_record, 'reject_quality_floor',
                                reason=f'空格入档质量下限拒({reward_value:.2f}<{floor_val:.2f})',
                                avg_reward=reward_value,
                                quality_floor=floor_val,
                                cell=cand_cell,
                            )
                            continue

            cand_critic_q_std = payload.get('critic_q_std_at_candidate', None)
            try:
                cand_critic_q_std = float(cand_critic_q_std) if cand_critic_q_std is not None else None
            except Exception:
                cand_critic_q_std = None

            if current_best is not None:
                best_reward_in_cell, best_idx_in_cell = current_best
                (best_reward_in_cell, incumbent_verified,
                 refit_down) = self._verify_incumbent(
                    cand_cell, best_idx_in_cell, float(best_reward_in_cell),
                    float(reward_value), env=env,
                )
                # Apply the replacement margin to occupied cells as well.
                replace_margin = float(
                    getattr(self.config, 'archive_min_replace_margin', 0.0) or 0.0)
                if reward_value < float(best_reward_in_cell) + replace_margin:
                    _reject(
                        result_record, 'reject_same_cell',
                        reason='同cell低reward拒',
                        cell_best_reward=float(best_reward_in_cell),
                        avg_reward=reward_value,
                        cell=cand_cell,
                    )
                    continue
                replace_idx = int(best_idx_in_cell)
                admission_reason = '同cell高reward替换'
                info = {
                    'reason': admission_reason,
                    'cell': cand_cell,
                    'cell_best_reward': float(best_reward_in_cell),
                    'cell_replace_idx': int(best_idx_in_cell),
                }
            else:

                should_add, ctrl_replace_idx, ctrl_info = self.admission_controller.should_admit(
                    self.archive, obs_descriptor, reward_value,
                )
                if not should_add:
                    action = (
                        'reject_near_cell'
                        if ctrl_info.get('reason', '').startswith('近邻cell低reward拒')
                        else 'reject_same_cell'
                    )
                    _reject(
                        result_record, action,
                        reason=ctrl_info.get('reason', 'reject'),
                        cell_best_reward=ctrl_info.get('cell_best_reward'),
                        cell=ctrl_info.get('cell'),
                        replace_via_distance=bool(ctrl_info.get('replace_via_distance', False)),
                        bd_distance_to_replaced=ctrl_info.get('bd_distance_to_replaced'),
                        avg_reward=reward_value,
                    )
                    continue
                if ctrl_replace_idx is not None and ctrl_replace_idx >= 0:
                    replace_idx = int(ctrl_replace_idx)
                    admission_reason = ctrl_info.get('reason', '近邻cell高reward替换')
                else:
                    replace_idx = -1
                    admission_reason = '新cell入档'
                info = {
                    'reason': admission_reason,
                    'cell': ctrl_info.get('cell'),
                    'cell_best_reward': ctrl_info.get('cell_best_reward'),
                    'cell_replace_idx': ctrl_info.get('cell_replace_idx', -1),
                    'replace_via_distance': bool(ctrl_info.get('replace_via_distance', False)),
                    'bd_distance_to_replaced': ctrl_info.get('bd_distance_to_replaced'),
                }

            network = payload.get('network')
            if network is None:
                _reject(result_record, 'reject_error')
                continue

            masks = payload.get('masks')
            try:
                processed_masks = binarize_masks(masks, 0.5) if masks is not None else None
                # Parameter sparsity and FLOPs describe the compact actor, not mask-entry sparsity.
                sparsity_val, inference_metrics = measure_pruned_actor(
                    network,
                    int(self.input_dim),
                    int(self.output_dim),
                    dense_hidden=[int(getattr(self.config, 'hidden_layer_size', 256))] * 2,
                    max_sparsity=float(getattr(self.config, 'max_reported_sparsity', 0.90)),
                )
            except Exception as exc:
                logger.warning('[archive] candidate preparation failed for %s: %s', candidate_id, exc)
                _reject(result_record, 'reject_error')
                continue

            buffer_id = candidate_id if candidate_id is not None else (
                mask_fingerprint(masks) if masks is not None else None
            )
            critic = _ensure_critic_module(payload.get('critic'), candidate_id)
            critic_target = _ensure_critic_module(payload.get('critic_target'), candidate_id)

            if critic is None and getattr(self, 'critic_bank', None) is not None and hasattr(self.critic_bank, 'get_critic'):
                try:
                    critic = self.critic_bank.get_critic(0)
                    critic_target = critic
                except Exception as exc:
                    logger.warning('[archive] critic fallback failed for %s: %s',
                                   candidate_id, exc)
            from networks import safe_deepcopy
            cpu_device = torch.device('cpu')
            actor_archive = safe_deepcopy(network, device_target=cpu_device) if network is not None else None
            critic_archive = safe_deepcopy(critic, device_target=cpu_device) if critic is not None else None
            critic_target_archive = safe_deepcopy(critic_target, device_target=cpu_device) if critic_target is not None else None
            individual = [
                actor_archive,
                reward_value,
                obs_descriptor.copy(),
                masks,
                float(sparsity_val),
                make_json_serializable(inference_metrics),
                critic_archive,
                critic_target_archive,
                buffer_id,
                int(payload.get('n_evals', 1)) + int(eval_result.get('n_evals', 0)),
            ]

            admission_reason = str((info or {}).get('reason', 'unknown'))

            archive_meta = dict(payload.get('archive_meta') or {})
            archive_meta.update({
                'source': str(payload.get('source', stage)),
                'admission_reason': admission_reason,
                'coverage_seed': bool(admission_reason == '新cell入档'),
                'coverage_fitness': payload.get('coverage_fitness'),
                'coverage_is_new_cell': payload.get('coverage_is_new_cell'),
                'coverage_is_batch_duplicate': payload.get('coverage_is_batch_duplicate'),
                'coverage_novelty': payload.get('coverage_novelty'),
                'emitter_target_cell': payload.get('emitter_target_cell'),
                'emitter_target_hit': payload.get('emitter_target_hit'),
                'emitter_target_distance': payload.get('emitter_target_distance'),
                'emitter_base_cell': payload.get('emitter_base_cell'),
                'private_buffer_steps': payload.get('private_buffer_steps'),
                'critic_q_std_at_candidate': float(cand_critic_q_std) if cand_critic_q_std is not None else None,
                # Replacement creates a new identity and must not inherit training history.
                'stable_id': self.stable_entry_ids.mint(),
                # Weights-only fingerprints can be recomputed from actor.pth.
                'policy_fingerprint': policy_fingerprint(network),
                'eval_seeds': list(payload.get('eval_seeds') or []),
                'eval_episodes': int(eval_result.get('n_evals', 0)),
                'eval_status': str(eval_result.get('evaluation_status', '')),
                'cached': bool(eval_result.get('from_cache', False)),
                'qd_side_candidate': bool(payload.get('_qd_side_candidate', False)),
                'qd_side_reason': payload.get('_qd_side_reason'),
            })
            individual.append(make_json_serializable(archive_meta))

            if replace_idx is not None and int(replace_idx) >= 0 and not (
                0 <= int(replace_idx) < len(self.archive)
            ):
                logger.error(
                    '[archive] invalid replacement index for %s: %s (archive_size=%d)',
                    candidate_id, replace_idx, len(self.archive),
                )
                _reject(result_record, 'reject_error', reason='invalid_replace_idx')
                continue

            replacing = replace_idx is not None and 0 <= int(replace_idx) < len(self.archive)
            old_buffer_id = None
            new_idx = int(replace_idx) if replacing else len(self.archive)
            if replacing:
                old_buffer_id = self.archive[new_idx][8] if len(self.archive[new_idx]) > 8 else None
                self.archive[new_idx] = individual
                action = 'replace'
            else:
                self.archive.append(individual)
                action = 'append'

            try:
                if replacing:
                    self.admission_controller.commit_admit(
                        self.archive, new_idx,
                        cell=tuple(info.get('cell')) if info.get('cell') is not None else cand_cell,
                        reward=reward_value,
                        old_idx=int(replace_idx),
                    )
                else:
                    self.admission_controller.commit_admit(
                        self.archive, new_idx, cand_cell, reward_value,
                    )
            except Exception as exc:
                logger.warning('[archive] admission cache sync failed for %s: %s',
                               candidate_id, exc)

            if replacing:
                if (
                    isinstance(old_buffer_id, str)
                    and old_buffer_id
                    and old_buffer_id != buffer_id
                    and hasattr(self.sparse_optimizer, 'unregister_archive_mask_id')
                ):
                    self.sparse_optimizer.unregister_archive_mask_id(old_buffer_id)
                if self.critic_bank.has(new_idx):
                    self.critic_bank.remove(new_idx)
            if critic is not None:
                try:

                    self.critic_bank.register_individual_critic(new_idx, base_critic=critic)
                except Exception as exc:
                    logger.warning('[archive] critic registration failed for %s: %s', candidate_id, exc)
            if buffer_id is not None and hasattr(self.sparse_optimizer, 'register_archive_mask_id'):
                self.sparse_optimizer.register_archive_mask_id(buffer_id)

            result_record.update({
                'avg_reward': reward_value,
                'reward_array': eval_result['rewards_arr'].tolist(),
                'archive_index': new_idx,
                'action': action,
                'admitted': True,
                'sparsity': float(sparsity_val),
                'cell': cand_cell,
                'coverage_seed': bool(archive_meta.get('coverage_seed', False)),
            })
            if return_results:
                results_list.append(result_record)

        if return_results:
            return results_list
        return None

    def _es_optimization_phase(self, env):
        return self.sparse_optimizer._es_optimization_phase(env=env)

    def _mask_search_phase(self, env):
        self.sparse_optimizer.actor = self.actor
        candidates = self.sparse_optimizer._mask_optimization_phase()
        self.current_masks = None
        if hasattr(self.sparse_optimizer, 'current_masks'):
            self.sparse_optimizer.current_masks = None
        return candidates


    def _topup_finetune_targets(self, selected_agents, selected_set):
        """Reach ft_targets_per_iter with reward-stratified, weakest-eligible extra elites."""
        target_n = int(getattr(self.config, 'ft_targets_per_iter', 0) or 0)
        if target_n <= 0 or len(selected_agents) >= target_n:
            return selected_agents
        pool_size = max(1, int(getattr(self.config, 'ft_target_extra_pool', 3) or 3))
        n_archive = len(self.archive)
        if n_archive == 0:
            return selected_agents

        order = sorted(
            range(n_archive),
            key=lambda j: float(self.archive[j][1]) if len(self.archive[j]) > 1 else 0.0,
        )
        by_reward = [j for j in order if j not in selected_set]
        if not by_reward:
            return selected_agents

        need = target_n - len(selected_agents)
        cooldown = max(0, int(getattr(self.config, 'ft_bottom_cooldown_iters', 0)))
        n_skipped_cooldown = 0
        if cooldown > 0:
            fresh, cooling = [], []
            for j in by_reward:
                (cooling if self._elite_in_cooldown(j) else fresh).append(j)
            if fresh:
                n_skipped_cooldown = len(cooling)
                by_reward = fresh + cooling
        n_slots = max(1, min(len(by_reward), need))
        strata = np.array_split(np.asarray(by_reward, dtype=np.int64), n_slots)
        picked = []
        for depth in range(pool_size):
            for stratum in strata:
                if len(picked) >= need:
                    break
                if depth < len(stratum):
                    idx = int(stratum[depth])
                    if idx not in selected_set:
                        picked.append(idx)
                        selected_set.add(idx)
            if len(picked) >= need:
                break

        if not picked:
            for idx in by_reward:
                if len(picked) >= need:
                    break
                picked.append(int(idx))
                selected_set.add(int(idx))

        if picked:
            logger.info(
                '[retraining] fine-tune targets topped up %d -> %d (%d/%d elites, '
                'reward-stratified weakest-first, pool=%d, cooldown_deferred=%d)',
                len(selected_agents), len(selected_agents) + len(picked),
                len(selected_agents) + len(picked), n_archive, pool_size,
                n_skipped_cooldown,
            )
        return list(selected_agents) + picked

    def _elite_in_cooldown(self, archive_idx: int) -> bool:
        """Check recent unsuccessful refinement by stable policy identity."""
        cooldown = max(0, int(getattr(self.config, 'ft_bottom_cooldown_iters', 0)))
        if cooldown <= 0:
            return False
        stable_id = entry_stable_id(self.archive[archive_idx]) \
            if 0 <= int(archive_idx) < len(self.archive) else None
        key = f'sid:{stable_id}' if stable_id is not None else f'idx:{int(archive_idx)}'
        rec = self.retraining_history.get(key)
        if not isinstance(rec, dict):
            return False
        last_iter = int(rec.get('last_iter', -10 ** 6))
        recent = list(rec.get('recent_rewards') or [])
        if len(recent) >= 2 and recent[-1] > recent[-2] + 1e-9:
            return False
        return (int(getattr(self, 'current_global_iter', 0) or 0) - last_iter) < cooldown

    def _pick_bottom_pool_member(self, pool, pool_size, mode, selected_set, cooldown):
        """Select a tail member uniformly, by recency or by lowest return, respecting cooldown."""
        if not pool:
            return None
        candidates = list(pool[:max(1, int(pool_size))])
        candidates = [c for c in candidates if c[0] not in selected_set] or candidates
        if cooldown > 0:
            eligible = [c for c in candidates if not self._elite_in_cooldown(c[0])]
            if not eligible:
                return None
            candidates = eligible

        if mode == 'weakest':
            return candidates[0][0]
        if mode == 'rotation':
            last_iters = []
            for idx, _r in candidates:
                stable_id = entry_stable_id(self.archive[idx]) \
                    if 0 <= idx < len(self.archive) else None
                key = f'sid:{stable_id}' if stable_id is not None else f'idx:{int(idx)}'
                rec = self.retraining_history.get(key) or {}
                last_iters.append((int(rec.get('last_iter', -10 ** 6)), idx))
            return min(last_iters)[1]
        return int(random.choice(candidates)[0])

    def _clustering_and_training_phase(self, env):
        training_details: List[Dict[str, Any]] = []
        distance_percentile = float(self.config.nbc_distance_percentile)

        n = len(self.archive)
        archive_rewards = np.zeros(n, dtype=np.float32)
        archive_features = []

        sparse_archive_indices = []

        for i, agent in enumerate(self.archive):
            archive_rewards[i] = float(agent[1]) if len(agent) > 1 else 0.0

            if len(agent) > 2 and agent[2] is not None:
                feature = np.asarray(agent[2], dtype=np.float32).flatten()
            else:
                feature = np.array([archive_rewards[i]], dtype=np.float32)
            archive_features.append(feature)

            sparse_archive_indices.append(int(i))

        if not sparse_archive_indices:
            logger.info("[retraining] empty archive; skip NBC fine-tune")
            return []

        rewards = archive_rewards[np.asarray(sparse_archive_indices, dtype=np.int64)]
        features_list = [archive_features[i] for i in sparse_archive_indices]
        max_dim = max(f.shape[0] for f in features_list) if features_list else 1
        X = np.zeros((len(sparse_archive_indices), max_dim), dtype=np.float32)
        for i, f in enumerate(features_list):
            X[i, :f.shape[0]] = f

        if self.ablation_no_nbc:
            sorted_indices = sorted(
                range(len(rewards)),
                key=lambda j: float(rewards[j]), reverse=True,
            )
            top_k = min(5, len(sorted_indices))
            selected_agents = [int(sparse_archive_indices[j]) for j in sorted_indices[:top_k]]
            selected_set = set(selected_agents)
            cutoff = float(rewards[sorted_indices[top_k - 1]]) if selected_agents else 0.0
            logger.info(
                "[ablation] skip NBC, select top-%d by reward (cutoff=%.4f)",
                len(selected_agents),
                cutoff,
            )
        else:
            clustering_result = nearest_better_clustering(
                rewards=rewards,
                X=X,
                distance_percentile=distance_percentile,
            )

            labels = clustering_result.labels_
            unique_labels = np.unique(labels)

            max_agents = {}
            second_agents = {}
            bottom_agents = {}
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                if len(cluster_indices) == 0:
                    continue
                cluster_rewards = [
                    (int(sparse_archive_indices[int(local_idx)]), rewards[int(local_idx)])
                    for local_idx in cluster_indices
                ]
                cluster_rewards.sort(key=lambda x: x[1], reverse=True)
                max_agents[label] = cluster_rewards[0]
                second_agents[label] = cluster_rewards[1] if len(cluster_rewards) > 1 else cluster_rewards[0]
                bottom_agents[label] = list(reversed(cluster_rewards))

            if not max_agents:
                logger.info("[retraining] no NBC clusters selected for fine-tune")
                return []

            max_cluster = max(max_agents, key=lambda lb: max_agents[lb][1])

            selected_agents = []
            selected_set = set()
            # This fraction of clusters selects from the weakest-member pool.
            bottom_frac = float(getattr(self.config, 'ft_bottom_cluster_frac', 0.0))
            bottom_pool = max(1, int(getattr(self.config, 'ft_bottom_pool_size', 3)))
            bottom_mode = str(getattr(self.config, 'ft_bottom_select_mode', 'random')).lower()
            cooldown = max(0, int(getattr(self.config, 'ft_bottom_cooldown_iters', 0)))
            best_cluster_tail_frac = float(
                getattr(self.config, 'ft_best_cluster_tail_frac', 0.0))
            iter_now = int(getattr(self, 'current_global_iter', 0) or 0)
            n_bottom_clusters = 0
            n_cooldown_skips = 0
            for label in unique_labels:
                if label not in max_agents:
                    continue
                pool = bottom_agents.get(label) or []
                use_bottom = (
                    bottom_frac > 0.0
                    and len(pool) > 1
                    and random.random() < bottom_frac
                )
                if label == max_cluster:
                    # Occasionally allocate the best cluster slot to another member.
                    idx = max_agents[label][0]
                    if (len(pool) > 1
                            and best_cluster_tail_frac > 0.0
                            and random.random() < best_cluster_tail_frac):
                        tail = self._pick_bottom_pool_member(
                            pool, bottom_pool, bottom_mode, selected_set, cooldown)
                        if tail is not None:
                            idx = tail
                            n_bottom_clusters += 1
                elif use_bottom:
                    n_bottom_clusters += 1
                    idx = self._pick_bottom_pool_member(
                        pool, bottom_pool, bottom_mode, selected_set, cooldown)
                    if idx is None:
                        # If every tail member is cooling down, use the cluster top-two rule.
                        top1_idx = max_agents[label][0]
                        top2_idx = second_agents[label][0]
                        idx = top2_idx if top1_idx in selected_set else top1_idx
                        n_cooldown_skips += 1
                else:
                    top1_idx = max_agents[label][0]
                    top2_idx = second_agents[label][0]
                    if top1_idx in selected_set and top2_idx != top1_idx:
                        idx = top2_idx
                    elif random.random() < 0.5:
                        idx = top1_idx
                    else:
                        idx = top2_idx

                if idx not in selected_set:
                    selected_agents.append(idx)
                    selected_set.add(idx)

            if bottom_frac > 0.0:
                logger.info(
                    '[retraining] bottom-tail fine-tune active: %d/%d clusters '
                    'served from the weakest members (pool=%d, frac=%.2f, mode=%s, '
                    'cooldown=%d, best-cluster-tail=%.2f, cooldown-fallbacks=%d)',
                    n_bottom_clusters, len(unique_labels), bottom_pool, bottom_frac,
                    bottom_mode, cooldown, best_cluster_tail_frac, n_cooldown_skips,
                )

        selected_agents = self._topup_finetune_targets(selected_agents, selected_set)

        if self.retraining_scheduler is None:
            self.retraining_scheduler = RetrainingScheduler(self, config=self.config)

        archive_view = list(self.archive)
        if self.ablation_shared_critic:
            archive_view = self._inject_shared_critic(archive_view)
        if self.ablation_no_memory:
            archive_view = self._inject_shared_memory_id(archive_view)

        # Direct trainer calls use the same interaction accountant as admission.
        self.attach_step_counters((f'finetune:{self.env_name}', env))
        before = self.total_env_steps
        training_details = self.retraining_scheduler.run_for_iter(
            archive=archive_view,
            env=env,
            history=self.retraining_history,
            iter_idx=int(getattr(self, 'current_global_iter', 0)),
            nbc_selected_indices=selected_agents,
        )
        measured = self.total_env_steps - before
        details = [d for d in (training_details or []) if isinstance(d, dict)]
        reported = sum(int(d.get('total_steps', 0)) for d in details)
        self.step_accountant.add_grad_updates(
            sum(int(d.get('grad_updates', 0)) for d in details))
        logger.info(
            '[retraining] agent fine-tune env steps: measured=%d self-reported=%d '
            '(agents=%d)',
            measured, reported, len(training_details),
        )
        if reported > 0 and abs(measured - reported) > max(200, 0.25 * reported):
            if not self._step_accounting_warned:
                self._step_accounting_warned = True
                logger.warning(
                    '[step-accounting] fine-tune self-report (%d) and measured env '
                    'steps (%d) disagree; the measured count is authoritative',
                    reported, measured,
                )

        return training_details

    def _ensure_shared_critic(self):
        if self.shared_critic is not None:
            return self.shared_critic
        from networks import CriticTD3
        self.shared_critic = CriticTD3(
            int(self.input_dim), int(self.output_dim),
            float(self.max_action), self.config,
        ).to(device)
        return self.shared_critic


    def _inject_shared_critic(self, archive_list):
        shared = self._ensure_shared_critic()
        out = []
        for ind in archive_list:
            if not isinstance(ind, (list, tuple)) or len(ind) < 8:
                out.append(ind)
                continue
            replaced = list(ind)
            for slot in (6, 7):
                if slot < len(replaced):
                    replaced[slot] = deepcopy(shared).to(device)
            out.append(tuple(replaced))
        return out

    def _inject_shared_memory_id(self, archive_list):
        """Label shared replay in archive metadata; train_one performs the actual routing."""
        if not self.ablation_no_memory:
            return archive_list
        shared_id = "shared_memory_pool"
        out = []
        for ind in archive_list:
            if not isinstance(ind, (list, tuple)) or len(ind) < 9:
                out.append(ind)
                continue
            replaced = list(ind)
            replaced[8] = shared_id
            out.append(tuple(replaced))
        return out

    def optimize_with_hybrid_algorithm(self, env, M=30, env_single=None, env_mask=None, start_iter: int = 0):
        self.env = env
        self.control_dt = resolve_control_dt(env, self.env_name)
        self.bd_spec = get_bd_spec(self.env_name, control_dt=self.control_dt)
        self.sparse_optimizer.env = env
        # All main, ES and mask environments share one interaction accountant.
        self.attach_step_counters(
            ('main', env), ('es_single', env_single), ('mask', env_mask))
        if self.retraining_scheduler is None:
            self.retraining_scheduler = RetrainingScheduler(self, config=self.config)

        param_vector = torch.cat([p.flatten() for p in self.actor.parameters()]).detach().cpu().numpy()
        num_params = len(param_vector)
        self.es_optimizer = sepCEM(
            num_params=num_params,
            mu_init=param_vector,
            sigma_init=self.config.es_sigma_init,
            damp=self.config.es_damp,
            damp_limit=self.config.es_damp_limit,
            pop_size=self.config.es_pop_size,
            antithetic=True,
            parents=self.config.es_pop_size // 2,
            elitism=False
        )
        self.param_sampler = IMSampler(self.es_optimizer)
        self.sparse_optimizer.es_optimizer = self.es_optimizer
        self.sparse_optimizer.sampler = self.param_sampler
        self.memory_dense = self.sparse_optimizer.memory_dense
        self.sparse_optimizer.actor = self.actor

        iter_wall_start_global = time.time()
        global_iter = start_iter

        for global_iter in range(start_iter, M):
            iter_wall_start = time.time()
            self.current_global_iter = global_iter
            if hasattr(self.sparse_optimizer, 'global_count'):
                self.sparse_optimizer.global_count = global_iter
            self.current_tabular = {}

            # Check the interaction cap between iterations; active calls may overshoot.
            if self.env_budget_exhausted():
                logger.info(
                    '[budget] env-step cap %d reached at iter %d; stopping',
                    int(getattr(self.config, 'max_env_steps', 0)), global_iter,
                )
                break

            self.actor_t.load_state_dict(self.actor.state_dict())

            # Also check the interaction cap between complete phases.
            budget_hit = self.env_budget_exhausted()
            if budget_hit:
                logger.info(
                    '[budget] env-step cap %d reached at iter %d; skipping the '
                    'remaining phases',
                    int(getattr(self.config, 'max_env_steps', 0)), global_iter,
                )

            es_candidates = []
            if not budget_hit:
                if self.ablation_no_es:
                    logger.info('[ablation] skip ES/CEM emitter phase (--ablation_no_es)')
                else:
                    es_candidates = self._es_optimization_phase(env_single)
                if self.ablation_no_td3:
                    for cand in es_candidates:
                        cand.pop('critic_target', None)
                        cand['critic'] = None
                self._evaluate_and_archive_candidates(es_candidates, [], env)

            budget_hit = budget_hit or self.env_budget_exhausted()
            mask_phase_candidates = []
            if not budget_hit:
                if self.ablation_no_mask:
                    logger.info("[ablation] skip mask search phase")
                else:
                    mask_phase_candidates = self._mask_search_phase(
                        env_mask if env_mask is not None else env)
                if mask_phase_candidates:
                    self._evaluate_and_archive_candidates([], mask_phase_candidates, env)

            budget_hit = budget_hit or self.env_budget_exhausted()
            training_details = []
            if not budget_hit:
                self.env = env
                self.sparse_optimizer.env = env
                training_details = self._clustering_and_training_phase(env)
            self.update_archive()

            archive_rewards = [float(item[1]) for item in self.archive] if self.archive else []
            iter_wall_time = time.time() - iter_wall_start

            _n_interactions = 0
            _total = int(getattr(self, 'total_env_steps', 0))
            if _total > 0:
                _n_interactions = _total
            else:
                _spi = int(getattr(self.config, 'steps_per_iter', 0))
                _n_interactions = _spi * (global_iter + 1) if _spi > 0 else 0
            accounting = self.step_accountant.snapshot()

            qd_metrics = None
            metrics_dict = None
            finetune_summary = self._summarize_finetune_details(training_details)
            if self.enable_qd_metrics and len(self.archive) > 0:
                qd_behavior_dim = self.bd_dim
                qd_grid_resolution = [self.qd_n_bins_per_dim] * qd_behavior_dim

                qd_metrics = compute_qd_metrics(
                    archive=self.archive,
                    BC='gait',
                    behavior_dim=qd_behavior_dim,
                    grid_resolution=qd_grid_resolution,
                    iteration=global_iter + 1,
                    wall_time=iter_wall_time,
                    default_dim=qd_behavior_dim,
                )
                metrics_dict = qd_metrics.to_dict()
                metrics_dict['bd_spec'] = make_json_serializable(self.bd_spec)
                metrics_dict['coverage_pct'] = qd_metrics.coverage_percent
                metrics_dict['finetune_summary'] = make_json_serializable(finetune_summary)
                if len(self.qd_metrics_history) > 0:
                    prev_metrics = self.qd_metrics_history[-1]
                    prev_coverage = prev_metrics.get('coverage_ratio', prev_metrics.get('coverage', 0))
                    if prev_coverage > 1.0:
                        prev_coverage = prev_coverage / 100.0
                    metrics_dict['delta_qd_score'] = qd_metrics.qd_score - prev_metrics.get('qd_score', 0)
                    metrics_dict['delta_coverage'] = qd_metrics.coverage - prev_coverage
                    metrics_dict['delta_archive_size'] = qd_metrics.archive_size - prev_metrics.get('archive_size', 0)
                else:
                    metrics_dict['delta_qd_score'] = 0
                    metrics_dict['delta_coverage'] = 0
                    metrics_dict['delta_archive_size'] = 0
                metrics_dict['n_evaluations'] = int(self.total_evaluations)
                metrics_dict['env_steps'] = int(_n_interactions)
                metrics_dict['step_accounting'] = accounting
                metrics_dict['eval_episodes'] = int(accounting['eval_episodes'])
                metrics_dict['eval_calls'] = int(accounting['eval_calls'])
                metrics_dict['grad_updates'] = int(accounting['grad_updates'])
                metrics_dict['env_steps_semantics'] = (
                    'counted_at_env_step_boundary_v2')
                metrics_dict['global_iter'] = int(global_iter + 1)
                metrics_dict['wall_time_sec'] = float(iter_wall_time)
                metrics_dict['cumulative_wall_time_sec'] = float(time.time() - iter_wall_start_global)
                metrics_dict['ablation'] = getattr(self, 'ablation_label', 'baseline')
                self.qd_metrics_history.append(metrics_dict)

                critic_novelty = float(self.critic_bank.archive_coverage_novelty())

                self.log_z({
                    'qd_score': qd_metrics.qd_score,
                    'coverage_pct': qd_metrics.coverage_percent,
                    'archive_size': qd_metrics.archive_size,
                    'best_reward': qd_metrics.best_reward,
                    'mean_reward': qd_metrics.mean_reward,
                    'mean_sparsity': qd_metrics.mean_sparsity,
                    'critic_coverage_novelty': critic_novelty,
                    'n_evaluations': self.total_evaluations,
                    'iter_wall_time': iter_wall_time,
                    'bd_dim': qd_behavior_dim,
                    'bd_names': self.bd_names,
                    'cells_occupied': qd_metrics.cells_occupied,
                    'total_cells': qd_metrics.total_cells,
                    'duplicate_cell_count': qd_metrics.duplicate_cell_count,
                    'generation': global_iter,
                    'iter': global_iter,
                })
            else:
                display_iter = (global_iter + 1) * 10
                archive_max = float(np.max(archive_rewards)) if archive_rewards else 0.0
                archive_mean = float(np.mean(archive_rewards)) if archive_rewards else 0.0
                archive_size = len(self.archive)

                logger.info(
                    "iter=%s archive_size=%s archive_max=%.4f archive_mean=%.4f wall_time=%.1fs",
                    display_iter, archive_size, archive_max, archive_mean, iter_wall_time,
                )

            self.save_archive_snapshot(
                global_iter=global_iter,
                tag='post_train',
                qd_metrics=qd_metrics,
                n_interactions=_n_interactions,
                elapsed_time=iter_wall_time,
                ablation_label=getattr(self, 'ablation_label', 'baseline'),
            )


            self._append_training_log_iteration({
                'global_iter': int(global_iter + 1),
                'wall_time_sec': float(iter_wall_time),
                'cumulative_wall_time_sec': float(time.time() - iter_wall_start_global),
                'env_steps': int(_n_interactions),
                'qd_metrics': metrics_dict,
                'finetune_summary': finetune_summary,
                'finetune_details': training_details,
            })

            self._write_metrics_csv_row(global_iter, qd_metrics,
                                       _n_interactions, iter_wall_time, time.time() - iter_wall_start_global)

            self._flush_tabular()

        self.save_archive_snapshot(
            global_iter=min(global_iter+1, M),
            tag='final',
            ablation_label=getattr(self, 'ablation_label', 'baseline'),
        )
        total_wall_time = time.time() - iter_wall_start_global
        logger.info("训练完成，总耗时: %.1fs", total_wall_time)

        final_qd_metrics = None
        if self.enable_qd_metrics and self.archive:
            qd_behavior_dim = self.bd_dim
            qd_grid_resolution = [self.qd_n_bins_per_dim] * qd_behavior_dim

            final_qd_metrics = compute_qd_metrics(
                archive=self.archive,
                BC='gait',
                behavior_dim=qd_behavior_dim,
                grid_resolution=qd_grid_resolution,
                iteration=M,
                wall_time=total_wall_time,
                default_dim=qd_behavior_dim,
            )
            print("\n" + "=" * 60)
            print(f"=== 最终 QD 指标 (环境: {self.env_name}, 描述符: {self.bd_names}) ===")
            print_qd_metrics(final_qd_metrics)
            print("=" * 60)

        if self.archive:
            best_individual = max(self.archive, key=lambda x: x[1])
            self.actor = best_individual[0]
            self.sparse_optimizer.actor = self.actor

            rewards = [float(x[1]) for x in self.archive]
            mean_fitness = float(np.mean(rewards)) if rewards else 0.0
            std_fitness = float(np.std(rewards)) if rewards else 0.0
            best_fitness = float(np.max(rewards)) if rewards else 0.0

            return {
                'best_network': best_individual[0],
                'best_score': best_individual[1],
                'best_masks': best_individual[3] if len(best_individual) > 3 else None,
                'best_sparsity': best_individual[4] if len(best_individual) > 4 else 0.0,
                'best_inference_metrics': best_individual[5] if len(best_individual) > 5 else {},
                'archive_size': len(self.archive),
                'best_fitness': best_fitness,
                'mean_fitness': mean_fitness,
                'std_fitness': std_fitness,
                'actor': self.actor,
                'global_iterations_completed': M,
                'total_wall_time_sec': total_wall_time,
                'qd_metrics': final_qd_metrics.to_dict() if final_qd_metrics else {},
                'qd_metrics_history': self.qd_metrics_history,
            }
        else:
            return {
                'best_network': None,
                'best_score': 0.0,
                'archive_size': 0,
                'best_fitness': 0.0,
                'mean_fitness': 0.0,
                'std_fitness': 0.0,
                'actor': self.actor,
                'global_iterations_completed': M,
                'total_wall_time_sec': total_wall_time,
                'qd_metrics': {},
                'qd_metrics_history': self.qd_metrics_history,
            }


def parse_args(argv=None):
    """Run controls and common budgets; full hyperparameters are defined in t.py."""
    defaults = SparseNetworkOptimizationConfig()
    parser = argparse.ArgumentParser(description='SV-QD-RL training')
    options = (
        'M',
        'seed',
        'env_name',
        'max_env_steps',
        'hidden_layer_size',
        'es_pop_size',
        'es_max_generations',
        'mask_pop_size',
        'mask_generations',
        'mask_td3_update_steps',
        'finetune_td3_steps',
        'actor_lr',
        'critic_lr',
        'batch_size',
        'memory_size',
        'local_memory_size',
        'max_episode_steps',
    )
    for name in options:
        value = getattr(defaults, name)
        parser.add_argument(f'--{name}', type=type(value), default=value)
    for flag, label in ABLATION_FLAG_LABELS:
        parser.add_argument(f'--{flag}', action='store_true', help=label)
    parser.add_argument('--disable_mask_search', action='store_true')
    parser.add_argument('--disable_td3_finetune', action='store_true')
    return parser.parse_args(argv)




def config_from_args(args):
    config = SparseNetworkOptimizationConfig(**vars(args))
    config.env_name = _norm_env_name(config.env_name)
    return config


def main():
    args = parse_args()
    ablation_label = resolve_ablation_label(args)
    print(f"\n=== Ablation: {ablation_label} ===\n")

    config = config_from_args(args)

    set_global_seeds(config.seed)

    env, input_dim, output_dim, max_action = _make_env_direct(config)
    env_single, _, _, _ = _make_env_direct(config)
    env_mask, _, _, _ = _make_env_direct(config)

    hybrid_optimizer = HybridSparseOptimizer(config, input_dim, output_dim, max_action)

    hybrid_optimizer.optimize_with_hybrid_algorithm(
        env, M=config.M, env_single=env_single, env_mask=env_mask,
    )

    logger.info('Results saved in %s', hybrid_optimizer.results_dir)

    env.close()

if __name__ == "__main__":
    main()
