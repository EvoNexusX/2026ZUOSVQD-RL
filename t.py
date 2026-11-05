"""SparseNetworkOptimizer: ES-driven policy search + mask-GA structural search."""
import numpy as np
import torch
import logging
import time
from typing import List, Dict, Optional, Set, Any
from copy import deepcopy
from dataclasses import dataclass

from memory import Memory, MaskedReplay, snapshot_replay_buffer
from mask_config import get_mask_threshold
from networks import Actor, CriticTD3, safe_deepcopy
from evaluation import evaluate_actor
from es_phase import (
    set_actor_params, get_actor_params, apply_es_mean_to_actor,
)
from mask_phase import (
    initialize_mask_ga, run_mask_generation, cleanup_mask_buffers,
)
from critic_bank import SharedCriticBank
from td3_trainer import evaluate_actor_with_td3_update
from behavior_descriptor import (
    GRID_BINS_PER_DIM,
    aggregate_bd_from_episodes,
    get_env_behavior_info,
    resolve_control_dt,
)
from policy_identity import policy_fingerprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class SparseNetworkOptimizationConfig:
    """Training defaults; main.py exposes common settings as CLI overrides."""
    M: int = 50
    seed: int = 1
    BC: str = 'gait'
    nbc_distance_percentile: float = 98.0
    seed_offset_scale: float = 0.0

    ft_tiered_enabled: bool = True
    ft_bottom_cluster_frac: float = 0.5      # fraction of clusters served from the bottom
    ft_bottom_pool_size: int = 3             # how many of the worst count as "bottom"
    # Optional empty-cell floor, expressed as an elite-return quantile.
    admission_quality_floor_quantile: Optional[float] = None
    # Minimum repeated evaluations for environments with genuine termination.
    admission_eval_episodes: int = 5
    admission_min_episodes_terminating: int = 3
    # Archive candidates and contested incumbents use common reset seeds.
    admission_crn_enabled: bool = True
    # Cached estimates are remeasured independently of the CRN switch.
    admission_reeval_enabled: bool = True
    admission_verify_incumbent: bool = True
    # Incumbent verification margin in reward units; zero always verifies.
    admission_verify_margin: float = 0.0
    # Absolute candidate screening margin; zero falls back to the relative margin.
    admission_screen_margin: float = 0.0
    # Relative screening margin applies only when the absolute margin is zero.
    admission_screen_margin_rel: float = 0.15
    # Confirm promising candidates and recompute both reward and behavioral cell.
    admission_screen_enabled: bool = True
    admission_screen_episodes: int = 1
    # Tail selection: uniform random, least recently trained, or weakest.
    ft_bottom_select_mode: str = 'random'
    # Cooldown after refinement without improvement.
    ft_bottom_cooldown_iters: int = 3
    # Probability of assigning the best cluster slot to a non-best member.
    ft_best_cluster_tail_frac: float = 0.25
    # Continuous refinement preserves target networks, anchors and update counters.
    finetune_stage_b_chunked: bool = False
    # Restore matching Adam moments together with best network weights.
    finetune_restore_optimizer_state: bool = True
    # Intermediate policies may enter other cells without beating the parent return.
    finetune_qd_candidate_enabled: bool = False
    finetune_qd_probe_interval: int = 250
    finetune_qd_probe_episodes: int = 3
    finetune_qd_candidate_max: int = 2
    finetune_qd_probe_direction_inject: bool = True
    # Environment-step cap checked between calls; zero disables it.
    max_env_steps: int = 0
    # Zero selects one target per cluster; positive values top up from weak elites.
    ft_targets_per_iter: int = 0
    ft_target_extra_pool: int = 3
    # Optionally allocate budget by return headroom instead of reward ratio.
    ft_budget_by_headroom: bool = False
    ft_headroom_full_mult: float = 2.0
    ft_headroom_zero_mult: float = 0.7

    hidden_layer_size: int = 256

    es_pop_size: int = 10
    es_sigma_init: float = 0.001
    es_damp: float = 1e-3
    es_damp_limit: float = 1e-5
    es_max_generations: int = 10
    # ES reuse, gradient training and fresh-sample allocation.
    es_n_grad: int = 5
    es_start_steps: int = 10000
    es_n_noisy: int = 0
    es_n_eval: int = 1

    mask_pop_size: int = 100
    mask_generations: int = 10
    mask_elite_k_ratio: float = 0.04
    target_sparsity_max: float = 0.99
    mask_coverage_new_cell_bonus: float = 10.0
    mask_coverage_duplicate_penalty: float = 0.4
    # Cap the same-cell crowding penalty.
    mask_coverage_duplicate_saturation: int = 1
    mask_fitness_diversity_weight: float = 0.75
    mask_fitness_reward_weight: float = 0.25
    mask_fitness_sparsity_weight: float = 0.0
    mask_td3_update_steps: int = 500
    mask_td3_update_frequency: int = 2
    mask_td3_warmup_episodes: int = 1
    mask_td3_warmup_steps: int = 256
    mask_td3_batch_size: int = 128
    mask_td3_enable_actor_update: bool = True
    mask_td3_conservative_actor_update: bool = False
    mask_td3_actor_min_replay_steps: int = 512
    mask_actor_explore_lambda: float = 0.05

    mask_critic_ensemble_size: int = 2
    mask_critic_ensemble_subset_ratio: float = 0.7
    mask_critic_ensemble_steps: int = 30
    mask_bd_farness_weight: float = 1.5
    mask_bd_farness_max_dist: int = 50

    use_novelty_critic: bool = True
    novelty_critic_n_heads: int = 4
    novelty_critic_lr: float = 3e-4
    novelty_critic_bd_dim: int = 2
    novelty_critic_n_bins: int = 50

    mask_poly_eta: float = 20.0
    mask_poly_prob: float = 0.15
    mask_tournament_k: int = 3
    mask_cauchy_prob: float = 0.15
    mask_cauchy_scale: float = 0.5
    mask_critic_weight: float = 1.0
    mask_critic_cauchy_weight: float = 1.0
    mask_critic_mutation_weight: float = 0.5
    mask_init_random_ratio: float = 0.5
    mask_init_min_value: float = 0.3
    mask_exploit_ratio: float = 0.80
    mask_critic_emit_ratio: float = 0.20
    mask_reset_stuck_threshold: int = 3
    mask_reset_inject_count: int = 10
    mask_archive_learning_rate: float = 1.0
    mask_critic_novelty_weight: float = 0.0
    mask_critic_fitness_weight: float = 0.0
    mask_adaptive_cauchy: bool = True
    mask_adaptive_cauchy_max: float = 0.50

    mask_td3_critic_dense_ratio: float = 0.3
    mask_td3_critic_sparse_ratio: float = 0.7
    mask_eval_episodes: int = 1
    mask_eval_write_rollout: bool = True
    sparse_replay_max_buffers: int = 0
    sparse_replay_memory_size: int = 10000
    finetune_prefill_episodes: int = 10
    finetune_critic_warmup_steps: int = 5000
    finetune_warmup_episodes_sparse: int = 4
    finetune_skip_repeated_critic_warmup: bool = True
    finetune_critic_sample_dense_ratio: float = 0.3
    finetune_critic_sample_sparse_ratio: float = 0.7
    finetune_lr_scale_low: float = 0.20
    finetune_lr_scale_mid: float = 0.15
    finetune_lr_scale_high: float = 0.10
    finetune_lr_scale_ultra: float = 0.05

    finetune_td3_steps: int = 5000
    finetune_td3_steps_dense: int = 8000
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 256
    tau: float = 0.005
    warmup_episodes: int = 4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    sparse_policy_noise: float = 0.1
    sparse_noise_clip: float = 0.3
    policy_delay: int = 2
    layer_norm: bool = False
    discount: float = 0.99
    local_memory_size: int = 1000000

    env_name: str = "Hopper-v5"
    max_episode_steps: int = 1000
    memory_size: int = 1000000

    enable_qd_metrics: bool = True
    qd_n_bins_per_dim: int = GRID_BINS_PER_DIM
    archive_save_models: bool = True
    archive_save_critics: bool = True
    archive_save_masks_npz: bool = True
    archive_bd_replace_radius: float = 0.0
    archive_min_replace_margin: float = 0.0

    finetune_eval_episodes_sparse: int = 2
    finetune_eval_episodes_dense: int = 3
    finetune_warmup_episodes_dense: int = 0
    retraining_critic_dir_inject: bool = True
    retraining_critic_dir_alpha: float = 0.05
    retraining_critic_dir_interval: int = 250

    finetune_budget_max_mult: float = 2.5
    finetune_diversity_radius: int = 2
    finetune_diversity_low_density_threshold: int = 2
    finetune_diversity_high_density_threshold: int = 6
    finetune_diversity_mult_isolated: float = 1.8
    finetune_diversity_mult_sparse: float = 1.4
    finetune_use_critic_std_bonus: bool = True
    finetune_critic_std_ref_percentile: float = 90.0

    finetune_skip_high_sparse_low_buffer: bool = True
    finetune_min_buffer_for_high_sparse: int = 1000
    finetune_high_sparse_low_buffer_mult: float = 0.3
    finetune_fallback_on_policy_episodes: int = 2
    finetune_private_replay_min_steps: int = 256
    finetune_actor_min_replay_steps: int = 256
    finetune_private_replay_noise: float = 0.1

    mask_update_ensemble_before_ask: bool = True
    critic_ensemble_K: int = 5
    critic_lambda_explore: float = 0.05
    critic_update_min_buffer: int = 256

    mask_use_critic_field_during_eval: bool = True
    mask_cfdi_eval_alpha: float = 0.02
    mask_cfdi_eval_noise: float = 0.005
    mask_cfdi_eval_batch: int = 128
    mask_cfdi_n_neighbors: int = 3

    disable_mask_search: bool = False
    disable_td3_finetune: bool = False

    ablation_no_mask: bool = False
    ablation_no_td3: bool = False
    ablation_no_nbc: bool = False
    ablation_shared_critic: bool = False
    ablation_no_memory: bool = False
    ablation_no_es: bool = False
    ablation_plain_actor_update: bool = False


class SparseNetworkOptimizer:
    """ES phase (CEM over actor params) + mask phase (GA over per-layer masks)."""

    def __init__(self, config, input_dim, output_dim, max_action=1.0):
        self.config = config
        self._start_time = time.time()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_action = max_action
        self.env_name = getattr(config, 'env_name', 'HalfCheetah-v5')

        bd_info = get_env_behavior_info(self.env_name)
        self.bd_dim = bd_info['descriptor_dim']

        self.current_best_network = None
        self.current_best_score = -float('inf')
        self.current_best_masks = None
        self.shared_parameters = None

        self.current_hidden_layers = [config.hidden_layer_size] * 2

        self.actor = None
        self.actor_t = None  # ES target actor, lazily created
        self.memory = None
        self.memory_dense = None
        self.memory_sparse = None
        self.current_memory_mode = None
        self.sampler = None
        self.env = None

        self.es_optimizer = None
        self.mask_es_optimizer = None
        self.mask_sampler = None
        self.layer_sizes = None

        self.old_fitness = []
        self.old_n_steps = []
        self.old_n_start = []
        self.old_es = []
        self.actor_steps = 0
        self.total_steps = 0
        self.reused_steps = 0
        self.old_mask_fitness = []

        self._init_replay_buffers()
        self._archive_mask_ids: Set[str] = set()
        self._archive_replay_cache: Dict[str, Any] = {}

        self.critic = None
        self.critic_bank = None

        self.param_mask = None
        self.masked_param_indices = None
        self.full_param_template = None
        self.current_mask_threshold = get_mask_threshold()

        self.M = int(max(1, getattr(config, 'M', getattr(config, 'mask_generations', 1))))
        self.global_count = 0

    def _init_replay_buffers(self):
        """memory_dense = ES replay; memory_sparse = per-mask-id replay."""
        capacity = int(getattr(self.config, 'memory_size', 1_000_000))
        sparse_capacity = int(getattr(self.config, 'sparse_replay_memory_size', 10_000))
        sparse_capacity = max(1024, min(sparse_capacity, capacity))
        if self.memory_dense is None:
            self.memory_dense = Memory(capacity, self.input_dim, self.output_dim)
        if self.memory_sparse is None:
            self.memory_sparse = MaskedReplay(
                base_buffer_cls=Memory,
                memory_size=sparse_capacity,
                state_dim=self.input_dim,
                action_dim=self.output_dim,
                base_kwargs=None,
                default_mask_id="default",
                max_buffers=max(
                    int(getattr(self.config, 'sparse_replay_max_buffers', 0) or 0),
                    int(getattr(self.config, 'mask_pop_size', 50)) + 1,
                ),
            )
        self.memory = self.memory_dense
        self.current_memory_mode = 'dense'

    def use_dense_memory(self):
        self.memory = self.memory_dense
        self.current_memory_mode = 'dense'

    def use_sparse_memory(self):
        self.memory = self.memory_sparse
        self.current_memory_mode = 'sparse'

    def get_sparse_memory(self, mask_id, create_if_missing=True):
        if self.memory_sparse is None:
            self._init_replay_buffers()
        if mask_id is None:
            return self.memory_sparse
        mask_id = str(mask_id)
        if hasattr(self.memory_sparse, 'get_buffer') and self.memory_sparse.get_buffer(mask_id) is None:
            cached = getattr(self, '_archive_replay_cache', {}).get(mask_id)
            if cached is not None:
                restored = self._restore_sparse_memory_snapshot(mask_id, cached)
                if restored is not None:
                    return restored
            if not create_if_missing:
                return None
            self.clone_sparse_memory(None, mask_id)
        if hasattr(self.memory_sparse, 'set_mask'):
            self.memory_sparse.set_mask(mask_id)
        if hasattr(self.memory_sparse, 'get_buffer'):
            return self.memory_sparse.get_buffer(mask_id)
        return self.memory_sparse

    def register_sparse_memory(self, mask_id, buffer) -> None:
        if mask_id is None or buffer is None:
            return
        if self.memory_sparse is None:
            self._init_replay_buffers()
        mid = str(mask_id)
        if hasattr(self.memory_sparse, '_buffers'):
            self.memory_sparse._buffers[mid] = buffer
            self.memory_sparse._current_id = mid
        if mid in getattr(self, '_archive_mask_ids', set()):
            snapshot = snapshot_replay_buffer(buffer)
            if snapshot is not None:
                if not hasattr(self, '_archive_replay_cache') or self._archive_replay_cache is None:
                    self._archive_replay_cache = {}
                self._archive_replay_cache[mid] = snapshot

    def clone_sparse_memory(self, src_id, new_id):
        if self.memory_sparse is None:
            self._init_replay_buffers()
        new_id = str(new_id)
        src_buf = None
        if src_id is not None:
            if hasattr(self.memory_sparse, 'get_buffer'):
                src_buf = self.memory_sparse.get_buffer(str(src_id))
        if hasattr(self.memory_sparse, 'get_buffer') and self.memory_sparse.get_buffer(new_id) is not None:
            return self.memory_sparse.get_buffer(new_id)
        if hasattr(self.memory_sparse, 'set_mask'):
            self.memory_sparse.set_mask(new_id)
        dst_buf = self.memory_sparse.get_buffer(new_id) if hasattr(self.memory_sparse, 'get_buffer') else None
        if dst_buf is None:
            return None
        if src_buf is not None:
            self._copy_replay_buffer(src_buf, dst_buf)
        return dst_buf

    def register_archive_mask_id(self, mask_id) -> None:
        if mask_id is None:
            return
        if self.memory_sparse is None:
            self._init_replay_buffers()
        if not hasattr(self, '_archive_mask_ids') or self._archive_mask_ids is None:
            self._archive_mask_ids = set()
        self._archive_mask_ids.add(str(mask_id))
        if not hasattr(self, '_archive_replay_cache') or self._archive_replay_cache is None:
            self._archive_replay_cache = {}
        if hasattr(self.memory_sparse, 'get_buffer'):
            buf = self.memory_sparse.get_buffer(str(mask_id))
            if buf is not None:
                snapshot = snapshot_replay_buffer(buf)
                if snapshot is not None:
                    self._archive_replay_cache[str(mask_id)] = snapshot

    def unregister_archive_mask_id(self, mask_id) -> None:
        if mask_id is None:
            return
        if self.memory_sparse is None:
            self._init_replay_buffers()
        if not hasattr(self, '_archive_mask_ids') or self._archive_mask_ids is None:
            return
        self._archive_mask_ids.discard(str(mask_id))
        if hasattr(self, '_archive_replay_cache') and self._archive_replay_cache is not None:
            self._archive_replay_cache.pop(str(mask_id), None)
        if hasattr(self.memory_sparse, 'release_ids'):
            self.memory_sparse.release_ids([str(mask_id)])

    def prune_sparse_memories(self, keep_ids):
        """Snapshot archive-owned replays to CPU; drop live GPU buffers."""
        if self.memory_sparse is None:
            self._init_replay_buffers()
        if keep_ids is None:
            keep_ids = []
        keep_set = set(map(str, keep_ids))
        if hasattr(self, '_archive_mask_ids') and self._archive_mask_ids:
            keep_set |= set(map(str, self._archive_mask_ids))
        if not hasattr(self, '_archive_replay_cache') or self._archive_replay_cache is None:
            self._archive_replay_cache = {}
        if hasattr(self.memory_sparse, 'get_buffer'):
            for mid in list(keep_set):
                buf = self.memory_sparse.get_buffer(str(mid))
                if buf is None:
                    continue
                snapshot = snapshot_replay_buffer(buf)
                if snapshot is not None:
                    self._archive_replay_cache[str(mid)] = snapshot
        if hasattr(self.memory_sparse, 'release_ids') and keep_set:
            self.memory_sparse.release_ids(keep_set)
        if hasattr(self.memory_sparse, 'clear_except'):
            self.memory_sparse.clear_except([])
        self._archive_replay_cache = {
            mid: buf for mid, buf in self._archive_replay_cache.items()
            if mid in keep_set
        }

    def _restore_sparse_memory_snapshot(self, mask_id, snapshot):
        if snapshot is None:
            return None
        if self.memory_sparse is None:
            self._init_replay_buffers()
        mid = str(mask_id)
        if hasattr(self.memory_sparse, 'set_mask'):
            self.memory_sparse.set_mask(mid)
        dst = self.memory_sparse.get_buffer(mid) if hasattr(self.memory_sparse, 'get_buffer') else None
        if dst is None:
            return None
        try:
            with torch.no_grad():
                for name in ('states', 'actions', 'n_states', 'rewards', 'dones'):
                    if not hasattr(dst, name) or name not in snapshot:
                        continue
                    src_t = snapshot[name]
                    dst_t = getattr(dst, name)
                    n = min(int(src_t.shape[0]), int(dst_t.shape[0]))
                    dst_t[:n].copy_(src_t[:n].to(dst_t.device))
            if hasattr(dst, 'pos'):
                dst.pos = int(snapshot.get('pos', 0)) % max(1, int(getattr(dst, 'memory_size', 1)))
            if hasattr(dst, 'left_ptr'):
                dst.left_ptr = int(snapshot.get('left_ptr', 0)) % max(1, int(getattr(dst, 'memory_size', 1)))
            if hasattr(dst, 'full'):
                dst.full = bool(snapshot.get('full', False))
            return dst
        except Exception as exc:
            logger.warning("[memory] restore sparse replay snapshot failed for %s: %s", mid, exc)
            return dst

    def _copy_replay_buffer(self, src, dst) -> None:
        if src is None or dst is None:
            return
        try:
            with torch.no_grad():
                if hasattr(src, 'states') and hasattr(dst, 'states'):
                    size = min(src.states.shape[0], dst.states.shape[0])
                    dst.states[:size].copy_(src.states[:size])
                if hasattr(src, 'actions') and hasattr(dst, 'actions'):
                    size = min(src.actions.shape[0], dst.actions.shape[0])
                    dst.actions[:size].copy_(src.actions[:size])
                if hasattr(src, 'n_states') and hasattr(dst, 'n_states'):
                    size = min(src.n_states.shape[0], dst.n_states.shape[0])
                    dst.n_states[:size].copy_(src.n_states[:size])
                if hasattr(src, 'rewards') and hasattr(dst, 'rewards'):
                    size = min(src.rewards.shape[0], dst.rewards.shape[0])
                    dst.rewards[:size].copy_(src.rewards[:size])
                if hasattr(src, 'dones') and hasattr(dst, 'dones'):
                    size = min(src.dones.shape[0], dst.dones.shape[0])
                    dst.dones[:size].copy_(src.dones[:size])

            if hasattr(dst, 'pos') and hasattr(src, 'pos'):
                dst.pos = int(src.pos) % int(getattr(dst, 'memory_size', max(1, dst.states.shape[0])))
            if hasattr(dst, 'full') and hasattr(src, 'full'):
                dst.full = bool(src.full)
            if hasattr(dst, 'left_ptr'):
                if hasattr(src, 'left_ptr'):
                    dst.left_ptr = int(src.left_ptr) % int(getattr(dst, 'memory_size', max(1, dst.states.shape[0])))
                else:
                    if bool(getattr(src, 'full', False)):
                        dst.left_ptr = int(getattr(src, 'pos', 0)) % int(getattr(dst, 'memory_size', max(1, dst.states.shape[0])))
                    else:
                        dst.left_ptr = 0
        except Exception as e:
            logger.warning(f"复��� replay buffer 失败: {e}")

    def get_memory_for_actor(self, actor):
        has_masks = (actor is not None) and (getattr(actor, 'masks', None) is not None)
        return self.memory_sparse if has_masks else self.memory_dense

    def _compute_seed_offset(self, episode_idx=0) -> int:
        """Return an evaluation seed offset including the episode index."""
        scale = float(getattr(self.config, 'seed_offset_scale', 1.0))
        steps = max(0, int(self.total_steps))
        episode = max(0, int(episode_idx))
        offset = float(steps) * scale + float(episode) * 1000003.0
        return int(max(0, round(offset)))

    def evaluate_actor(self, actor, env, memory=None, n_episodes=1,
                      max_episode_steps=None, return_rollout=False,
                      return_bd_traj=False, **kwargs):
        base_seed = kwargs.get('base_seed', int(getattr(self.config, 'seed', 42)))
        step_offset = kwargs.get('step_offset', self._compute_seed_offset())
        env_name = kwargs.get('env_name', getattr(self.config, 'env_name', None))
        bc_type = 'gait'
        ant_bc_mode = kwargs.get('ant_bc_mode', getattr(self.config, 'ant_bc_mode', '2d'))

        return evaluate_actor(
            actor=actor, env=env,
            max_action=self.max_action,
            memory=memory, n_episodes=n_episodes,
            base_seed=base_seed, step_offset=step_offset,
            max_episode_steps=max_episode_steps,
            return_rollout=return_rollout, return_bd_traj=return_bd_traj,
            env_name=env_name, bc_type=bc_type, ant_bc_mode=ant_bc_mode,
        )

    def evaluate_actor_with_td3_update(self, actor, env, memory, config, max_action,
                                       seed_offset_func, max_episode_steps=1000,
                                       update_steps=10, update_frequency=2,
                                       sample_memory=None, write_memory=None,
                                       critic_td3=None, return_updated_critics=False,
                                       warmup_episodes=None,
                                       return_bd=False, env_name=None, **kwargs):
        return evaluate_actor_with_td3_update(
            actor=actor, env=env, memory=memory,
            config=config, max_action=max_action,
            seed_offset_func=seed_offset_func,
            max_episode_steps=max_episode_steps,
            update_steps=update_steps, update_frequency=update_frequency,
            sample_memory=sample_memory, write_memory=write_memory,
            critic_td3=critic_td3, return_updated_critics=return_updated_critics,
            warmup_episodes=warmup_episodes,
            exploration_noise_std=kwargs.get('exploration_noise_std', None),
            return_bd=return_bd,
            env_name=env_name or getattr(config, 'env_name', None),
            collect_eval_rollout_to_memory=bool(kwargs.get('collect_eval_rollout_to_memory', False)),
            eval_episodes=int(kwargs.get('eval_episodes', getattr(config, 'mask_eval_episodes', 1))),
            enable_actor_update=bool(kwargs.get('enable_actor_update',
                                                getattr(config, 'mask_td3_enable_actor_update', True))),
            conservative_actor_update=bool(kwargs.get('conservative_actor_update',
                                                      getattr(config, 'mask_td3_conservative_actor_update', False))),
            actor_sample_memory=kwargs.get('actor_sample_memory', None),
            actor_min_replay_steps=kwargs.get('actor_min_replay_steps', None),
            batch_size_override=kwargs.get('batch_size_override', None),
            grad_clip_norm=float(kwargs.get('grad_clip_norm', 0.5)),
            warmup_max_steps=kwargs.get('warmup_max_steps', None),
        )

    def _ensure_shared_critic(self):
        """Lazy-initialize the shared critic (clone from critic_bank or fresh)."""
        if not hasattr(self, 'critic') or self.critic is None:
            cb = getattr(self, 'critic_bank', None)
            if cb is not None and hasattr(cb, 'clone_shared_critic'):
                try:
                    self.critic = cb.clone_shared_critic(
                        lr=float(getattr(self.config, 'critic_lr', 1e-3)),
                        device=str(device),
                    )
                    return self.critic
                except Exception:
                    pass
            self.critic = CriticTD3(self.input_dim, self.output_dim, self.max_action, self.config).to(device)
            if not hasattr(self.critic, 'optimizer') or self.critic.optimizer is None:
                lr_c = getattr(self.config, 'critic_lr', 1e-3)
                self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        return self.critic

    def _ensure_critic_bank(self) -> SharedCriticBank:
        if not hasattr(self, 'critic_bank') or self.critic_bank is None:
            K = int(getattr(self.config, 'critic_ensemble_K', 5))
            lambda_explore = float(getattr(self.config, 'critic_lambda_explore', 0.05))
            self.critic_bank = SharedCriticBank(
                state_dim=self.input_dim,
                action_dim=self.output_dim,
                max_action=self.max_action,
                config=self.config,
                device=str(device),
            )
            self.critic_bank.K = K
            self.critic_bank.lambda_explore = lambda_explore
        return self.critic_bank


    def _initialize_mask_es_optimizer(self):
        """Initialize mask search from the configured network dimensions."""
        external_archive = getattr(self, 'external_archive', None)
        self.mask_es_optimizer, self.layer_sizes = initialize_mask_ga(
            self.input_dim,
            self.current_hidden_layers,
            self.output_dim,
            config=self.config,
            external_archive=external_archive,
        )
        self.old_mask_fitness = []

    def _es_eval_candidate(self, actor, memory, return_rollout=True, env_name=None, bc_type="gait"):
        """ES-phase eval: reward + BD trajectory in one rollout (saves a re-eval)."""
        rewards_arr, rollout_datas, _legacy_bd, x_traj_list, xpos_traj_list = self.evaluate_actor(
            actor, self.env, memory=memory, n_episodes=1,
            base_seed=self.config.seed, step_offset=self.total_steps,
            return_rollout=bool(return_rollout), return_bd_traj=True,
            env_name=env_name or self.env_name, bc_type=bc_type,
        )

        rewards_arr = np.asarray(rewards_arr, dtype=np.float32).reshape(-1)
        f = float(rewards_arr.mean()) if rewards_arr.size > 0 else 0.0
        steps = len(rollout_datas[0]) if rollout_datas and len(rollout_datas[0]) > 0 else int(rewards_arr.size)
        if not x_traj_list or not xpos_traj_list:
            raise RuntimeError(
                f"[_es_eval_candidate] empty BD trajectories "
                f"(x_traj_list={len(x_traj_list)}, xpos_traj_list={len(xpos_traj_list)}). "
                f"env.unwrapped.data is inaccessible."
            )

        obs_descriptor = aggregate_bd_from_episodes(
            x_traj_list, xpos_traj_list, env_name or self.env_name,
            dt=resolve_control_dt(self.env, env_name or self.env_name),
        )

        archive_payload = {
            'rewards_arr': rewards_arr,
            'rollout_datas': rollout_datas,
            'x_traj_list': x_traj_list,
            'xpos_traj_list': xpos_traj_list,
            'obs_descriptor': np.asarray(obs_descriptor, dtype=np.float32).reshape(-1),
            'eval_seeds': [int(self.config.seed) + int(self.total_steps)],
            '_es_evaluated': True,
        }
        main_opt = getattr(self, 'main_optimizer', None)
        if main_opt is not None:
            main_opt.total_evaluations += 1
            # EnvStepAccountant already counts ES interactions at env.step.
        return f, steps, archive_payload

    def _es_optimization_phase(self, env=None) -> List[Dict[str, Any]]:
        if env is not None:
            self.env = env
        if self.actor_t is None:
            self.actor_t = Actor(self.input_dim, self.output_dim, self.max_action,
                                 hidden_layers=self.current_hidden_layers, args=self.config,
                                 masks=None).to(device)
            self.actor_t.load_state_dict(self.actor.state_dict())
        start_steps = int(getattr(self.config, 'es_start_steps', 10000))
        n_grad = int(getattr(self.config, 'es_n_grad', 5))
        n_noisy = int(getattr(self.config, 'es_n_noisy', 0))
        all_candidates: List[Dict[str, Any]] = []

        for iteration in range(self.config.es_max_generations):
            iter_wall_start = time.time()
            fitness = np.zeros(self.config.es_pop_size)
            self.n_start = np.zeros(self.config.es_pop_size)
            self.n_steps = np.zeros(self.config.es_pop_size)
            archive_payloads: List[Dict[str, Any]] = [None] * self.config.es_pop_size
            es_params, n_r, idx_r = self.sampler.ask(self.config.es_pop_size, self.old_es)
            # IMSampler places reused slots first; gradient-modified slots must be remeasured.
            n_r = int(min(max(0, n_r), int(self.config.es_pop_size)))
            idx_r = [int(k) for k in list(idx_r)[:n_r]]
            n_grad = min(n_grad, int(self.config.es_pop_size))

            critic_td3 = None
            critic_t = None
            if self.total_steps > start_steps:
                critic_td3 = self._ensure_shared_critic()
                if not hasattr(self, '_critic_t') or self._critic_t is None:
                    self._critic_t = safe_deepcopy(critic_td3, device_target=device)
                critic_t = self._critic_t
                steps_for_update = int(self.actor_steps) + int(self.reused_steps)
                n_grad_eff = max(1, int(n_grad))
                num_c_updates = int(steps_for_update // n_grad_eff)
                num_a_updates = int(steps_for_update)
                for i in range(n_grad):
                    set_actor_params(self.actor, es_params[i], self.param_mask, self.masked_param_indices)
                    set_actor_params(self.actor_t, es_params[i], self.param_mask, self.masked_param_indices)
                    # Preserve Adam state unless the actor parameter objects have changed.
                    _opt = getattr(self.actor, 'optimizer', None)
                    _opt_ok = False
                    if _opt is not None:
                        try:
                            _tracked = {id(p) for g in _opt.param_groups for p in g['params']}
                            _opt_ok = (_tracked == {id(p) for p in self.actor.parameters()})
                        except Exception:
                            _opt_ok = False
                    if not _opt_ok:
                        self.actor.optimizer = torch.optim.Adam(
                            self.actor.parameters(), lr=self.config.actor_lr)

                    mem_for_actor = self.get_memory_for_actor(self.actor)
                    for _ in range(num_c_updates):
                        critic_td3.update(mem_for_actor, self.config.batch_size, actor_t=self.actor, critic_t=critic_t)
                    for _ in range(num_a_updates):
                        self.actor.update(mem_for_actor, self.config.batch_size, critic_td3, self.actor_t)

                    es_params[i] = get_actor_params(self.actor, self.param_mask, self.masked_param_indices)

            self.actor_steps = 0
            self.reused_steps = 0

            for i in range(n_noisy):
                set_actor_params(self.actor, es_params[i], self.param_mask, self.masked_param_indices)
                f, steps, _payload = self._es_eval_candidate(
                    self.actor, self.get_memory_for_actor(self.actor),
                    return_rollout=True,
                )
                self.actor_steps += steps
                archive_payloads[i] = _payload
                print(f"Noisy actor {i} fitness:{f:.4f}")

            for i in range(self.config.es_pop_size):
                # Remeasure gradient-trained reused slots and every fresh sample.
                if i < n_grad or i >= n_r:
                    set_actor_params(self.actor, es_params[i], self.param_mask, self.masked_param_indices)

                    mem_for_actor = self.get_memory_for_actor(self.actor)
                    pos = mem_for_actor.get_pos() if hasattr(mem_for_actor, 'get_pos') else 0
                    f, steps, _payload = self._es_eval_candidate(
                        self.actor, mem_for_actor,
                        return_rollout=True,
                    )
                    self.actor_steps += steps
                    fitness[i] = f
                    self.n_steps[i] = steps
                    self.n_start[i] = pos
                    archive_payloads[i] = _payload
                else:
                    # idx_r maps each reused slot directly to its previous population index.
                    idx = idx_r[i]
                    fitness[i] = self.old_fitness[idx] if len(self.old_fitness) > idx else 0.0
                    self.n_steps[i] = self.old_n_steps[idx] if len(self.old_n_steps) > idx else 0.0
                    self.n_start[i] = self.old_n_start[idx] if len(self.old_n_start) > idx else 0.0
                    self.reused_steps += self.n_steps[i]

                    memory_size = getattr(self.config, 'memory_size', 1000000)
                    mem_for_actor = self.get_memory_for_actor(self.actor)
                    if hasattr(mem_for_actor, 'repeat'):
                        mem_for_actor.repeat(int(self.n_start[i]), int((self.n_start[i] + self.n_steps[i]) % memory_size))
                    reused_payload = None
                    if self.old_archive_payloads is not None and 0 <= idx < len(self.old_archive_payloads):
                        reused_payload = self.old_archive_payloads[idx]
                    archive_payloads[i] = reused_payload

            self.es_optimizer.tell(es_params, np.asarray(fitness, dtype=np.float64))
            self.total_steps += self.actor_steps
            self.old_fitness = deepcopy(fitness.tolist())
            self.old_n_steps = deepcopy(self.n_steps.tolist())
            self.old_n_start = deepcopy(self.n_start.tolist())
            self.old_es = es_params
            self.old_archive_payloads = archive_payloads

            set_actor_params(self.actor, self.es_optimizer.mu, self.param_mask, self.masked_param_indices)
            n_eval = int(getattr(self.config, 'es_n_eval', 1))
            f_mu, _ = self.evaluate_actor(
                self.actor, self.env, memory=None, n_episodes=n_eval,
                base_seed=self.config.seed, step_offset=self.total_steps,
            )
            iter_wall_time = time.time() - iter_wall_start
            print(f"[ES] iter {iteration} | best={float(np.max(fitness)):.4f} "
                    f"mean={float(np.mean(fitness)):.4f} mu={f_mu:.4f} "
                    f"n_reused={n_r} total_steps={self.total_steps} "
                    f"wall={iter_wall_time:.1f}s")

            c_for_payload = self._ensure_shared_critic()
            pop_n = int(self.config.es_pop_size)

            for i in range(pop_n):
                set_actor_params(self.actor, es_params[i], self.param_mask, self.masked_param_indices)

                payload_data = archive_payloads[i] or {}
                reused = (i >= n_grad and i < n_r)
                rewards_arr_cached = payload_data.get('rewards_arr')
                if rewards_arr_cached is not None and np.size(rewards_arr_cached) > 0:
                    eval_avg_reward = float(np.mean(rewards_arr_cached))
                else:
                    eval_avg_reward = float(fitness[i])
                x_traj_list = payload_data.get('x_traj_list') or []
                xpos_traj_list = payload_data.get('xpos_traj_list') or []
                _obs_desc = None
                _force_reeval = True
                if x_traj_list and xpos_traj_list:
                    _obs_desc = aggregate_bd_from_episodes(
                        x_traj_list, xpos_traj_list, self.env_name,
                        dt=resolve_control_dt(self.env, self.env_name),
                    ).astype(np.float32)
                    _force_reeval = False

                candidate_net = safe_deepcopy(self.actor, device_target='cpu')
                payload = {
                    'source': 'es',
                    'network': candidate_net,
                    'critic': safe_deepcopy(c_for_payload, device_target='cpu') if c_for_payload is not None else None,
                    'critic_target': safe_deepcopy(c_for_payload, device_target='cpu') if c_for_payload is not None else None,
                    'masks': None,
                    'sparsity': 0.0,
                    'avg_reward': eval_avg_reward,
                    'obs_descriptor': _obs_desc,
                    'rewards_arr': rewards_arr_cached if (rewards_arr_cached is not None and np.size(rewards_arr_cached) > 0) else None,
                    'x_traj_list': x_traj_list,
                    'xpos_traj_list': xpos_traj_list,
                    # Bind cached reward and BD to the exact evaluated network.
                    'policy_fingerprint': policy_fingerprint(candidate_net),
                    'eval_seeds': list(payload_data.get('eval_seeds') or []),
                    '_es_evaluated': True,
                    '_cached_episodes': int(
                        np.size(rewards_arr_cached) if rewards_arr_cached is not None else 0
                    ),
                    '_es_slot': int(i),
                    '_es_reused': bool(reused),
                    'env_steps': int(sum(max(0, len(traj) - 1) for traj in x_traj_list)),
                    'n_evals': 1,
                }
                if _force_reeval:
                    payload['_force_reeval'] = True
                all_candidates.append(payload)

        self.use_dense_memory()
        apply_es_mean_to_actor(
            self.actor, self.es_optimizer,
            self.param_mask, self.masked_param_indices, self.full_param_template,
        )
        return all_candidates

    def _mask_optimization_phase(self) -> List[Dict[str, Any]]:
        if bool(getattr(self.config, 'disable_mask_search', False)):
            print('[mask] disable_mask_search=True — Mask GA 跳过')
            return []
        self.use_sparse_memory()
        self._initialize_mask_es_optimizer()
        mask_phase_candidates: List[Dict[str, Any]] = []

        critic_bank = self._ensure_critic_bank()
        main_opt = getattr(self, 'main_optimizer', None)

        # Use a fixed ES critic snapshot throughout the mask phase.
        from networks import safe_deepcopy as _sdc
        if self.critic is None:
            try:
                self.critic = self._ensure_shared_critic()
            except Exception:
                self.critic = None
        es_critic_snapshot = _sdc(self.critic, device_target='cpu') \
            if self.critic is not None else None

        def _maybe_update_ensemble():
            if (not bool(getattr(self.config, 'mask_update_ensemble_before_ask', True))
                    or critic_bank is None
                    or self.memory_dense is None):
                return
            try:
                if hasattr(self.memory_dense, 'size') and self.memory_dense.size() < int(
                        getattr(self.config, 'critic_update_min_buffer', 256)):
                    return
                critic_bank.update_per_generation(self.memory_dense, self.actor)
            except Exception as e:
                logger.debug(f"shared critic update failed: {e}")

        for generation in range(self.config.mask_generations):
            _maybe_update_ensemble()

            buffer_id_prefix = (
                f"mask_gc{int(getattr(self, 'global_count', 0))}_gen{int(generation)}"
            )
            pending_buffer_ids = [
                f"{buffer_id_prefix}_ind{i}"
                for i in range(int(self.config.mask_pop_size))
            ]
            if hasattr(self.memory_sparse, 'protect_ids'):
                self.memory_sparse.protect_ids(pending_buffer_ids)

            temp_candidates, fitness, candidate_masks, obs_descriptors, emb_descriptors, critic_profiles, critic_signals = run_mask_generation(
                generation=generation,
                mask_es_optimizer=self.mask_es_optimizer,
                layer_sizes=self.layer_sizes,
                actor=self.actor,
                env=self.env,
                env_name=self.env_name,
                config=self.config,
                memory_sparse=self.memory_sparse,
                memory_dense=self.memory_dense,
                clone_sparse_memory_func=self.clone_sparse_memory,
                evaluate_with_td3_func=self.evaluate_actor_with_td3_update,
                critic_bank=critic_bank,
                seed_offset_func=self._compute_seed_offset,
                archive=getattr(self, 'external_archive', None),
                bd_n_bins=int(getattr(self.config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM)),
                buffer_id_prefix=buffer_id_prefix,
                snapshot_callback=lambda bid, snap: self._archive_replay_cache.__setitem__(bid, snap),
                es_critic_snapshot=es_critic_snapshot,
            )
            mask_env_steps = sum(int(c.get('mask_eval_steps', 0)) for c in temp_candidates)
            self.total_steps += int(mask_env_steps)

            expected_n = int(self.config.mask_pop_size)
            tell_shapes = {
                'candidates': len(temp_candidates),
                'masks': np.asarray(candidate_masks).shape,
                'fitness': np.asarray(fitness).shape,
                'obs': np.asarray(obs_descriptors).shape,
                'emb': np.asarray(emb_descriptors).shape,
            }
            if (
                len(temp_candidates) != expected_n
                or np.asarray(candidate_masks).shape[0] != expected_n
                or np.asarray(fitness).shape != (expected_n,)
                or np.asarray(obs_descriptors).shape[0] != expected_n
                or np.asarray(emb_descriptors).shape[0] != expected_n
            ):
                raise RuntimeError(
                    f"MaskGA ask/tell contract violated at generation {generation}: "
                    f"expected_N={expected_n}, shapes={tell_shapes}"
                )
            self.mask_es_optimizer.tell(
                candidate_masks,
                fitness,
                obs_descriptors=obs_descriptors,
                emb_descriptors=emb_descriptors,
                critic_profiles=critic_profiles,
                critic_signals=critic_signals,
                candidates=temp_candidates,
            )
            ga_fitness = np.asarray(
                getattr(self.mask_es_optimizer, 'fitness', fitness),
                dtype=np.float64,
            )
            for cand in temp_candidates:
                ci = int(cand.get('_i', -1))
                if 0 <= ci < ga_fitness.shape[0]:
                    cand['td3_reward'] = float(cand.get('avg_reward', 0.0))
                    cand['fitness'] = float(ga_fitness[ci])
            self.old_mask_fitness = deepcopy(ga_fitness.tolist())

            is_final_generation = generation == int(self.config.mask_generations) - 1
            submit_candidates = self._select_submit_candidates_per_generation(
                temp_candidates=temp_candidates,
                is_final_generation=is_final_generation,
            )
            if main_opt is not None and submit_candidates:
                writer_summary = self._write_candidates_to_main_archive(
                    submit_candidates, main_opt,
                )
            else:
                writer_summary = {
                    'append_count': 0,
                    'replace_count': 0,
                    'rejected_count': len(temp_candidates) - len(submit_candidates),
                    'reject_error_count': 0,
                    'admitted_candidate_ids': [],
                }

            logger.info(
                "Mask generation %d: fitness %.4f, candidates %d, admitted %d/%d, steps %d",
                generation, float(ga_fitness.max()), len(temp_candidates),
                writer_summary.get('append_count', 0), writer_summary.get('replace_count', 0),
                self.total_steps,
            )

            admitted_ids = set(writer_summary.get('admitted_candidate_ids', []))
            if hasattr(self.memory_sparse, 'release_ids'):
                self.memory_sparse.release_ids(pending_buffer_ids)
            retained_by_id = {
                c.get('candidate_id'): c
                for c in mask_phase_candidates
                if c.get('candidate_id') in self._archive_mask_ids
            }
            for cand in temp_candidates:
                buffer_id = cand.get('buffer_id')
                if buffer_id not in admitted_ids:
                    continue
                per_cand_critic = cand.get('critic')
                if per_cand_critic is None and critic_bank is not None:
                    per_cand_critic = critic_bank.clone_shared_critic(
                        lr=float(getattr(self.config, 'critic_lr', 1e-3)),
                        device='cpu',
                    )
                per_cand_critic_target = cand.get('critic_target')
                if per_cand_critic_target is None and critic_bank is not None:
                    per_cand_critic_target = critic_bank.clone_shared_critic(
                        lr=float(getattr(self.config, 'critic_lr', 1e-3)),
                        device='cpu',
                    )
                retained_by_id[buffer_id] = {
                    'source': 'mask',
                    '_mask_evaluated': True,
                    '_force_reeval': False,
                    'n_evals': cand.get('n_evals', 1),
                    'network': cand['network'],
                    'critic': per_cand_critic,
                    'critic_target': per_cand_critic_target,
                    'masks': cand['masks'],
                    'candidate_id': buffer_id,
                    'memory_id': buffer_id,
                    'sparsity': cand.get('sparsity', 0.0),
                    'fitness': cand.get('fitness', 0.0),
                    'coverage_fitness': cand.get('coverage_fitness'),
                    'coverage_is_new_cell': cand.get('coverage_is_new_cell'),
                    'coverage_is_batch_duplicate': cand.get('coverage_is_batch_duplicate'),
                    'coverage_novelty': cand.get('coverage_novelty'),
                    'private_buffer_steps': cand.get('private_buffer_steps'),
                    'avg_reward': cand.get('avg_reward'),
                    'obs_descriptor': cand.get('obs_descriptor'),
                    'target_bd': cand.get('target_bd'),
                }
            mask_phase_candidates = list(retained_by_id.values())

            cleanup_mask_buffers([], self.memory_sparse, set())
            for cand in temp_candidates:
                if cand.get('buffer_id') in admitted_ids:
                    continue
                cand['network'] = None
                cand['critic'] = None
                cand['critic_target'] = None
                cand['masks'] = None

            self.global_count += 1

        return mask_phase_candidates

    def _select_submit_candidates_per_generation(self, temp_candidates,
                                                 is_final_generation):
        """Submit one best-return candidate per cell for full admission reevaluation."""
        cfg = self.config
        if not temp_candidates:
            return []
        only_final = bool(getattr(cfg, 'mask_admission_only_final_generation', False))
        per_gen_enabled = bool(getattr(cfg, 'mask_admission_per_generation', True))

        if only_final:
            if not is_final_generation:
                return []
        elif not per_gen_enabled:
            return []

        best_by_cell = {}
        seen_buffer_ids = set()
        for cand in temp_candidates:
            bid = cand.get('buffer_id')
            if bid in seen_buffer_ids:
                continue
            seen_buffer_ids.add(bid)
            bd = cand.get('obs_descriptor', None)
            cell = cand.get('coverage_cell', cand.get('bd_cell', None))
            if isinstance(cell, (list, tuple, np.ndarray)):
                cell = tuple(cell)
            bd_valid = (
                isinstance(bd, np.ndarray)
                and bd.size >= 2
                and np.all(np.isfinite(bd))
            )
            if not bd_valid:
                continue
            key = cell
            incumbent = best_by_cell.get(key)
            if incumbent is None or float(cand.get('avg_reward', -np.inf)) > float(
                    incumbent.get('avg_reward', -np.inf)):
                best_by_cell[key] = cand
        return list(best_by_cell.values())

    def _write_candidates_to_main_archive(self, candidates, main_opt):
        """Submit full evaluated mask population to the canonical MAP-Elites writer.

        archive tuple (10 fields):
            [0] network, [1] reward, [2] BD, [3] masks, [4] sparsity,
            [5] inference_metrics, [6] critic, [7] critic_target,
            [8] buffer_id, [9] n_evals
        """
        if not candidates:
            return {
                'append_count': 0, 'replace_count': 0,
                'rejected_count': 0, 'reject_error_count': 0,
                'admitted_candidate_ids': [],
            }
        if main_opt is None:
            raise RuntimeError("main optimizer is required for mask archive admission")

        for candidate in candidates:
            candidate.setdefault('source', 'mask')

        if getattr(main_opt, 'env', None) is None:
            raise RuntimeError("main optimizer environment is not initialized")

        results = main_opt._evaluate_and_archive_candidates(
            es_candidates=None, mask_candidates=candidates,
            env=main_opt.env, return_results=True,
        ) or []

        append_count = sum(1 for result in results if result.get('action') == 'append')
        replace_count = sum(1 for result in results if result.get('action') == 'replace')
        rejected_count = len(results) - append_count - replace_count
        reject_error_count = sum(1 for result in results if result.get('action') == 'reject_error')
        admitted_candidate_ids = [
            result.get('candidate_id')
            for result in results
            if result.get('action') in ('append', 'replace')
            and result.get('candidate_id') is not None
        ]
        return {
            'append_count': append_count,
            'replace_count': replace_count,
            'rejected_count': rejected_count,
            'reject_error_count': reject_error_count,
            'admitted_candidate_ids': admitted_candidate_ids,
        }
