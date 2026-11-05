"""Mask-phase GA + TD3 evaluation pipeline for sparse-network search."""
import numpy as np
import torch
import logging
from collections import Counter

from pruning_utils import build_structurally_pruned_actor
from adaptive_archive_admission import (
    archive_cell_statistics,
    archive_occupied_cells,
    canonicalize_gait_bd,
    gait_bd_to_cell,
)
from mask_config import binarize_masks, get_mask_threshold
from behavior_descriptor import GRID_BINS_PER_DIM, BD_DIM
from networks import safe_deepcopy
from memory import MixedMemory, snapshot_replay_buffer

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SPARSITY_CAP = 0.99

# Normalize reward and diversity to the same batch-relative [-0.5, 1.5] range.
_BATCH_NORM_LO_Q = 5.0
_BATCH_NORM_HI_Q = 95.0
_BATCH_NORM_EPS = 1e-9


def _batch_norm(values):
    """Scale by the 5th/95th percentiles; constant or nonfinite batches become zeros."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros(arr.shape, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    valid = arr[finite]
    if valid.size == 1:
        out[finite] = 0.5
        return out
    lo = float(np.percentile(valid, _BATCH_NORM_LO_Q))
    hi = float(np.percentile(valid, _BATCH_NORM_HI_Q))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < _BATCH_NORM_EPS:
        out[finite] = np.where(valid > 0, 0.5, -0.5)
        return out
    out[finite] = np.clip((valid - lo) / (hi - lo), -0.5, 1.5)
    return out


def cap_sparsity(sp, cap=MAX_SPARSITY_CAP):
    """Clamp `sp` to [0, cap]; non-numeric input → 0.0."""
    try:
        v = float(sp)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > cap:
        return cap
    return v


def flat_to_neuron_masks(flat_mask, hidden_sizes):
    """Split a flat (sum(hidden_sizes),) array into per-hidden-layer 1-D arrays."""
    masks = []
    offset = 0
    for h in hidden_sizes:
        masks.append(flat_mask[offset:offset + h].copy())
        offset += h
    return masks


def neuron_masks_to_weight_masks(neuron_masks, layer_sizes):
    """Expand per-layer neuron masks to 2-D weight masks aligned with Linear.weight.

    For `layer_sizes = [in, h1, h2, out]` and `neuron_masks = [mask_h1, mask_h2]`:
      - layer 0 (in→h1): shape (in, h1), each column = mask_h1[j]
      - layer 1 (h1→h2): shape (h1, h2), each column = mask_h2[j]
      - layer 2 (h2→out): shape (h2, out), all-ones (output never pruned)
    """
    weight_masks = []
    n_layers = len(layer_sizes) - 1
    for i in range(n_layers):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        if i < len(neuron_masks):
            nmask = np.asarray(neuron_masks[i], dtype=np.float32).flatten()
            if nmask.size != out_dim:
                nmask = np.ones(out_dim, dtype=np.float32)
            wmask = np.tile(nmask.reshape(1, -1), (in_dim, 1))
        else:
            wmask = np.ones((in_dim, out_dim), dtype=np.float32)
        weight_masks.append(wmask)
    return weight_masks


def structural_sparsity(neuron_masks, layer_sizes):
    """Sparsity = 1 - (kept_params / dense_params), capped at MAX_SPARSITY_CAP."""
    threshold = get_mask_threshold()
    dense_params = 0
    for i in range(len(layer_sizes) - 1):
        dense_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]

    kept = []
    for nm in neuron_masks:
        n_kept = max(1, int(np.sum(np.asarray(nm) >= threshold)))
        kept.append(n_kept)

    sparse_params = 0
    for i in range(len(layer_sizes) - 1):
        in_d = layer_sizes[0] if i == 0 else kept[i - 1]
        out_d = kept[i] if i < len(kept) else layer_sizes[i + 1]
        sparse_params += in_d * out_d + out_d

    raw = 1.0 - (sparse_params / max(dense_params, 1))
    return cap_sparsity(raw)

# Require enough private replay before scoring a critic signal.
_MIN_PRIVATE_BUFFER_ROWS = 32


def iter_candidates_with_private_buffer(candidates, memory_sparse):
    """Yield (index, candidate, replay) with a critic snapshot and enough private replay."""
    if not hasattr(memory_sparse, 'get_buffer'):
        return
    for i, cand in enumerate(candidates):
        if cand.get('skipped') == 'no_es_critic_snapshot':
            continue
        buf_id = cand.get('buffer_id')
        if buf_id is None:
            continue
        try:
            private_buf = memory_sparse.get_buffer(buf_id)
        except Exception:
            private_buf = None
        if (private_buf is None or not hasattr(private_buf, 'size')
                or private_buf.size() < _MIN_PRIVATE_BUFFER_ROWS):
            continue
        yield i, cand, private_buf


def run_mask_generation(
    generation, mask_es_optimizer, layer_sizes, actor, env, env_name,
    config=None, memory_sparse=None, memory_dense=None,
    clone_sparse_memory_func=None, evaluate_with_td3_func=None,
    critic_ensemble=None, critic_bank=None, seed_offset_func=None,
    archive=None, bd_n_bins=GRID_BINS_PER_DIM,
    buffer_id_prefix=None, snapshot_callback=None,
    es_critic_snapshot=None,
):
    """Evaluate one mask-generation: prune+TD3 each candidate, then write
    `coverage_*` annotations and return the (P,)-shaped diversity fitness.
    """
    pop_size = int(getattr(config, 'mask_pop_size', 30))
    max_episode_steps = int(getattr(config, 'max_episode_steps', 1000))
    hidden_sizes = layer_sizes[1:-1]
    archive = archive if archive is not None else []

    candidate_masks = np.asarray(mask_es_optimizer.ask(pop_size), dtype=np.float64)
    expected_mask_dim = int(sum(hidden_sizes))
    if candidate_masks.shape != (pop_size, expected_mask_dim):
        raise RuntimeError(
            f"mask-optimizer ask contract violated at generation {generation}: "
            f"expected {(pop_size, expected_mask_dim)}, got {candidate_masks.shape}"
        )

    fitness = np.zeros(pop_size, dtype=np.float64)
    all_candidates = []

    for i in range(pop_size):
        flat_mask = candidate_masks[i]

        neuron_masks = flat_to_neuron_masks(flat_mask, hidden_sizes)
        binary_neuron_masks = binarize_masks(neuron_masks)
        weight_masks = neuron_masks_to_weight_masks(binary_neuron_masks, layer_sizes)

        test_actor = safe_deepcopy(actor, device_target=device)
        pruned_actor = build_structurally_pruned_actor(test_actor, weight_masks)
        pruned_actor = pruned_actor.to(device)

        overall_sparsity = cap_sparsity(
            structural_sparsity(binary_neuron_masks, layer_sizes)
        )
        layer_sparsities = []
        threshold = get_mask_threshold()
        for nm in binary_neuron_masks:
            n_pruned = int(np.sum(np.asarray(nm) < threshold))
            layer_sparsities.append(n_pruned / max(len(nm), 1))

        prefix = str(buffer_id_prefix) if buffer_id_prefix else f"mask_gen{generation}"
        buffer_id = f"{prefix}_ind{i}"
        clone_sparse_memory_func(None, buffer_id)
        sparse_buffer = memory_sparse.get_buffer(buffer_id) if hasattr(memory_sparse, 'get_buffer') else memory_sparse

        cand_obs_desc = None

        # Each candidate gets an independent copy of the fixed ES critic snapshot.
        _per_cand_critic = None
        if es_critic_snapshot is not None:
            try:
                from networks import safe_deepcopy as _sdc
                _per_cand_critic = _sdc(es_critic_snapshot, device_target=device)
                if _per_cand_critic is not None and not hasattr(_per_cand_critic, 'optimizer'):
                    _per_cand_critic.optimizer = torch.optim.Adam(
                        _per_cand_critic.parameters(),
                        lr=float(getattr(config, 'critic_lr', 1e-3)),
                    )
            except Exception:
                _per_cand_critic = None
        if _per_cand_critic is None:
            fitness[i] = float('-inf')
            all_candidates.append({
                'source': 'mask',
                '_mask_evaluated': True,
                '_force_reeval': False,
                'n_evals': 0,
                'network': None,
                'masks': weight_masks,
                'neuron_masks': [nm.copy() for nm in binary_neuron_masks],
                'flat_mask': flat_mask.copy(),
                'sparsity': overall_sparsity,
                'fitness': float('-inf'),
                'buffer_id': buffer_id,
                'avg_reward': float('-inf'),
                'obs_descriptor': None,
                '_i': i,
                'critic': None,
                'critic_target': None,
                'skipped': 'no_es_critic_snapshot',
            })
            del pruned_actor, test_actor
            continue

        _sparse_policy_noise = float(getattr(config, 'sparse_policy_noise',
                                             getattr(config, 'policy_noise', 0.2)))
        _sparse_noise_clip = float(getattr(config, 'sparse_noise_clip',
                                            getattr(config, 'noise_clip', 0.5)))

        # No candidate BD is available before learning; never reuse the previous one.
        base_bd = None

        if _per_cand_critic is not None:
            _per_cand_critic.policy_noise = _sparse_policy_noise
            _per_cand_critic.noise_clip = _sparse_noise_clip

        # Pre-learning direction proposals sample shared replay.
        if (critic_bank is not None
                and bool(getattr(config, 'mask_use_critic_field_during_eval', True))):
            try:
                from critic_guided_updater import inject_critic_field_direction
                if memory_dense is not None and hasattr(memory_dense, 'size') and memory_dense.size() > 0:
                    projected_agent_idx = None
                    if (archive is not None and len(archive) > 0
                            and bool(getattr(config, 'mask_cfdi_use_projected_idx', True))):
                        try:
                            cand_bd = base_bd
                            if cand_bd is not None:
                                cand_cell = gait_bd_to_cell(cand_bd, bd_n_bins)
                                if cand_cell is not None:
                                    ti, tj = cand_cell
                                    best_idx, best_dist = None, None
                                    for ai, ag in enumerate(archive):
                                        try:
                                            bd_a = ag[2] if len(ag) > 2 else None
                                            if bd_a is None:
                                                continue
                                            cell_a = gait_bd_to_cell(bd_a, bd_n_bins)
                                            if cell_a is None:
                                                continue
                                            d = abs(cell_a[0] - ti) + abs(cell_a[1] - tj)
                                            if best_dist is None or d < best_dist:
                                                best_dist = d
                                                best_idx = ai
                                        except Exception:
                                            continue
                                    projected_agent_idx = best_idx
                        except Exception:
                            projected_agent_idx = None

                    inject_critic_field_direction(
                        actor=pruned_actor,
                        critic=_per_cand_critic,
                        critic_bank=critic_bank,
                        memory=memory_dense,
                        mask=weight_masks,
                        archive=archive,
                        agent_idx=projected_agent_idx,
                        n_neighbors=int(getattr(config, 'mask_cfdi_n_neighbors', 3)),
                        alpha=float(getattr(config, 'mask_cfdi_eval_alpha', 0.1)),
                        noise_std=float(getattr(config, 'mask_cfdi_eval_noise', 0.005)),
                        batch_size=int(getattr(config, 'mask_cfdi_eval_batch', 128)),
                        grad_clip_norm=1.0,
                    )
            except Exception as e:
                logger.debug(f"mask CFDI (eval) failed: {e}")

        # The TD3 ablation also disables structural-candidate learning.
        _no_td3 = bool(getattr(config, 'ablation_no_td3', False))
        requested_update_steps = 0 if _no_td3 else max(
            0, int(getattr(config, 'mask_td3_update_steps', 500)))
        update_frequency = max(1, int(getattr(config, 'mask_td3_update_frequency', 2)))
        dense_size = int(memory_dense.size()) if memory_dense is not None and hasattr(memory_dense, 'size') else 0
        sparse_size_before = int(sparse_buffer.size()) if sparse_buffer is not None and hasattr(sparse_buffer, 'size') else 0
        mask_batch_size = max(1, int(getattr(config, 'mask_td3_batch_size', 128)))
        update_batch_size = mask_batch_size * 2
        dense_ratio = float(getattr(config, 'mask_td3_critic_dense_ratio', 0.3))
        sparse_ratio = float(getattr(config, 'mask_td3_critic_sparse_ratio', 0.7))
        if memory_dense is not None and sparse_buffer is not None:
            td3_sample_memory = MixedMemory(
                buffers=[memory_dense, sparse_buffer],
                ratio=[dense_ratio, sparse_ratio],
            )
            sample_source = 'mixed_dense_sparse'
        elif memory_dense is not None:
            td3_sample_memory = memory_dense
            sample_source = 'dense'
        else:
            td3_sample_memory = sparse_buffer
            sample_source = 'sparse'
        warmup_episodes = max(0, int(getattr(config, 'mask_td3_warmup_episodes', 1)))
        warmup_steps = max(0, int(getattr(config, 'mask_td3_warmup_steps', 256)))
        if requested_update_steps > 0 and sparse_size_before < mask_batch_size:
            warmup_episodes = max(1, warmup_episodes)
            warmup_steps = max(mask_batch_size, warmup_steps)
        effective_update_steps = (
            requested_update_steps
            if dense_size + max(sparse_size_before, warmup_steps) >= update_batch_size
            else 0
        )
        enable_actor_update = bool(getattr(config, 'mask_td3_enable_actor_update', True))
        conservative_actor_update = bool(getattr(config, 'mask_td3_conservative_actor_update', True))
        if _no_td3:
            # Without critic learning, evaluate the policy directly.
            enable_actor_update = False

        td3_score, _steps, td3_obs_desc = evaluate_with_td3_func(
                actor=pruned_actor,
                env=env,
                memory=sparse_buffer,
                config=config,
                max_action=float(getattr(actor, 'max_action', 1.0)),
                seed_offset_func=seed_offset_func,
                max_episode_steps=max_episode_steps,
                update_steps=requested_update_steps,
                update_frequency=update_frequency,
                sample_memory=td3_sample_memory,
                write_memory=sparse_buffer,
                critic_td3=_per_cand_critic,
                return_updated_critics=False,
                warmup_episodes=warmup_episodes,
                warmup_max_steps=warmup_steps,
                exploration_noise_std=_sparse_policy_noise,
                return_bd=True,
                env_name=env_name,
                eval_episodes=max(1, int(getattr(config, 'mask_eval_episodes', 1))),
                batch_size_override=mask_batch_size,
                enable_actor_update=enable_actor_update,
                conservative_actor_update=conservative_actor_update,
                actor_sample_memory=sparse_buffer,
                actor_min_replay_steps=max(1, int(getattr(
                    config, 'mask_td3_actor_min_replay_steps', mask_batch_size
                ))),
                lambda_explore=float(getattr(
                    config, 'mask_actor_explore_lambda', 0.05
                )),
                collect_eval_rollout_to_memory=bool(
                    getattr(config, 'mask_eval_write_rollout', True)
                ),
            )

        base_score = float(td3_score)
        base_bd, _ = canonicalize_gait_bd(td3_obs_desc)

        if sparse_buffer is not None and buffer_id and snapshot_callback is not None:
            try:
                snap = snapshot_replay_buffer(sparse_buffer)
                if snap is not None:
                    snapshot_callback(str(buffer_id), snap)
            except Exception as e:
                logger.debug(f"inline snapshot for {buffer_id} failed: {e}")

        score = float(base_score)
        logger.debug(f"candidate {i}: TD3={td3_score:.2f}, eval={score:.2f}, "
                     f"fitness={score:.2f}, sparsity={overall_sparsity:.2%}")

        fitness[i] = float(score)

        pruned_hidden = [layer.out_features for layer in pruned_actor.layers]

        critic_state = None
        if _per_cand_critic is not None:
            critic_state = {
                k: v.detach().cpu().clone()
                for k, v in _per_cand_critic.state_dict().items()
            }

        candidate = {
            'source': 'mask',
            '_mask_evaluated': True,
            '_force_reeval': False,
            'n_evals': 1,
            'network': safe_deepcopy(pruned_actor, device_target='cpu'),
            'masks': weight_masks,
            'neuron_masks': [nm.copy() for nm in binary_neuron_masks],
            'flat_mask': flat_mask.copy(),
            'sparsity': overall_sparsity,
            'layer_sparsities': layer_sparsities,
            'fitness': float(score),
            'buffer_id': buffer_id,
            'pruned_hidden_sizes': pruned_hidden,
            'avg_reward': float(score),
            'mask_eval_steps': int(_steps),
            'mask_td3_update_steps_requested': int(requested_update_steps),
            'mask_td3_update_steps_effective': int(effective_update_steps),
            'mask_td3_dense_size': int(dense_size),
            'mask_td3_sparse_size_before': int(sparse_size_before),
            'mask_td3_batch_size': int(mask_batch_size),
            'mask_td3_warmup_steps': int(warmup_steps),
            'mask_td3_sample_source': sample_source,
            'mask_td3_actor_update_enabled': bool(enable_actor_update),
            'mask_td3_conservative_actor_update': bool(conservative_actor_update),
            'obs_descriptor': (
                np.asarray(base_bd, dtype=np.float32).flatten()
                if base_bd is not None else cand_obs_desc
            ),
            '_i': i,
            'critic': critic_state,
            'critic_target': critic_state,
        }
        if sparse_buffer is not None and hasattr(sparse_buffer, 'size'):
            try:
                candidate['private_buffer_steps'] = int(sparse_buffer.size())
            except Exception:
                candidate['private_buffer_steps'] = -1

        all_candidates.append(candidate)
        del pruned_actor, test_actor

    archive_stats = archive_cell_statistics(archive, bd_n_bins)
    archive_cells = set(archive_stats['cells'])
    batch_seen_cells = set()
    n_true_global_new = 0
    n_batch_duplicates = 0
    n_invalid_bd = 0
    obs_arr = np.full((pop_size, BD_DIM), np.nan, dtype=np.float32)
    emb_arr = np.zeros((pop_size, 0), dtype=np.float32)

    for expected_i, candidate in enumerate(all_candidates):
        candidate_i = int(candidate.get('_i', -1))
        if candidate_i != expected_i:
            raise RuntimeError(
                f"mask-optimizer candidate order violated at generation {generation}: "
                f"expected row {expected_i}, got {candidate_i}"
            )

        raw_bd = candidate.get('obs_descriptor')
        bd, bd_status = canonicalize_gait_bd(raw_bd)
        candidate['bd_status'] = bd_status
        candidate['bd_is_global_new'] = False
        candidate['bd_is_batch_duplicate'] = False
        candidate['bd_is_true_global_new'] = False

        if bd is None:
            n_invalid_bd += 1
        else:
            candidate['obs_descriptor'] = bd
            obs_arr[candidate_i] = bd
            cell = gait_bd_to_cell(bd, bd_n_bins)
            if cell is None:
                n_invalid_bd += 1
            else:
                global_new = cell not in archive_cells
                batch_duplicate = cell in batch_seen_cells
                true_global_new = global_new and not batch_duplicate
                candidate['bd_cell'] = cell
                candidate['bd_is_global_new'] = global_new
                candidate['bd_is_batch_duplicate'] = batch_duplicate
                candidate['bd_is_true_global_new'] = true_global_new
                if batch_duplicate:
                    n_batch_duplicates += 1
                if true_global_new:
                    n_true_global_new += 1
                batch_seen_cells.add(cell)

    diversity_fitness = compute_diversity_scores(all_candidates, archive, bd_n_bins, config)
    for i, cand in enumerate(all_candidates):
        cand['coverage_fitness'] = float(diversity_fitness[i])
    fitness = compose_mask_fitness(all_candidates, diversity_fitness, config)
    for i, cand in enumerate(all_candidates):
        cand['fitness'] = float(fitness[i])

    if len(all_candidates) != pop_size:
        raise RuntimeError(
            f"mask-optimizer evaluation contract violated at generation {generation}: "
            f"evaluated={len(all_candidates)}, expected={pop_size}"
        )
    full_masks = np.stack(
        [np.asarray(candidate['flat_mask'], dtype=np.float64) for candidate in all_candidates],
        axis=0,
    )
    expected_shape = (pop_size, expected_mask_dim)
    contract_shapes = {
        'fitness': fitness.shape, 'masks': full_masks.shape,
        'obs': obs_arr.shape, 'emb': emb_arr.shape,
    }
    if (
        fitness.shape != (pop_size,)
        or full_masks.shape != expected_shape
        or obs_arr.shape[0] != pop_size
        or emb_arr.shape[0] != pop_size
    ):
        raise RuntimeError(
            f"mask-optimizer full-population contract violated at generation {generation}: "
            f"expected_N={pop_size}, shapes={contract_shapes}"
        )
    if not np.array_equal(full_masks, candidate_masks):
        raise RuntimeError(
            f"mask-optimizer ask-order solutions changed before tell at generation {generation}"
        )

    critic_profiles = np.asarray(
        [float(c.get('critic_q_std_at_candidate', 0.0)) for c in all_candidates],
        dtype=np.float64,
    )

    # Shared-critic disagreement on common Gaussian probe states.
    critic_signals = np.zeros(pop_size, dtype=np.float64)
    if critic_bank is not None and hasattr(critic_bank, 'shared_critic'):
        try:
            n_probe = int(getattr(config, 'mask_critic_signal_probe', 32))
            probe_states = torch.randn(n_probe, critic_bank.state_dim, device=critic_bank.device_str)
            for i, cand in enumerate(all_candidates):
                if cand.get('skipped') == 'no_es_critic_snapshot':
                    continue
                try:
                    net = cand.get('network')
                    if net is None:
                        continue
                    net_dev = net.to(critic_bank.device_str).eval()
                    with torch.no_grad():
                        acts = net_dev(probe_states)
                        acts = torch.clamp(acts, -critic_bank.max_action, critic_bank.max_action)
                    q1, q2 = critic_bank.shared_critic(probe_states, acts)
                    std_val = float(q1.squeeze(-1).std().item() + q2.squeeze(-1).std().item())
                    critic_signals[i] = std_val
                except Exception:
                    continue
            cs_max = float(critic_signals.max()) if critic_signals.size > 0 else 0.0
            if cs_max > 0:
                critic_signals = critic_signals / cs_max
        except Exception as e:
            logger.debug(f"critic_signals compute failed: {e}")

    # Critic-bank disagreement on the candidate private replay states.
    if (critic_bank is not None
            and hasattr(critic_bank, 'compute_uncertainty_from_buffer')
            and bool(getattr(config, 'mask_use_bd_conditional_critic', True))):
        try:
            bd_cond_signals = np.zeros(pop_size, dtype=np.float64)
            computed = 0
            for i, cand, private_buf in iter_candidates_with_private_buffer(
                    all_candidates, memory_sparse):
                actor_net = cand.get('network')
                if actor_net is None:
                    continue
                n_samples = int(getattr(config, 'mask_bd_conditional_n_samples', 64))
                unc = critic_bank.compute_uncertainty_from_buffer(
                    buffer=private_buf, actor=actor_net, n_samples=n_samples,
                )
                bd_cond_signals[i] = float(unc)
                computed += 1
            if computed > 0:
                if critic_signals.max() > 0:
                    critic_signals = np.maximum(critic_signals, bd_cond_signals)
                else:
                    critic_signals = bd_cond_signals
                logger.debug(f"[mask gen {generation}] BD-conditional critic: computed {computed}/{pop_size}")
        except Exception as e:
            logger.debug(f"BD-conditional critic signal failed: {e}")

    # Descriptor-conditioned predictor disagreement.
    if (critic_bank is not None
            and getattr(critic_bank, 'novelty_bank', None) is not None
            and bool(getattr(config, 'use_novelty_critic', False))):
        try:
            novelty_signals = np.zeros(pop_size, dtype=np.float64)
            computed = 0
            for i, cand, private_buf in iter_candidates_with_private_buffer(
                    all_candidates, memory_sparse):
                cand_bd = cand.get('obs_descriptor')
                if cand_bd is None:
                    continue
                bd_arr = np.asarray(cand_bd, dtype=np.float64).flatten()
                if bd_arr.size < 2:
                    continue
                n_samples = int(getattr(config, 'mask_bd_conditional_n_samples', 64))
                unc = critic_bank.compute_novelty_signal_from_buffer(
                    buffer=private_buf, candidate_bd=bd_arr[:2], n_samples=n_samples,
                )
                novelty_signals[i] = float(unc)
                computed += 1
            if computed > 0:
                critic_signals = np.maximum(critic_signals, novelty_signals)
                logger.debug(f"[mask gen {generation}] BD-novelty critic: computed {computed}/{pop_size}")
        except Exception as e:
            logger.debug(f"BD-novelty critic signal failed: {e}")

    return all_candidates, fitness.copy(), full_masks, obs_arr, emb_arr, critic_profiles, critic_signals


def cleanup_mask_buffers(candidates, memory_sparse, archive_mask_ids):
    """Release replay namespaces not retained by the canonical archive writer."""
    keep_ids = set()
    if archive_mask_ids:
        keep_ids.update(archive_mask_ids)
    for cand in candidates:
        buffer_id = cand.get('memory_id') or cand.get('candidate_id') or cand.get('buffer_id')
        if buffer_id:
            keep_ids.add(str(buffer_id))
    if hasattr(memory_sparse, 'clear_except'):
        memory_sparse.clear_except(list(keep_ids))


def _polynomial_mutation(x, eta=20.0, prob=0.1, rng=None):
    """Polynomial mutation (Deb & Agrawal). Per-gene probability `prob`."""
    if rng is None:
        rng = np.random.default_rng()
    out = np.asarray(x, dtype=np.float64).copy()
    N, D = out.shape
    mask = rng.random((N, D)) < prob
    if not mask.any():
        return out
    u = rng.random((N, D))
    delta1 = out[mask]
    delta2 = 1.0 - out[mask]
    mut_pow = 1.0 / (eta + 1.0)
    val = 2.0 * u[mask] - 1.0 + 1e-12
    delta_q = np.where(
        val <= 0.0,
        np.power(2.0 * u[mask] + (1.0 - 2.0 * u[mask]) * np.power(delta1, eta + 1.0), mut_pow) - 1.0,
        1.0 - np.power(2.0 * (1.0 - u[mask]) + 2.0 * (u[mask] - 0.5) * np.power(delta2, eta + 1.0), mut_pow),
    )
    out[mask] = np.clip(out[mask] + delta_q, 0.0, 1.0)
    return out


def _cauchy_perturbation(x, scale=0.5, prob=0.15, rng=None):
    """Per-individual Cauchy long-tail perturbation (clipped to [0, 1])."""
    if rng is None:
        rng = np.random.default_rng()
    out = np.asarray(x, dtype=np.float64).copy()
    N = out.shape[0]
    triggers = rng.random(N) < prob
    if not triggers.any():
        return out
    u = rng.random((int(triggers.sum()), out.shape[1]))
    noise = scale * np.tan(np.pi * (u - 0.5))
    out[triggers] = np.clip(out[triggers] + noise, 0.0, 1.0)
    return out


def _tournament_select(fitness, k, rng, bd_distances=None):
    """k-tournament WITHOUT replacement; tiebreak by BD distance if provided."""
    n = len(fitness)
    if n == 0:
        return 0
    k = min(k, n)
    idxs = rng.choice(n, size=k, replace=False)
    sub_fitness = fitness[idxs]
    best_local = int(np.argmax(sub_fitness))
    best_idx = int(idxs[best_local])
    if bd_distances is None:
        return best_idx
    best_f = float(sub_fitness[best_local])
    sub_dist = bd_distances[idxs]
    for li in range(k):
        fi = float(sub_fitness[li])
        if fi > best_f:
            best_f = fi
            best_idx = int(idxs[li])
        elif fi == best_f and sub_dist[li] > sub_dist[best_local]:
            best_idx = int(idxs[li])
            best_local = li
    return int(best_idx)


class MaskGA:
    """Mask-GA: tournament + uniform crossover + poly mutation + Cauchy
    perturbation + critic-modulated exploration + top-K elite preservation.
    Init is fully random (no ES dependency).
    """

    def __init__(
        self, num_mask_params, layer_sizes, pop_size=50,
        poly_eta=20.0, poly_prob=0.15,
        cauchy_prob=0.15, cauchy_scale=0.5,
        tournament_k=3, elite_k=None, elite_k_ratio=0.04,
        critic_weight=0.1, critic_cauchy_weight=1.0, critic_mutation_weight=0.5,
        init_random_ratio=0.5, init_min_value=0.3,
        exploit_ratio=0.80, critic_emit_ratio=0.20,
        reset_stuck_threshold=3, reset_inject_count=10,
        archive_learning_rate=1.0,
        critic_novelty_weight=0.0, critic_fitness_weight=0.0,
        adaptive_cauchy=True, adaptive_cauchy_max=0.50,
        bd_aware_emit=None, external_archive=None, config=None,
        target_sparsity_max=0.99, n_bins=GRID_BINS_PER_DIM, seed=None,
    ):
        self.layer_sizes = list(layer_sizes)
        self.hidden_sizes = layer_sizes[1:-1]
        self.num_mask_params = int(num_mask_params)
        self.pop_size = int(pop_size)
        self.config = config
        self.rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()
        self.poly_eta = float(poly_eta)
        self.poly_prob = float(poly_prob)
        self.cauchy_prob = float(cauchy_prob)
        self._cauchy_prob_base = float(cauchy_prob)
        self.cauchy_scale = float(cauchy_scale)
        self.adaptive_cauchy = bool(adaptive_cauchy)
        self.adaptive_cauchy_max = float(adaptive_cauchy_max)
        self.critic_cauchy_weight = float(critic_cauchy_weight)
        self.critic_mutation_weight = float(critic_mutation_weight)
        self.init_random_ratio = float(init_random_ratio)
        self.init_min_value = float(init_min_value)
        self.exploit_ratio = float(exploit_ratio)
        self.critic_emit_ratio = float(critic_emit_ratio)
        self.reset_stuck_threshold = int(reset_stuck_threshold)
        self.reset_inject_count = int(reset_inject_count)
        self.archive_learning_rate = float(archive_learning_rate)
        self.critic_novelty_weight = float(critic_novelty_weight)
        self.critic_fitness_weight = float(critic_fitness_weight)
        self._no_improve_gens = 0
        self._last_critic_signal = np.zeros(self.pop_size, dtype=np.float64)
        self._last_bd_distances = np.zeros(self.pop_size, dtype=np.float64)
        self.tournament_k = int(tournament_k)
        self.elite_k_ratio = float(elite_k_ratio)
        self.elite_k = max(1, int(round(self.pop_size * self.elite_k_ratio))) if elite_k is None else int(elite_k)
        self.critic_weight = float(critic_weight)
        self.elites = []
        self.external_archive = external_archive
        self.n_bins = int(n_bins)
        self.target_sparsity_max = float(target_sparsity_max)
        self._mask_max = float(target_sparsity_max)
        if bd_aware_emit is None:
            bd_aware_emit = bool(
                getattr(config, 'mask_bd_aware_emit', True) if config is not None else True
            )
        self.bd_aware_emit = bool(bd_aware_emit)
        self.population = None
        self.fitness = None
        self.scores = None
        self._initialized = False
        self.generation = 0
        self.best_fitness = float("-inf")


    def _seed_critic_emit_toward_sparse(self, n_critic, D):
        """Return (n_critic, D) seeds biased toward sparse BD regions, or None for uniform fallback."""
        if not self.bd_aware_emit or n_critic <= 0:
            return None
        if not self.external_archive or len(self.external_archive) < 2:
            return None

        aligned_masks = []
        aligned_bds = []
        aligned_cells = []
        for it in self.external_archive:
            if it is None or len(it) <= 3:
                continue
            bd_raw = it[2] if len(it) > 2 else None
            bd_arr, _ = canonicalize_gait_bd(bd_raw)
            if bd_arr is None:
                continue
            cell = gait_bd_to_cell(bd_arr, self.n_bins)
            if cell is None:
                continue
            mask_obj = it[3]
            if mask_obj is None:
                continue
            try:
                arr = np.asarray(mask_obj).flatten().astype(np.float64)
            except Exception:
                continue
            if arr.size != self.num_mask_params:
                continue
            arr = np.clip(arr, 0.0, self._mask_max)
            aligned_masks.append(arr)
            aligned_bds.append(bd_arr[:2])
            aligned_cells.append(cell)
        if not aligned_cells:
            return None

        archive_masks = np.stack(aligned_masks, axis=0)
        ac_bds_arr = np.asarray(aligned_bds, dtype=np.float64)

        cell_density = Counter(aligned_cells)
        all_cells = list(cell_density.keys())
        weights = np.array(
            [1.0 / (cell_density[c] + 1.0) for c in all_cells], dtype=np.float64,
        )
        weights /= weights.sum()
        n_pick = min(n_critic, len(all_cells))
        picked = self.rng.choice(len(all_cells), size=n_pick, replace=True, p=weights)

        cell_size = 1.0 / max(int(self.n_bins), 1)
        out = np.zeros((n_critic, D), dtype=np.float64)
        for i, ci in enumerate(picked):
            tc = all_cells[ci]
            tc_center = np.array(
                [(tc[0] + 0.5) * cell_size, (tc[1] + 0.5) * cell_size],
                dtype=np.float64,
            )
            nearest = int(np.argmin(
                np.linalg.norm(ac_bds_arr - tc_center[None, :], axis=1)
            ))
            density = float(cell_density[tc])
            noise_scale = 0.2 + 0.15 / (1.0 + density)
            out[i] = np.clip(
                archive_masks[nearest] + self.rng.normal(0.0, noise_scale, size=D),
                0.0, self._mask_max,
            )

        if n_pick < n_critic:
            out[n_pick:] = self.rng.uniform(
                0.0, self._mask_max, size=(n_critic - n_pick, D),
            )
        return out

    def _initialize_population_from_es(self):
        """Random init: (1 - init_random_ratio) fraction in [init_min_value, 1],
        init_random_ratio fraction in [0, 1]. No ES / actor-weight dependency.
        """
        N, D = self.pop_size, self.num_mask_params
        pop = np.zeros((N, D), dtype=np.float64)
        n_random = int(N * self.init_random_ratio)
        n_sparse = N - n_random
        if n_sparse > 0:
            pop[:n_sparse] = self.rng.uniform(
                self.init_min_value, self._mask_max, size=(n_sparse, D),
            )
        if n_random > 0:
            pop[n_sparse:n_sparse + n_random] = self.rng.uniform(
                0.0, self._mask_max, size=(n_random, D),
            )
        return pop

    def ask(self, pop_size=None):
        """Tournament + crossover + poly/Cauchy mutation; critic-modulated scales;
        adaptive Cauchy on plateau; top-K elite preservation.
        """
        requested = self.pop_size if pop_size is None else int(pop_size)
        if requested != self.pop_size:
            raise RuntimeError(
                f"MaskGA pop_size is fixed at {self.pop_size}; requested {requested}"
            )
        N, D = self.pop_size, self.num_mask_params
        if not self._initialized or self.population is None:
            pop = self._initialize_population_from_es()
            if pop.shape != (N, D):
                raise RuntimeError(
                    f"MaskGA initialization shape mismatch: "
                    f"expected {(N, D)}, got {pop.shape}"
                )
            self.population = pop
            self._initialized = True
            return self.population.copy()
        pop_curr = np.asarray(self.population, dtype=np.float64)
        if pop_curr.shape != (N, D):
            raise RuntimeError(
                f"MaskGA population shape mismatch before ask: "
                f"expected {(N, D)}, got {pop_curr.shape}"
            )
        fitness_curr = (
            np.asarray(self.fitness, dtype=np.float64)
            if self.fitness is not None
            else np.ones(N, dtype=np.float64)
        )

        # Split offspring between tournament crossover and random initialization.
        n_exploit = max(1, int(N * self.exploit_ratio))
        n_critic = N - n_exploit

        new_pop = np.zeros((N, D), dtype=np.float64)
        bd_distances = self._last_bd_distances
        if bd_distances.shape != (N,):
            bd_distances = np.zeros(N, dtype=np.float64)
        for i in range(n_exploit):
            p1 = _tournament_select(
                fitness_curr, self.tournament_k, self.rng, bd_distances=bd_distances,
            )
            p2 = _tournament_select(
                fitness_curr, self.tournament_k, self.rng, bd_distances=bd_distances,
            )
            xover_mask = self.rng.random(D) < 0.5
            new_pop[i] = np.where(xover_mask, pop_curr[p1], pop_curr[p2])

        if n_critic > 0:
            seeded_from_archive = self._seed_critic_emit_toward_sparse(n_critic=n_critic, D=D)
            if seeded_from_archive is not None:
                new_pop[n_exploit:n_exploit + n_critic] = seeded_from_archive
            else:
                new_pop[n_exploit:n_exploit + n_critic] = self.rng.uniform(
                    0.0, self._mask_max, size=(n_critic, D),
                )

        if self.critic_mutation_weight > 0.0 and self._last_critic_signal.max() > 1e-6:
            for i in range(N):
                factor = 1.0 + self.critic_mutation_weight * self._last_critic_signal[i]
                factor = min(2.5, max(0.5, factor))
                base = self.poly_prob if i < n_exploit else max(self.poly_prob, 0.10)
                p_i = base * factor
                mask_i = self.rng.random(D) < p_i
                if mask_i.any():
                    delta = self.rng.standard_normal(D) * 0.05
                    new_pop[i, mask_i] = np.clip(
                        new_pop[i, mask_i] + delta[mask_i], 0.0, self._mask_max,
                    )
        else:
            new_pop = _polynomial_mutation(
                new_pop, eta=self.poly_eta, prob=self.poly_prob, rng=self.rng,
            )

        if self.cauchy_prob > 0.0:
            if (self.critic_cauchy_weight > 0.0
                    and self._last_critic_signal.max() > 1e-6):
                scale_per = self.cauchy_scale * (
                    1.0 + self.critic_cauchy_weight * self._last_critic_signal
                )
                scale_per = np.clip(scale_per, 0.05, 1.5)
                triggers = self.rng.random(N) < self.cauchy_prob
                if triggers.any():
                    u = self.rng.random((int(triggers.sum()), D))
                    idx = np.where(triggers)[0]
                    noise = scale_per[idx, None] * np.tan(np.pi * (u - 0.5))
                    new_pop[idx] = np.clip(new_pop[idx] + noise, 0.0, self._mask_max)
            else:
                new_pop = _cauchy_perturbation(
                    new_pop, scale=self.cauchy_scale,
                    prob=self.cauchy_prob, rng=self.rng,
                )

        # After stagnation, reseed the tail from elites with a Cauchy perturbation.
        if (self._no_improve_gens >= self.reset_stuck_threshold
                and self.elites
                and n_critic > 0):
            elite_stack = np.stack(self.elites, axis=0)
            n_inject = min(self.reset_inject_count, len(elite_stack), n_critic)
            if n_inject > 0:
                idx = self.rng.choice(len(elite_stack), size=n_inject, replace=False)
                new_pop[N - n_inject:N] = elite_stack[idx]
                new_pop[N - n_inject:N] = _cauchy_perturbation(
                    new_pop[N - n_inject:N],
                    scale=self.cauchy_scale * 3.0,
                    prob=0.5,
                    rng=self.rng,
                )

        if self.elite_k > 0 and self.fitness is not None:
            elite_idx = np.argsort(-self.fitness)[: self.elite_k]
            n_keep = min(self.elite_k, len(elite_idx), N)
            new_pop[:n_keep] = pop_curr[elite_idx[:n_keep]]

        self.population = np.clip(new_pop, 0.0, self._mask_max)
        return self.population.copy()

    def tell(self, solutions, scores, obs_descriptors=None, emb_descriptors=None,
             critic_profiles=None, candidates=None, critic_signals=None):
        """Compute final fitness = diversity + critic_weight × critic_signal."""
        del emb_descriptors, critic_profiles, candidates
        solutions = np.asarray(solutions, dtype=np.float64)
        raw_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        N = self.pop_size

        # Cache nearest-archive distance for the next selection tie-break.
        bd_distances = np.zeros(N, dtype=np.float64)
        if obs_descriptors is not None and self.external_archive:
            try:
                obs_arr = np.asarray(obs_descriptors, dtype=np.float64)
                if obs_arr.ndim == 1:
                    obs_arr = obs_arr[None, :]
                archive_cells = archive_occupied_cells(
                    self.external_archive, self.n_bins,
                )
                if archive_cells:
                    ac_list = list(archive_cells)
                    ac_arr = np.asarray(ac_list, dtype=np.float64)
                    for i in range(N):
                        if i >= obs_arr.shape[0] or obs_arr.shape[1] < BD_DIM:
                            continue
                        bd_i, st = canonicalize_gait_bd(obs_arr[i, :BD_DIM])
                        if bd_i is None or st != 'valid':
                            continue
                        cell = gait_bd_to_cell(bd_i, self.n_bins)
                        if cell is None:
                            continue
                        ci = np.asarray(cell, dtype=np.float64)
                        d = np.abs(ac_arr - ci[None, :]).sum(axis=1)
                        bd_distances[i] = float(d.min())
            except Exception:
                pass
        self._last_bd_distances = bd_distances
        expected_solution_shape = (N, self.num_mask_params)
        if solutions.shape != expected_solution_shape:
            raise RuntimeError(
                f"MaskGA tell solutions contract violated: "
                f"expected {expected_solution_shape}, got {solutions.shape}"
            )
        if raw_scores.shape != (N,):
            raise RuntimeError(
                f"MaskGA tell scores contract violated: "
                f"expected {(N,)}, got {raw_scores.shape}"
            )
        if self.population is None or np.asarray(self.population).shape != expected_solution_shape:
            raise RuntimeError(
                f"MaskGA internal population contract violated before tell: "
                f"expected {expected_solution_shape}, got "
                f"{None if self.population is None else np.asarray(self.population).shape}"
            )
        solutions = np.clip(solutions, 0.0, self._mask_max)

        critic_signal_arr = np.zeros(N, dtype=np.float64)
        if critic_signals is not None:
            try:
                cs = np.asarray(critic_signals, dtype=np.float64).reshape(-1)
                if cs.shape[0] == N:
                    critic_signal_arr = np.nan_to_num(cs, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass

        # Alpha blends diversity and critic signals; critic_weight breaks selection ties.
        diversity_fitness = raw_scores.copy()
        if (self.archive_learning_rate < 1.0
                and self.critic_novelty_weight > 0.0
                and critic_signal_arr.max() > 1e-6):
            critic_novelty = (
                self.critic_novelty_weight
                * critic_signal_arr
                * (10.0 / max(0.1, critic_signal_arr.max()))
            )
            diversity_fitness = (
                self.archive_learning_rate * raw_scores
                + (1.0 - self.archive_learning_rate) * critic_novelty
            )
        if self.critic_fitness_weight > 0.0 and critic_signal_arr.max() > 1e-6:
            diversity_fitness = diversity_fitness * (
                1.0 + self.critic_fitness_weight * critic_signal_arr
            )
        fitness = diversity_fitness + self.critic_weight * critic_signal_arr
        self.fitness = fitness
        self.scores = raw_scores
        self._last_critic_signal = critic_signal_arr.copy()

        order = np.argsort(-fitness)
        n_keep = min(self.elite_k, N)
        self.elites = [solutions[i].copy() for i in order[:n_keep]]
        self.generation += 1
        new_best = float(fitness[order[0]])
        if self.generation > 1 and new_best <= self.best_fitness + 1e-3:
            self._no_improve_gens += 1
        else:
            self._no_improve_gens = 0
        self.best_fitness = max(self.best_fitness, new_best)
        if self.adaptive_cauchy and self._no_improve_gens >= 2:
            self.cauchy_prob = min(
                self.adaptive_cauchy_max,
                self._cauchy_prob_base * (1.5 ** min(self._no_improve_gens, 6)),
            )
        elif self.adaptive_cauchy:
            self.cauchy_prob = self._cauchy_prob_base


def compute_diversity_scores(candidates, archive, n_bins, config):
    """Pure-function diversity fitness used in `run_mask_generation`.

    Returns shape (P,) and writes `coverage_*` keys into each candidate for
    downstream coverage summaries. The critic signal is consumed in
    `MaskGA.tell`, not here.
    """
    P = len(candidates)
    coverage_new_cell_bonus = float(getattr(config, 'mask_coverage_new_cell_bonus', 10.0))
    coverage_duplicate_penalty = float(getattr(config, 'mask_coverage_duplicate_penalty', 3.0))
    # Saturate duplicate penalties so extra same-cell hits do not dominate reward.
    coverage_duplicate_saturation = max(
        1, int(getattr(config, 'mask_coverage_duplicate_saturation', 1)),
    )
    exploration_weight = float(getattr(config, 'mask_exploration_weight', 3.5))
    coverage_gap_weight = float(getattr(config, 'mask_coverage_gap_weight', 3.5))
    coverage_gap_radius = int(getattr(config, 'mask_coverage_gap_radius', 3))
    bd_farness_weight = float(getattr(config, 'mask_bd_farness_weight', 1.5))
    bd_farness_max_dist = int(getattr(config, 'mask_bd_farness_max_dist', 50))
    explore_radius = int(getattr(config, 'mask_explore_radius', 2))

    archive_cells = archive_occupied_cells(archive, n_bins)

    # Measure the empty fraction of the occupied-cell neighborhood, clipped to the grid.
    gap_total = {}
    gap_filled = {}
    for oc in archive_cells:
        oi, oj = oc
        for di in range(-coverage_gap_radius, coverage_gap_radius + 1):
            for dj in range(-coverage_gap_radius, coverage_gap_radius + 1):
                if abs(di) + abs(dj) > coverage_gap_radius:
                    continue
                key = (
                    int(np.clip(oi + di, 0, n_bins - 1)),
                    int(np.clip(oj + dj, 0, n_bins - 1)),
                )
                gap_total[key] = gap_total.get(key, 0) + 1
                if key in archive_cells:
                    gap_filled[key] = gap_filled.get(key, 0) + 1

    farness = np.full((n_bins, n_bins), 1.0, dtype=np.float32)
    if archive_cells:
        ac_list = list(archive_cells)
        for ci in range(n_bins):
            for cj in range(n_bins):
                d_min = bd_farness_max_dist
                for oc in ac_list:
                    d = abs(oc[0] - ci) + abs(oc[1] - cj)
                    if d < d_min:
                        d_min = d
                farness[ci, cj] = min(d_min, bd_farness_max_dist) / bd_farness_max_dist

    batch_cell_counts = {}
    fitness = np.zeros(P, dtype=np.float64)

    cells = [None] * P
    bds = [None] * P
    valid = [False] * P
    for i, cand in enumerate(candidates):
        raw_bd = cand.get('obs_descriptor')
        bd_arr, status = canonicalize_gait_bd(raw_bd)
        if bd_arr is None or status != 'valid':
            valid[i] = False
            continue
        cell = gait_bd_to_cell(bd_arr, n_bins)
        cells[i] = cell
        bds[i] = bd_arr
        valid[i] = True
        if cell is not None:
            batch_cell_counts[cell] = batch_cell_counts.get(cell, 0) + 1

    # Rank each cell by descending return and descriptor, independent of batch order.
    groups: dict = {}
    for i in range(P):
        if valid[i] and cells[i] is not None:
            groups.setdefault(cells[i], []).append(i)

    rank_in_cell = {}
    bonus_holder = {}
    for cell, members in groups.items():
        def _key(i):
            reward = candidates[i].get('avg_reward', float('-inf'))
            try:
                reward = float(reward)
            except Exception:
                reward = float('-inf')
            bd_key = tuple(np.round(np.asarray(bds[i], dtype=np.float64).flatten(), 6))
            return (-reward if np.isfinite(reward) else float('inf'), bd_key)

        ordered = sorted(members, key=_key)
        # Equal return and descriptor receive the same rank.
        rank = -1
        prev_key = object()
        for i in ordered:
            key = _key(i)
            if key != prev_key:
                rank += 1
            rank_in_cell[i] = rank
            prev_key = key
        if ordered:
            bonus_holder[ordered[0]] = cell

    for i, cand in enumerate(candidates):
        if not valid[i]:
            cand['coverage_cell'] = None
            cand['coverage_is_new_cell'] = False
            cand['coverage_is_batch_duplicate'] = False
            cand['coverage_is_true_global_new'] = False
            cand['coverage_explore_signal'] = 0.0
            cand['coverage_gap_signal'] = 0.0
            cand['coverage_farness_signal'] = 0.0
            cand['coverage_empty_cell_bonus'] = 0.0
            cand['coverage_novelty_signal'] = 0.0
            cand['coverage_cell_rank'] = None
            fitness[i] = -coverage_duplicate_penalty
            continue
        cell = cells[i]
        rank = rank_in_cell.get(i, 0)
        prior = float(rank)
        is_new = bool(cell is not None and cell not in archive_cells)
        is_best_in_cell = bool(rank == 0)
        if is_new and cell is not None:
            n_nb = sum(
                1 for oc in archive_cells
                if abs(oc[0] - cell[0]) <= explore_radius
                and abs(oc[1] - cell[1]) <= explore_radius
            )
            explore_sig = 1.0 / (1.0 + n_nb)
        else:
            explore_sig = 0.0
        if cell is not None:
            tot = gap_total.get(cell, 0)
            gap_sig = float(1.0 - gap_filled.get(cell, 0) / max(1, tot)) if tot else 1.0
        else:
            gap_sig = 0.0
        farness_sig = float(farness[cell[0], cell[1]]) if cell is not None else 0.0
        empty_bonus = coverage_new_cell_bonus if (is_new and is_best_in_cell) else 0.0
        fitness[i] = (
            empty_bonus
            + exploration_weight * explore_sig
            + coverage_gap_weight * gap_sig
            + bd_farness_weight * farness_sig
            - coverage_duplicate_penalty * min(float(prior), float(coverage_duplicate_saturation))
        )
        cand['coverage_cell'] = cell
        cand['coverage_is_new_cell'] = is_new
        cand['coverage_is_batch_duplicate'] = bool(rank > 0)
        cand['coverage_is_true_global_new'] = bool(is_new and is_best_in_cell)
        cand['coverage_cell_rank'] = int(rank)
        cand['coverage_cell_size'] = int(len(groups.get(cell, []))) if cell is not None else 0
        cand['coverage_explore_signal'] = float(explore_sig)
        cand['coverage_gap_signal'] = float(gap_sig)
        cand['coverage_farness_signal'] = float(farness_sig)
        cand['coverage_empty_cell_bonus'] = float(empty_bonus)
    return fitness


def compose_mask_fitness(candidates, diversity_fitness, config):
    """Blend normalized diversity, return and sparsity for selection.
    Constant signals become zero, leaving the other terms to determine selection."""
    div_w = float(getattr(config, 'mask_fitness_diversity_weight', 0.75))
    rew_w = float(getattr(config, 'mask_fitness_reward_weight', 0.25))
    sp_w = float(getattr(config, 'mask_fitness_sparsity_weight', 0.0))
    total = div_w + rew_w + sp_w
    if total <= 0.0:
        div_w, rew_w, sp_w, total = 1.0, 0.0, 0.0, 1.0
    div_w, rew_w, sp_w = div_w / total, rew_w / total, sp_w / total

    diversity = np.asarray(diversity_fitness, dtype=np.float64).reshape(-1)
    rewards = np.asarray(
        [float(c.get('avg_reward', float('-inf'))) for c in candidates],
        dtype=np.float64,
    )
    sparsity = np.asarray(
        [float(c.get('sparsity', 0.0)) for c in candidates], dtype=np.float64,
    )

    div_n = _batch_norm(diversity)
    rew_n = _batch_norm(rewards)
    sp_n = _batch_norm(np.nan_to_num(sparsity, nan=0.0, posinf=1.0, neginf=0.0))
    fitness = div_w * div_n + rew_w * rew_n + sp_w * sp_n

    # Candidates without finite TD3 returns remain unselectable.
    no_reward = ~np.isfinite(rewards)
    if no_reward.any():
        fitness[no_reward] = float('-inf')
    return fitness


def initialize_mask_ga(input_dim, hidden_layers, output_dim,
                       config=None, external_archive=None):
    """Return a randomly initialized MaskGA and layer sizes."""
    layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
    num_mask_params = int(sum(hidden_layers))

    n_bins = int(getattr(config, 'qd_n_bins_per_dim', GRID_BINS_PER_DIM)) if config is not None else GRID_BINS_PER_DIM
    seed = int(getattr(config, 'seed', 42)) if config is not None else 42
    pop_size = int(getattr(config, 'mask_pop_size', 50)) if config is not None else 50

    def _g(attr, default):
        return getattr(config, attr, default) if config is not None else default

    mask_optimizer = MaskGA(
        num_mask_params=num_mask_params,
        layer_sizes=layer_sizes,
        pop_size=pop_size,
        poly_eta=_g('mask_poly_eta', 20.0),
        poly_prob=_g('mask_poly_prob', 0.15),
        cauchy_prob=_g('mask_cauchy_prob', 0.15),
        cauchy_scale=_g('mask_cauchy_scale', 0.5),
        tournament_k=_g('mask_tournament_k', 3),
        elite_k_ratio=_g('mask_elite_k_ratio', 0.04),
        critic_weight=_g('mask_critic_weight', 1.0),
        critic_cauchy_weight=_g('mask_critic_cauchy_weight', 1.0),
        critic_mutation_weight=_g('mask_critic_mutation_weight', 0.5),
        exploit_ratio=_g('mask_exploit_ratio', 0.80),
        critic_emit_ratio=_g('mask_critic_emit_ratio', 0.20),
        reset_stuck_threshold=_g('mask_reset_stuck_threshold', 3),
        reset_inject_count=_g('mask_reset_inject_count', 10),
        archive_learning_rate=_g('mask_archive_learning_rate', 1.0),
        critic_novelty_weight=_g('mask_critic_novelty_weight', 0.0),
        critic_fitness_weight=_g('mask_critic_fitness_weight', 0.0),
        init_random_ratio=_g('mask_init_random_ratio', 0.5),
        init_min_value=_g('mask_init_min_value', 0.3),
        adaptive_cauchy=_g('mask_adaptive_cauchy', True),
        adaptive_cauchy_max=_g('mask_adaptive_cauchy_max', 0.50),
        external_archive=external_archive,
        config=config,
        target_sparsity_max=_g('target_sparsity_max', 0.99),
        n_bins=n_bins,
        seed=seed,
    )
    return mask_optimizer, layer_sizes
