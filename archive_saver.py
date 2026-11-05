"""Save training metadata, measured archive snapshots, and refinement summaries.

Archive exports contain inference weights and descriptors. They do not contain
replay buffers, optimizer moments or RNG state for restarting a training run.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from archive_io_utils import make_json_serializable
from behavior_descriptor import BD_SPEC_ID, get_bd_spec

ENV_STEPS_SEMANTICS = "counted_at_env_step_boundary_v2"
PROVENANCE_FIELDS = (
    "stable_id", "policy_fingerprint", "eval_seeds", "eval_episodes",
    "eval_status", "cached", "qd_side_candidate", "qd_side_reason", "source",
    "admission_reason", "coverage_seed",
)


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(make_json_serializable(value), handle, indent=2, ensure_ascii=False)


def init_results_dirs(config, env_name, bc_type, input_dim, output_dim,
                      max_action, ablation_label=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.environ.get("RESULTS_ROOT") or os.path.join(os.getcwd(), "results")
    results = Path(root) / f"{env_name}_{bc_type}_{timestamp}"
    archive = results / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_root": root,
        "results_dir": str(results),
        "archive_dir": str(archive),
        "archive_manifest_path": str(archive / "manifest.json"),
        "training_log_path": str(results / "training_log.json"),
    }
    fields = getattr(config, "__dataclass_fields__", vars(config))
    initial = {
        "env_name": env_name, "bc_type": bc_type,
        "input_dim": input_dim, "output_dim": output_dim, "max_action": max_action,
        "created_at": timestamp, "bd_spec": get_bd_spec(env_name),
        "ablation": ablation_label or "baseline",
    }
    _write_json(results / "init_info.json", initial)
    _write_json(paths["training_log_path"], {
        "experiment_config": {key: getattr(config, key, None) for key in fields},
        **initial, "iterations": [],
    })
    return paths


def _save_weights(directory, filename, model, enabled):
    if not enabled or model is None:
        return ""
    torch.save(model.state_dict(), directory / filename)
    return filename


def save_archive_snapshot(
    archive, global_iter, tag, archive_dir, archive_manifest_path,
    archive_save_models, archive_save_critics, archive_save_masks_npz,
    qd_metrics=None, n_interactions=0, elapsed_time=0.0, bd_spec=None,
    ablation_label=None,
):
    """Export each retained policy, its value networks, masks and measurements."""
    snapshot_name = f"iter_{global_iter:04d}_{tag}"
    snapshot = Path(archive_dir) / snapshot_name
    snapshot.mkdir(parents=True, exist_ok=True)
    bd_spec = bd_spec or {"bd_spec_id": BD_SPEC_ID}
    ablation = ablation_label or "baseline"
    records = []
    for idx, individual in enumerate(archive):
        if not isinstance(individual, (list, tuple)) or len(individual) < 3:
            continue
        values = list(individual[:10]) + [None] * max(0, 10 - len(individual))
        actor, reward, descriptor, masks, sparsity, metrics, critic, target, buffer_id, n_evals = values
        directory = snapshot / f"individual_{idx:04d}"
        directory.mkdir(exist_ok=True)
        files = {
            "actor_file": _save_weights(directory, "actor.pth", actor, archive_save_models),
            "critic_file": _save_weights(directory, "critic.pth", critic, archive_save_critics),
            "critic_target_file": _save_weights(directory, "critic_target.pth", target, archive_save_critics),
            "masks_file": "",
        }
        if archive_save_masks_npz and masks is not None:
            np.savez(directory / "masks.npz", **{
                f"layer_{i}": np.asarray(mask, dtype=np.float32)
                for i, mask in enumerate(masks)
            })
            files["masks_file"] = "masks.npz"
        info = {
            "idx": idx, "reward": float(reward) if reward is not None else None,
            "sparsity": float(sparsity) if sparsity is not None else 0.0,
            "inference_metrics": metrics if metrics else {},
            "buffer_id": buffer_id, "n_evals": int(n_evals) if n_evals is not None else 1,
            **files,
            "has_masks": bool(files["masks_file"]),
            "has_critic": bool(files["critic_file"]),
            "has_critic_target": bool(files["critic_target_file"]),
            "bd_spec_id": bd_spec.get("bd_spec_id", BD_SPEC_ID), "ablation": ablation,
        }
        metadata = next((item for item in reversed(individual[-2:])
                         if isinstance(item, dict)), {})
        info.update({key: metadata[key] for key in PROVENANCE_FIELDS
                     if metadata.get(key) is not None})
        info["env_steps_semantics"] = str(metadata.get("env_steps_semantics", ENV_STEPS_SEMANTICS))
        if descriptor is not None:
            info["obs_descriptor"] = np.asarray(descriptor, dtype=np.float32).tolist()
        _write_json(directory / "info.json", info)
        records.append(info)

    rewards = np.array([item["reward"] for item in records if item["reward"] is not None], dtype=np.float32)
    sparsities = np.array([item["sparsity"] for item in records], dtype=np.float32)
    counters = {
        "iter": int(global_iter), "tag": tag, "n_individuals": len(archive),
        "n_interactions": int(n_interactions), "elapsed_time": float(elapsed_time),
    }
    _write_json(snapshot / "summary.json", {
        **counters, "n_individuals": len(archive) if records else 0,
        "best_reward": float(rewards.max()) if rewards.size else 0.0,
        "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
        "mean_sparsity": float(sparsities.mean()) if sparsities.size else 0.0,
        "bd_spec": bd_spec, "ablation": ablation,
    })
    if qd_metrics is not None:
        _write_json(snapshot / "qd_metrics.json", qd_metrics.to_dict())
    manifest_path = Path(archive_manifest_path)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest.append({**counters, "snapshot_dir": snapshot_name,
        "bd_spec_id": bd_spec.get("bd_spec_id", BD_SPEC_ID), "ablation": ablation})
    _write_json(manifest_path, manifest)
    return str(snapshot)


def build_training_detail(
    archive_idx, network_type, sparsity, base_reward, finetuned_reward,
    warmup_episodes=0, training_time_sec=0.0, td3_result=None,
):
    """Return the refinement fields consumed by the training log."""
    result = td3_result or {}
    improvement = finetuned_reward - base_reward
    return {
        "archive_idx": archive_idx, "network_type": network_type, "sparsity": sparsity,
        "base_reward": base_reward, "finetuned_reward": finetuned_reward,
        "improvement": improvement,
        "improvement_pct": improvement / (abs(base_reward) + 1e-8) * 100.0,
        "warmup_episodes": warmup_episodes, "training_time_sec": training_time_sec,
        "update_steps_used": int(result.get("update_steps_used", 0)),
        "lr_scale": float(result.get("lr_scale", 1.0)),
        "stopped_reason": str(result.get("stopped_reason", "")),
        "initial_eval": float(result.get("initial_eval", 0.0)),
        "n_evals": int(result.get("n_evals", 0)),
    }
