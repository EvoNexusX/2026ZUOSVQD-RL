from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch

from adaptive_archive_admission import canonicalize_gait_bd, gait_bd_to_cell
from behavior_descriptor import BD_BOUNDS, BD_DIM, GRID_BINS_PER_DIM

@dataclass
class QDMetrics:
    """Metrics for the canonical svqd-rl behavioral archive."""
    qd_score: float = 0.0
    coverage: float = 0.0
    coverage_percent: float = 0.0

    archive_size: int = 0
    best_reward: float = -float('inf')
    mean_reward: float = 0.0
    std_reward: float = 0.0
    median_reward: float = 0.0

    mean_sparsity: float = 0.0
    std_sparsity: float = 0.0
    min_sparsity: float = 0.0
    max_sparsity: float = 0.0

    dense_count: int = 0
    medium_sparse_count: int = 0
    high_sparse_count: int = 0
    ultra_sparse_count: int = 0

    grid_resolution: Tuple[int, ...] = ()
    cells_occupied: int = 0
    total_cells: int = 0
    duplicate_cell_count: int = 0
    coverage_arithmetic_ok: bool = True

    timestamp: str = ""
    iteration: int = 0
    wall_time: float = 0.0

    tier_details: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iter': self.iteration,
            'generation': self.iteration,
            'wall_time': self.wall_time,
            'qd_score': self.qd_score,
            'coverage_pct': self.coverage_percent,
            'coverage_ratio': self.coverage,
            'cells_occupied': self.cells_occupied,
            'total_cells': self.total_cells,
            'archive_size': self.archive_size,
            'best_reward': self.best_reward,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'median_reward': self.median_reward,
            'mean_sparsity': self.mean_sparsity,
            'std_sparsity': self.std_sparsity,
            'sparsity_range': f'{self.min_sparsity:.2%}-{self.max_sparsity:.2%}',
            'tier_counts': {
                'dense': self.dense_count,
                'medium_sparse': self.medium_sparse_count,
                'high_sparse': self.high_sparse_count,
                'ultra_sparse': self.ultra_sparse_count,
            },
        }


class BehaviorDescriptorGrid:
    """Grid-based behavior descriptor space for coverage calculation."""

    SPARSITY_TIER_BINS = [0.0, 0.2, 0.5, 0.7, 0.9, 1.01]
    SPARSITY_TIER_NAMES = ['dense', 'low_sparse', 'medium_sparse', 'high_sparse', 'ultra_sparse']

    def __init__(self, behavior_dim=BD_DIM, grid_resolution=None, behavior_bounds=None):
        self.grid_resolution = list(grid_resolution or [GRID_BINS_PER_DIM] * BD_DIM)
        bounds = BD_BOUNDS if behavior_bounds is None else behavior_bounds
        if (behavior_dim != BD_DIM or len(self.grid_resolution) != BD_DIM
                or self.grid_resolution[0] != self.grid_resolution[1]
                or not np.array_equal(np.asarray(bounds), np.asarray(BD_BOUNDS))):
            raise ValueError("svqd-rl metrics require a square grid over [0, 1]^2")
        self.n_bins = int(self.grid_resolution[0])
        if self.n_bins < 1:
            raise ValueError("grid resolution must be positive")
        self.total_cells = self.n_bins ** BD_DIM
        self.grid = {}
        self.filled_cells = set()

    def add_solution(self, behavior_descriptor, reward, archive_index):
        cell = gait_bd_to_cell(behavior_descriptor, self.n_bins)
        if cell is None:
            raise ValueError(f"invalid canonical gait descriptor: {behavior_descriptor!r}")
        index = cell[0] + cell[1] * self.n_bins
        self.grid.setdefault(index, []).append((float(reward), int(archive_index)))
        self.filled_cells.add(index)

    def compute_coverage(self):
        return len(self.filled_cells), len(self.filled_cells) / self.total_cells

    def compute_qd_score(self):
        # Sorted cell order matches the dense grid's floating-point summation.
        total = 0.0
        for index in sorted(self.grid):
            finite = [reward for reward, _ in self.grid[index] if np.isfinite(reward)]
            if finite:
                total += max(finite)
        return total


def compute_qd_metrics(
    archive: List,
    BC: str = 'gait',
    behavior_dim: Optional[int] = None,
    grid_resolution: Optional[List[int]] = None,
    iteration: int = 0,
    wall_time: float = 0.0,
    default_dim: int = BD_DIM,
    behavior_bounds: Optional[List[Tuple[float, float]]] = None,
    ant_bc_mode: str = "2d",
) -> QDMetrics:
    metrics = QDMetrics()
    metrics.timestamp = datetime.now().isoformat()
    metrics.iteration = iteration
    metrics.wall_time = wall_time

    n = len(archive)
    metrics.archive_size = n
    if n == 0:
        return metrics

    rewards = np.zeros(n, dtype=np.float64)
    sparsities = np.zeros(n, dtype=np.float32)
    bd_raw: List[Optional[np.ndarray]] = [None] * n

    for i, item in enumerate(archive):
        if len(item) > 1 and item[1] is not None:
            rewards[i] = float(item[1])
        if len(item) > 4 and item[4] is not None:
            sparsities[i] = float(item[4])
        if len(item) > 2 and item[2] is not None:
            desc = item[2]
            if isinstance(desc, torch.Tensor):
                desc = desc.detach().cpu().numpy()
            bd_raw[i] = np.asarray(desc, dtype=np.float32).flatten()
        else:
            bd_raw[i] = None

    finite_rewards = rewards[np.isfinite(rewards)] if n > 0 else rewards
    n_finite = int(finite_rewards.size)
    metrics.best_reward = float(np.max(finite_rewards)) if n_finite > 0 else -float('inf')
    metrics.mean_reward = float(np.mean(finite_rewards)) if n_finite > 0 else 0.0
    metrics.std_reward = float(np.std(finite_rewards)) if n_finite > 0 else 0.0
    metrics.median_reward = float(np.median(finite_rewards)) if n_finite > 0 else 0.0

    # Average sparsity over all elites, including dense policies.
    metrics.mean_sparsity = float(np.mean(sparsities)) if n > 0 else 0.0
    metrics.std_sparsity = float(np.std(sparsities)) if n > 0 else 0.0
    metrics.min_sparsity = float(np.min(sparsities)) if n > 0 else 0.0
    metrics.max_sparsity = float(np.max(sparsities)) if n > 0 else 0.0

    tier_bins = np.array(BehaviorDescriptorGrid.SPARSITY_TIER_BINS, dtype=np.float32)
    tier_ids = np.digitize(sparsities, tier_bins[1:-1], right=False)

    for t_idx, t_name in enumerate(BehaviorDescriptorGrid.SPARSITY_TIER_NAMES):
        tier_mask = (tier_ids == t_idx)
        count = int(np.sum(tier_mask))
        setattr(metrics, f'{t_name}_count', count)
        tier_rewards = rewards[tier_mask]
        tier_sparsities_arr = sparsities[tier_mask]
        metrics.tier_details[t_name] = {
            'count': count,
            'mean_reward': float(np.mean(tier_rewards)) if len(tier_rewards) > 0 else 0.0,
            'max_reward': float(np.max(tier_rewards)) if len(tier_rewards) > 0 else -float('inf'),
            'mean_sparsity': float(np.mean(tier_sparsities_arr)) if len(tier_sparsities_arr) > 0 else 0.0,
        }

    if BC != 'gait':
        raise ValueError("Only the canonical gait descriptor is supported")
    behavior_dim = BD_DIM
    if grid_resolution is None:
        grid_resolution = [GRID_BINS_PER_DIM] * BD_DIM
    behavior_bounds = list(BD_BOUNDS)
    valid_bd_idx = []
    valid_rows = []
    for i, raw_bd in enumerate(bd_raw):
        bd, status = canonicalize_gait_bd(raw_bd)
        if status == 'valid' and bd is not None:
            valid_bd_idx.append(i)
            valid_rows.append(bd)
    bd_array = (np.stack(valid_rows).astype(np.float32) if valid_rows
                else np.zeros((0, BD_DIM), dtype=np.float32))

    metrics.valid_bd_count = len(valid_bd_idx)

    grid = BehaviorDescriptorGrid(
        behavior_dim=behavior_dim,
        grid_resolution=grid_resolution,
        behavior_bounds=behavior_bounds,
    )
    for k, i in enumerate(valid_bd_idx):
        grid.add_solution(bd_array[k], rewards[i], i)

    cells_occupied, coverage_ratio = grid.compute_coverage()
    metrics.grid_resolution = tuple(grid.grid_resolution)
    metrics.cells_occupied = cells_occupied
    metrics.total_cells = grid.total_cells
    metrics.coverage = coverage_ratio
    metrics.coverage_percent = coverage_ratio * 100.0
    metrics.duplicate_cell_count = max(0, metrics.valid_bd_count - cells_occupied)

    metrics.qd_score = grid.compute_qd_score()

    # Cell count, coverage ratio and coverage percentage must agree.
    expected_ratio = (float(cells_occupied) / float(metrics.total_cells)
                      if metrics.total_cells > 0 else 0.0)
    metrics.coverage_arithmetic_ok = bool(
        abs(expected_ratio - float(metrics.coverage)) <= 1e-9
        and abs(float(metrics.coverage_percent) - 100.0 * float(metrics.coverage)) <= 1e-6
    )

    return metrics

compute_archive_qd_metrics = compute_qd_metrics


def print_qd_metrics(metrics: QDMetrics, prefix: str = ""):
    print(f"{prefix}=== QD Metrics (Iteration {metrics.iteration}) ===")
    print(f"{prefix}  QD-Score:   {metrics.qd_score:.2f}")
    print(f"{prefix}  Coverage:   {metrics.coverage_percent:.1f}% ({metrics.cells_occupied}/{metrics.total_cells} cells)")
    print(f"{prefix}  Archive:    {metrics.archive_size} solutions")
    print(f"{prefix}  Best Reward: {metrics.best_reward:.2f}")
    print(f"{prefix}  Mean Reward: {metrics.mean_reward:.2f} ± {metrics.std_reward:.2f}")
    print(f"{prefix}  Sparsity:    {metrics.mean_sparsity:.1%} [{metrics.min_sparsity:.0%}-{metrics.max_sparsity:.0%}]")
    print(f"{prefix}  Tiers:       dense={metrics.dense_count}, "
          f"med={metrics.medium_sparse_count}, "
          f"high={metrics.high_sparse_count}, "
          f"ultra={metrics.ultra_sparse_count}")
    if metrics.wall_time > 0:
        print(f"{prefix}  Wall Time:   {metrics.wall_time:.1f}s")
    print(f"{prefix}" + "=" * 40)
