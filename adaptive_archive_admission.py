import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Any

from behavior_descriptor import BD_DIM, GRID_BINS_PER_DIM


def canonicalize_gait_bd(value, reject_zero: bool = False) -> Tuple[Optional[np.ndarray], str]:
    """Validate and normalize a gait BD to the canonical 2-D [0, 1] contract.

    Returns ``(bd, status)`` where status is one of ``valid``, ``missing``,
    ``short``, ``nonfinite``, ``zero``, or ``invalid``. Out-of-bounds values
    are clipped, matching archive admission and QD metrics.
    """
    if value is None:
        return None, 'missing'
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        arr = np.asarray(value, dtype=np.float32).flatten()
    except Exception:
        return None, 'invalid'
    if arr.size < BD_DIM:
        return None, 'short'
    arr = arr[:BD_DIM]
    if not np.all(np.isfinite(arr)):
        return None, 'nonfinite'
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    if reject_zero and float(np.linalg.norm(arr)) < 1e-8:
        return None, 'zero'
    return arr, 'valid'


def gait_bd_to_cell(value, n_bins: int = GRID_BINS_PER_DIM) -> Optional[Tuple[int, int]]:
    """Map a valid canonical gait BD to a fixed 2-D MAP-Elites cell."""
    bd, status = canonicalize_gait_bd(value)
    if status != 'valid' or bd is None:
        return None
    n = max(int(n_bins), 1)
    coords = []
    for v in bd:
        fv = float(v)
        if fv >= 1.0:
            coords.append(n - 1)
            continue
        scaled = fv * n
        nearest = round(scaled)
        tolerance = np.finfo(np.float32).eps * max(n, 1) * 2.0
        if abs(scaled - nearest) <= tolerance:
            scaled = float(nearest)
        coords.append(max(0, min(int(np.floor(scaled)), n - 1)))
    return tuple(coords)


def archive_occupied_cells(archive: List, n_bins: int = GRID_BINS_PER_DIM) -> set:
    """Return occupied cells, ignoring entries without valid behavioral descriptors."""
    cells = set()
    for individual in archive or []:
        raw_bd = individual[2] if individual is not None and len(individual) > 2 else None
        bd, status = canonicalize_gait_bd(raw_bd)
        if status != 'valid' or bd is None:
            continue
        cell = gait_bd_to_cell(bd, n_bins)
        if cell is not None:
            cells.add(cell)
    return cells


def archive_cell_statistics(archive: List, n_bins: int = GRID_BINS_PER_DIM) -> Dict[str, Any]:
    """Return auditable validity and fixed-grid coverage statistics."""
    cells = []
    valid = zero = invalid = 0
    for individual in archive or []:
        raw_bd = individual[2] if individual is not None and len(individual) > 2 else None
        bd, status = canonicalize_gait_bd(raw_bd)
        if status == 'valid' and bd is not None:
            valid += 1
            cell = gait_bd_to_cell(bd, n_bins)
            if cell is not None:
                cells.append(cell)
        elif status == 'zero':
            zero += 1
        else:
            invalid += 1
    occupied = len(set(cells))
    total = max(int(n_bins), 1) ** 2
    return {
        'bd_valid': valid,
        'bd_zero': zero,
        'bd_invalid': invalid,
        'occupied_cells': occupied,
        'duplicate_cells': max(0, len(cells) - occupied),
        'total_cells': total,
        'coverage_ratio': occupied / max(total, 1),
        'coverage_percent': occupied / max(total, 1) * 100.0,
        'cells': set(cells),
    }


class AdaptiveCriticArchiveAdmission:
    """MAP-Elites admission: retain the highest-return policy per fixed 2-D cell."""

    def __init__(self, n_bins: int = GRID_BINS_PER_DIM,
                 bd_replace_radius: float = 0.0,
                 min_replace_margin: float = 0.0):
        self.n_bins = int(n_bins)
        self.bd_replace_radius = float(bd_replace_radius)
        self.min_replace_margin = float(min_replace_margin)
        self._cell_best: Dict[Tuple[int, int], Tuple[float, int]] = {}
        self._idx_cell: Dict[int, Tuple[int, int]] = {}

    def rebuild_cache(self, archive: List):
        """Rebuild the cell/idx caches after a bulk archive mutation."""
        self._cell_best.clear()
        self._idx_cell.clear()
        for i, ind in enumerate(archive):
            bd = self._bd(ind)
            if bd is None:
                continue
            cell = gait_bd_to_cell(bd, self.n_bins)
            if cell is None:
                continue
            rew = self._reward(ind)
            self._idx_cell[i] = cell
            prev = self._cell_best.get(cell)
            if prev is None or rew > prev[0]:
                self._cell_best[cell] = (rew, i)

    def _update_cache_on_admit(self, idx, cell, reward):
        self._idx_cell[idx] = cell
        prev = self._cell_best.get(cell)
        if prev is None or reward > prev[0]:
            self._cell_best[cell] = (reward, idx)

    def _update_cache_on_replace(self, old_idx, new_idx, cell, new_reward):
        old_cell = self._idx_cell.pop(old_idx, None)
        if old_cell is not None:
            prev = self._cell_best.get(old_cell)
            if prev is not None and prev[1] == old_idx:
                self._cell_best.pop(old_cell, None)
        self._idx_cell[new_idx] = cell
        prev = self._cell_best.get(cell)
        if prev is None or new_reward > prev[0] or new_idx == prev[1]:
            self._cell_best[cell] = (new_reward, new_idx)

    @property
    def total_cells(self) -> int:
        return self.n_bins ** 2

    @staticmethod
    def _bd(individual) -> Optional[np.ndarray]:
        if individual is None or len(individual) < 3:
            return None
        bd, status = canonicalize_gait_bd(individual[2])
        return bd if status == 'valid' else None

    @staticmethod
    def _reward(individual) -> float:
        if individual is None or len(individual) < 2:
            return float('-inf')
        return float(individual[1])

    def should_admit(self, archive, candidate_bd, candidate_reward):
        info = {
            'reason': 'unknown',
            'cell': None,
            'cell_best_reward': None,
            'cell_replace_idx': -1,
            'n_cells_filled': len(self._cell_best),
            'n_cells_total': self.n_bins ** 2,
            'replace_via_distance': False,
            'bd_distance_to_replaced': None,
        }

        if len(self._idx_cell) != len(archive):
            self.rebuild_cache(archive)

        if not (isinstance(candidate_reward, (int, float)) and np.isfinite(candidate_reward)):
            info['reason'] = 'reward_nonfinite'
            return False, -1, info

        if candidate_bd is None:
            info['reason'] = 'no_bd'
            info['bd_status'] = 'missing'
            return False, -1, info
        cand_bd, bd_status = canonicalize_gait_bd(candidate_bd)
        info['bd_status'] = bd_status
        if cand_bd is None:
            info['reason'] = f'invalid_bd:{bd_status}'
            return False, -1, info

        cand_cell = gait_bd_to_cell(cand_bd, self.n_bins)
        if cand_cell is None:
            info['reason'] = 'cell_map_fail'
            return False, -1, info
        info['cell'] = cand_cell

        target_cell, target_idx, target_reward, distance = self._find_replacement_target(
            cand_cell, cand_bd
        )
        if target_cell is None:
            info['reason'] = 'new_cell_admit'
            return True, -1, info

        info['cell_best_reward'] = float(target_reward)
        info['cell_replace_idx'] = int(target_idx)
        info['bd_distance_to_replaced'] = float(distance)
        info['replace_via_distance'] = bool(distance > 0.0)

        if float(candidate_reward) >= target_reward + self.min_replace_margin:
            info['reason'] = 'replace_within_cell' if distance == 0.0 else 'replace_via_distance'
            return True, int(target_idx), info
        info['reason'] = 'reject_lower_reward'
        return False, -1, info

    def _find_replacement_target(self, cand_cell, cand_bd):
        """Pick a replacement target. Exact-cell wins; if bd_replace_radius > 0,
        fall back to the nearest occupied cell within radius * cell_size.
        """
        cell_info = self._cell_best.get(cand_cell)
        if cell_info is not None:
            best_reward, best_idx = cell_info
            return cand_cell, best_idx, float(best_reward), 0.0

        if self.bd_replace_radius <= 0.0:
            return None, None, float('-inf'), float('inf')

        r = float(self.bd_replace_radius)
        cell_size = 1.0 / max(int(self.n_bins), 1)
        max_steps = max(1, int(np.ceil(r * (1.0 / cell_size))))
        best_target = None
        best_reward = float('-inf')
        best_dist = float('inf')
        cx, cy = cand_cell
        for dx in range(-max_steps, max_steps + 1):
            nx = cx + dx
            if nx < 0 or nx >= self.n_bins:
                continue
            for dy in range(-max_steps, max_steps + 1):
                ny = cy + dy
                if ny < 0 or ny >= self.n_bins:
                    continue
                cell = (nx, ny)
                if cell == cand_cell:
                    continue
                cand = self._cell_best.get(cell)
                if cand is None:
                    continue
                cell_center = np.array([(nx + 0.5) * cell_size,
                                        (ny + 0.5) * cell_size], dtype=np.float32)
                d = float(np.linalg.norm(cand_bd - cell_center))
                if d > r:
                    continue
                rew = float(cand[0])
                if (rew > best_reward) or (rew == best_reward and d < best_dist):
                    best_reward = rew
                    best_dist = d
                    best_target = (cell, int(cand[1]))
        if best_target is None:
            return None, None, float('-inf'), float('inf')
        return best_target[0], best_target[1], best_reward, best_dist

    def commit_admit(self, archive, idx, cell, reward, old_idx=None):
        if old_idx is not None and 0 <= old_idx < len(archive):
            self._update_cache_on_replace(old_idx, idx, cell, reward)
        else:
            self._update_cache_on_admit(idx, cell, reward)

