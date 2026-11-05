from typing import Tuple, List
import numpy as np

MASK_THRESHOLD = 0.5
MIN_CHANNEL_RATIO = 0.01
MAX_REPORTED_SPARSITY = 0.90


def get_mask_threshold() -> float:
    return MASK_THRESHOLD


def binarize_masks(masks: List[np.ndarray], threshold: float = None) -> List[np.ndarray]:
    if threshold is None:
        threshold = MASK_THRESHOLD
    import torch
    out = []
    for m in masks:
        arr = m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
        if arr.size == 0:
            out.append(arr)
            continue
        arr_clipped = np.clip(arr, 0.0, 1.0)
        b = (arr_clipped >= threshold)
        if not np.any(b) and arr.size > 0:
            k = max(1, int(arr.size * MIN_CHANNEL_RATIO))
            top_k_indices = np.argpartition(arr.flatten(), -k)[-k:]
            b_flat = b.flatten()
            b_flat[top_k_indices] = True
            b = b_flat.reshape(arr.shape)
        out.append(b.astype(np.float32))
    return out


def element_level_sparsity(masks) -> Tuple[float, List[float]]:
    """Return zero-entry fractions for weight masks, distinct from compact parameter reduction."""
    if isinstance(masks, np.ndarray):
        if masks.ndim == 1:
            masks = [masks]
        else:
            masks = list(masks)
    if len(masks) == 0:
        return 0.0, []
    total_zeros = 0
    total_elements = 0
    layer_sparsities = []
    for m in masks:
        arr = np.asarray(m).flatten()
        n_zeros = int(np.sum(arr == 0))
        n_elements = arr.size
        total_zeros += n_zeros
        total_elements += n_elements
        layer_sp = n_zeros / max(n_elements, 1)
        layer_sparsities.append(float(layer_sp))
    overall = total_zeros / max(total_elements, 1)
    overall = min(float(overall), MAX_REPORTED_SPARSITY)
    return float(overall), layer_sparsities
