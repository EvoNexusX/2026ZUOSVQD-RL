import numpy as np
import torch
from typing import List, Set


def update_archive_bcs(archive: List):
    """Refresh cached BD entries on each archive item (cast to float32)."""
    for idx, item in enumerate(archive):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        bd = item[2]
        if isinstance(bd, torch.Tensor):
            bd_val = bd.detach().cpu().numpy()
        else:
            bd_val = np.asarray(bd) if bd is not None else None
        if bd_val is not None:
            bd_val = bd_val.flatten().astype(np.float32)
            archive[idx][2] = bd_val


def sync_archive_buffers(archive: List, sparse_optimizer) -> Set[str]:
    """Keep sparse_optimizer's per-cell replay buffers in sync with the archive."""
    keep_ids = set()
    for item in archive:
        if isinstance(item, (list, tuple)) and len(item) > 8:
            bid = item[8]
            if isinstance(bid, str) and bid:
                keep_ids.add(bid)

    if hasattr(sparse_optimizer, 'register_archive_mask_id'):
        for bid in keep_ids:
            sparse_optimizer.register_archive_mask_id(bid)
    if hasattr(sparse_optimizer, '_archive_mask_ids'):
        sparse_optimizer._archive_mask_ids = set(keep_ids)
    if hasattr(sparse_optimizer, 'prune_sparse_memories'):
        sparse_optimizer.prune_sparse_memories(keep_ids)

    return keep_ids