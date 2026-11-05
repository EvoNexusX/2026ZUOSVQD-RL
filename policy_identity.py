"""Weight fingerprints identify measurements; monotonic entry IDs identify training history."""
import hashlib
from typing import Optional

import numpy as np

__all__ = ["policy_fingerprint", "StableEntryId", "entry_stable_id"]


def _hash_update(digest, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())


def policy_fingerprint(network) -> Optional[str]:
    """Hash actor parameters and buffers, excluding masks already applied by pruning.
    Return None if no parameters are available; mask_fingerprint identifies masks separately."""
    if network is None:
        return None
    digest = hashlib.sha1()
    touched = False
    try:
        params = network.parameters()
    except Exception:
        params = None
    if params is not None:
        for p in params:
            _hash_update(digest, p.detach().cpu().numpy())
            touched = True
        try:
            for _, buf in network.named_buffers():
                _hash_update(digest, buf.detach().cpu().numpy())
        except Exception:
            pass
    if not touched:
        return None
    return digest.hexdigest()[:16]


class StableEntryId:
    """Monotonic policy identities; archive-slot replacements receive new IDs."""

    def __init__(self, start: int = 1) -> None:
        self._next = int(start)

    def mint(self) -> int:
        value = self._next
        self._next += 1
        return value

def entry_stable_id(individual, fallback: Optional[int] = None) -> Optional[int]:
    """Read an individual's stable id from its trailing metadata slot."""
    if individual is None:
        return fallback
    try:
        if isinstance(individual, dict):
            value = individual.get('stable_id')
            return int(value) if value is not None else fallback
        if len(individual) >= 1:
            value = individual[-1]
            if isinstance(value, dict) and value.get('stable_id') is not None:
                return int(value['stable_id'])
        # The trailing metadata dictionary carries the stable identity.
        for slot in (individual[-1], individual[-2] if len(individual) > 1 else None):
            if isinstance(slot, dict) and slot.get('stable_id') is not None:
                return int(slot['stable_id'])
    except Exception:
        pass
    return fallback


