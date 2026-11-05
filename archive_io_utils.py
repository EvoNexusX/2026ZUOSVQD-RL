import hashlib
from typing import Any, List
import numpy as np


def make_json_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    try:
        return str(obj)
    except Exception:
        return None


def mask_fingerprint(masks: List[np.ndarray]) -> str:
    if masks is None or len(masks) == 0:
        return "empty"
    flat_data = []
    for m in masks:
        if m is not None:
            arr = np.asarray(m)
            flat_data.append(arr.tobytes())
    if not flat_data:
        return "empty"
    combined = b''.join(flat_data)
    return hashlib.md5(combined).hexdigest()[:12]