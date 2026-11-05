"""Separate rollout completion from the replay TD mask.
Either flag ends an episode; only termination zeros the bootstrap. Legacy
four-field step results obtain truncation from the environment wrapper flag."""
from typing import Any, Tuple

import numpy as np

__all__ = [
    "episode_done",
    "replay_terminal",
    "time_limit_truncated",
    "unpack_step",
]


def _time_limit_truncated(env) -> bool:
    """Read the legacy wrapper truncation flag, or return False if unavailable."""
    if env is None:
        return False
    try:
        return bool(getattr(env, 'truncated', False))
    except Exception:
        return False


def episode_done(terminated, truncated) -> bool:
    """End/reset the episode on either flag."""
    return bool(terminated) or bool(truncated)


def replay_terminal(terminated, truncated=None) -> float:
    """Return the termination TD mask; pure time-limit truncation still bootstraps."""
    del truncated  # deliberately unused: termination alone decides the mask.
    return 1.0 if bool(terminated) else 0.0


def unpack_step(step_out, env=None) -> Tuple[Any, Any, bool, bool, dict]:
    """Normalise a raw `env.step(...)` return into the 5-field canonical form."""
    if isinstance(step_out, tuple) and len(step_out) == 5:
        next_obs, reward, terminated, truncated, info = step_out
        term_arr = np.asarray(terminated)
        trunc_arr = np.asarray(truncated)
        if term_arr.ndim > 0:
            # A vectorized rollout ends when every lane is done.
            done_arr = term_arr.astype(bool) | trunc_arr.astype(bool)
            terminated_b = bool(done_arr.all())
            truncated_b = bool(trunc_arr.astype(bool).all())
            info = info if isinstance(info, dict) else {}
            info['_done_per_lane'] = done_arr
        else:
            terminated_b = bool(term_arr)
            truncated_b = bool(trunc_arr)
        return next_obs, reward, terminated_b, truncated_b, (
            info if isinstance(info, dict) else {}
        )

    if isinstance(step_out, tuple) and len(step_out) == 4:
        next_obs, reward, done, info = step_out
        truncated_b = bool(done) and _time_limit_truncated(env)
        terminated_b = bool(done) and not truncated_b
        return next_obs, reward, terminated_b, truncated_b, (
            info if isinstance(info, dict) else {}
        )

    raise ValueError(
        f"unsupported env.step return with {len(step_out) if isinstance(step_out, tuple) else type(step_out)} "
        "elements; expected a 4-tuple or 5-tuple"
    )
