from typing import List, Sequence, Optional, Dict, Any, Tuple
import numpy as np

SUPPORTED_GYM_ENVS = {
    'Hopper-v5',
    'Walker2d-v5',
    'HalfCheetah-v5',
    'Ant-v5',
    'Humanoid-v5',
}


def _norm_env_name(env_name: str) -> str:
    if env_name not in SUPPORTED_GYM_ENVS:
        raise ValueError(
            f"Unsupported gait descriptor environment {env_name!r}; "
            f"supported environments are {sorted(SUPPORTED_GYM_ENVS)}."
        )
    return env_name

ENV_FOOT_BODY_IDS: Dict[str, List[int]] = {
    'Hopper-v5':      [4],
    'Walker2d-v5':    [4, 7],
    'HalfCheetah-v5': [4, 7],
    'Ant-v5':         [4, 7, 10, 13],
    'Humanoid-v5':    [6, 9],
}

ENV_FOOT_Z_THRESHOLD: Dict[str, float] = {
    'Hopper-v5':      0.05,
    'Walker2d-v5':    0.10,
    'HalfCheetah-v5': 0.10,
    'Ant-v5':         0.20,
    'Humanoid-v5':    0.20,
}

ENV_VX_MAX: Dict[str, float] = {
    'Hopper-v5':      5.0,
    'Walker2d-v5':    6.0,
    'HalfCheetah-v5': 12.0,
    'Ant-v5':         5.0,
    'Humanoid-v5':    5.0,
}

ENV_CONTROL_DT: Dict[str, float] = {
    'Hopper-v5':      0.008,
    'Walker2d-v5':    0.008,
    'HalfCheetah-v5': 0.05,
    'Ant-v5':         0.05,
    'Humanoid-v5':    0.015,
}

BD_SPEC_ID: str = 'gait2d_xpos_v2'
BD_DIM: int = 2
BD_BOUNDS: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
GRID_BINS_PER_DIM: int = 50
BD_NAMES: Tuple[str, str] = ('v_norm', 'foot_duty')


def get_v_x_max(env_name: str) -> float:
    return ENV_VX_MAX.get(_norm_env_name(env_name), 5.0)


def get_control_dt(env_name: str) -> float:
    normalized = _norm_env_name(env_name)
    if normalized not in ENV_CONTROL_DT:
        raise ValueError(f"No control timestep configured for environment: {env_name}")
    return float(ENV_CONTROL_DT[normalized])


def resolve_control_dt(env=None, env_name: Optional[str] = None) -> float:
    if env is not None:
        try:
            dt = float(env.unwrapped.dt)
            if np.isfinite(dt) and dt > 0.0:
                return dt
        except Exception:
            pass
    if env_name is None:
        raise ValueError("env or env_name is required to resolve control dt")
    return get_control_dt(env_name)


def get_foot_body_ids(env_name: str) -> List[int]:
    return list(ENV_FOOT_BODY_IDS.get(_norm_env_name(env_name), []))


def get_foot_z_threshold(env_name: str) -> float:
    return ENV_FOOT_Z_THRESHOLD.get(_norm_env_name(env_name), 0.10)


def get_bd_spec(env_name: str, control_dt: Optional[float] = None) -> Dict[str, Any]:
    normalized = _norm_env_name(env_name)
    return {
        'bd_spec_id': BD_SPEC_ID,
        'env_name': normalized,
        'bd_dim': BD_DIM,
        'bd_names': list(BD_NAMES),
        'bd_bounds': [list(bound) for bound in BD_BOUNDS],
        'grid_bins_per_dim': GRID_BINS_PER_DIM,
        'total_cells': GRID_BINS_PER_DIM ** BD_DIM,
        'foot_body_ids': get_foot_body_ids(normalized),
        'v_x_max': get_v_x_max(normalized),
        'control_dt': float(control_dt) if control_dt is not None else get_control_dt(normalized),
        'z_threshold': get_foot_z_threshold(normalized),
        'contact_source': 'xpos_z_threshold',
    }


def compute_average_velocity(
    x_traj: Sequence[float],
    dt: Optional[float] = None,
    v_x_max: Optional[float] = None,
    env_name: Optional[str] = None,
    normalize: bool = True,
) -> float:
    if len(x_traj) < 2:
        return 0.0
    if dt is None:
        if env_name is None:
            raise ValueError("dt or env_name is required to compute physical velocity")
        dt = get_control_dt(env_name)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"control dt must be finite and positive, got {dt!r}")
    T = len(x_traj) - 1
    total_time = T * dt
    if total_time <= 0:
        return 0.0
    dx = float(x_traj[-1] - x_traj[0])
    v_raw = max(0.0, dx / total_time)
    if not normalize:
        return v_raw
    if v_x_max is None and env_name is not None:
        v_x_max = get_v_x_max(env_name)
    if v_x_max is None or v_x_max <= 0:
        raise ValueError(f"v_x_max must be finite and positive, got {v_x_max!r}")
    return float(np.clip(v_raw / v_x_max, 0.0, 1.0))


def compute_per_foot_contact(
    xpos_traj: Sequence[np.ndarray],
    foot_body_ids: Sequence[int],
    z_threshold: float = 0.05,
) -> np.ndarray:
    n_foot = len(foot_body_ids)
    if len(xpos_traj) < 2 or n_foot == 0:
        return np.zeros(n_foot, dtype=np.float32)
    xs_list = []
    for x in xpos_traj:
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 2 and a.shape[1] >= 3:
            xs_list.append(a)
    if not xs_list:
        return np.zeros(n_foot, dtype=np.float32)
    reference_shape = xs_list[0].shape
    if any(frame.shape != reference_shape for frame in xs_list[1:]):
        raise ValueError(
            "xpos trajectory contains inconsistent frame shapes: "
            f"expected {reference_shape}, got {[frame.shape for frame in xs_list]}"
        )
    xs = np.stack(xs_list, axis=0)
    T, n_bodies, _ = xs.shape
    idxs = np.array([b if b >= 0 else n_bodies + b for b in foot_body_ids], dtype=np.int64)
    valid = (idxs >= 0) & (idxs < n_bodies)
    if not valid.any():
        return np.zeros(n_foot, dtype=np.float32)
    contacts = xs[:, idxs[valid], 2] <= z_threshold
    counts = contacts.sum(axis=0).astype(np.float32)
    total = np.zeros(n_foot, dtype=np.float32)
    total[valid] = counts
    return np.clip(total / float(T), 0.0, 1.0)


def compute_average_foot_contact(
    xpos_traj: Sequence[np.ndarray],
    foot_body_ids: Optional[Sequence[int]] = None,
    z_threshold: Optional[float] = None,
    env_name: Optional[str] = None,
) -> float:
    if foot_body_ids is None:
        if env_name is None:
            raise ValueError("Must provide foot_body_ids or env_name")
        foot_body_ids = get_foot_body_ids(env_name)
    if z_threshold is None:
        z_threshold = get_foot_z_threshold(env_name) if env_name else 0.10
    c_k = compute_per_foot_contact(xpos_traj, foot_body_ids, z_threshold)
    if len(c_k) == 0:
        return 0.0
    return float(np.mean(c_k))


def compute_bd(
    x_traj: Optional[Sequence[float]] = None,
    xpos_traj: Optional[Sequence[np.ndarray]] = None,
    foot_body_ids: Optional[Sequence[int]] = None,
    v_x_max: Optional[float] = None,
    z_threshold: Optional[float] = None,
    dt: Optional[float] = None,
    *,
    env_name: Optional[str] = None,
) -> np.ndarray:
    if env_name is not None:
        if foot_body_ids is None:
            foot_body_ids = get_foot_body_ids(env_name)
        if v_x_max is None:
            v_x_max = get_v_x_max(env_name)
        if z_threshold is None:
            z_threshold = get_foot_z_threshold(env_name)

    if x_traj is None or xpos_traj is None or foot_body_ids is None:
        raise ValueError(
            "compute_bd requires (x_traj, xpos_traj, foot_body_ids). "
            "Or pass env_name to use defaults."
        )
    if z_threshold is None:
        z_threshold = 0.10

    v = compute_average_velocity(
        x_traj, dt=dt, v_x_max=v_x_max, env_name=env_name, normalize=True,
    )
    f = compute_average_foot_contact(xpos_traj, foot_body_ids, z_threshold)
    bd = np.array([v, f], dtype=np.float32)
    if bd.shape != (BD_DIM,) or not np.all(np.isfinite(bd)):
        raise RuntimeError(f"invalid gait descriptor generated: {bd!r}")
    return np.clip(bd, 0.0, 1.0).astype(np.float32, copy=False)


def aggregate_bd_from_episodes(
    episodes_x: List[Sequence[float]],
    episodes_xpos: List[Sequence[np.ndarray]],
    env_name: str,
    dt: Optional[float] = None,
) -> np.ndarray:
    if len(episodes_x) != len(episodes_xpos):
        raise ValueError(
            "gait descriptor episode count mismatch: "
            f"x={len(episodes_x)}, xpos={len(episodes_xpos)}"
        )
    if not episodes_x:
        raise ValueError("cannot aggregate gait descriptor from empty episode trajectories")
    bd_per_ep = [
        compute_bd(x_traj=x_traj, xpos_traj=xpos_traj, env_name=env_name, dt=dt)
        for x_traj, xpos_traj in zip(episodes_x, episodes_xpos)
    ]
    return np.mean(np.stack(bd_per_ep, axis=0), axis=0).astype(np.float32)

ENV_BEHAVIOR_INFO: Dict[str, Dict[str, Any]] = {
    'Hopper-v5': {
        'name': 'Hopper',
        'descriptor_dim': 2,
        'bd_names': ['v_norm', 'foot_duty'],
        'n_feet': 1,
        'description': '2D BD: forward velocity + single-foot contact duty',
    },
    'Walker2d-v5': {
        'name': 'Walker2d',
        'descriptor_dim': 2,
        'bd_names': ['v_norm', 'foot_duty'],
        'n_feet': 2,
        'description': '2D BD: forward velocity + mean foot contact duty',
    },
    'HalfCheetah-v5': {
        'name': 'HalfCheetah',
        'descriptor_dim': 2,
        'bd_names': ['v_norm', 'foot_duty'],
        'n_feet': 2,
        'description': '2D BD: forward velocity + mean foot contact duty',
    },
    'Ant-v5': {
        'name': 'Ant',
        'descriptor_dim': 2,
        'bd_names': ['v_norm', 'foot_duty'],
        'n_feet': 4,
        'description': '2D BD: forward velocity + 4-foot mean contact duty',
    },
    'Humanoid-v5': {
        'name': 'Humanoid',
        'descriptor_dim': 2,
        'bd_names': ['v_norm', 'foot_duty'],
        'n_feet': 2,
        'description': '2D BD: forward velocity + mean foot contact duty',
    },
}


def get_env_behavior_info(env_name: str) -> Dict[str, Any]:
    normalized = _norm_env_name(env_name)
    if normalized in ENV_BEHAVIOR_INFO:
        return dict(ENV_BEHAVIOR_INFO[normalized])
    return {
        'name': 'Unknown',
        'descriptor_dim': BD_DIM,
        'bd_names': ['v_norm', 'foot_duty'],
        'n_feet': 1,
        'description': '2D BD: forward velocity + mean foot contact duty',
    }
