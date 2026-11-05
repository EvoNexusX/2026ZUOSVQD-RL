from copy import deepcopy
from typing import Optional, List, Sequence, Tuple
import numpy as np
import torch

from replay_flags import replay_terminal, unpack_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_max_episode_steps(env, override):
    if override is not None:
        return int(override)
    if hasattr(env, '_max_episode_steps'):
        return int(env._max_episode_steps)
    spec = getattr(env, 'spec', None)
    if spec is not None:
        return int(getattr(spec, 'max_episode_steps', 1000))
    return 1000


def resolve_episode_seeds(
    n_episodes: int,
    base_seed: Optional[int],
    step_offset: int = 0,
    episode_seeds: Optional[Sequence[int]] = None,
) -> List[Optional[int]]:
    """Seeds for episodes 0..n-1, independent of how long earlier episodes lived.

    `episode_seeds` wins when supplied: admission needs an explicit, logged,
    reproducible list rather than a formula. Otherwise the CRN rule
    `base_seed + step_offset + k` applies. Returning `None` means "reset
    unseeded", which is the honest representation of an unseeded evaluation.
    """
    n = max(0, int(n_episodes))
    if episode_seeds is not None:
        # None means an unseeded reset; preserve its position in the episode list.
        raw = list(episode_seeds)[:n]
        seeds = [int(s) if s is not None else None for s in raw]
        while len(seeds) < n:
            seeds.append(None)
        return seeds
    if base_seed is None or int(base_seed) < 0:
        return [None] * n
    return [int(base_seed) + int(step_offset) + int(k) for k in range(n)]


def evaluate_actor(
    actor,
    env,
    max_action: float,
    memory=None,
    n_episodes: int = 1,
    random: bool = False,
    noise=None,
    render: bool = False,
    base_seed: Optional[int] = None,
    step_offset: int = 0,
    max_episode_steps: Optional[int] = None,
    return_rollout: bool = False,
    return_bd_traj: bool = False,
    env_name: str = None,
    bc_type: str = "gait",
    ant_bc_mode: str = "2d",
    episode_seeds: Optional[Sequence[int]] = None,
    step_counter=None,
):
    # The third rollout item is retained for tuple compatibility. Canonical BD
    # is computed from x_traj_list/xpos_traj_list by behavior_descriptor.py.
    try:
        actor_device = next(actor.parameters()).device
    except Exception:
        actor_device = device

    max_steps = _resolve_max_episode_steps(env, max_episode_steps)

    if not random:
        def policy(state):
            state = np.asarray(state, dtype=np.float32)
            if state.ndim == 1:
                state = state.reshape(1, -1)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=actor_device)
            action = actor(state_t).cpu().data.numpy()
            if action.ndim == 2 and action.shape[0] == 1:
                action = action[0]
            return np.clip(action, -max_action, max_action)

    scores: List[float] = []
    steps = 0
    rollout_datas: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]] = []
    x_traj_list: List[List[float]] = []
    xpos_traj_list: List[List[np.ndarray]] = []


    ep_seeds = resolve_episode_seeds(n_episodes, base_seed, step_offset, episode_seeds)

    for ep_idx in range(n_episodes):
        score = 0.0
        ep_rollout: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        ep_x_traj: List[float] = []
        ep_xpos_traj: List[np.ndarray] = []

        # Episode k uses the same explicit reset seed for every compared candidate.
        ep_seed = ep_seeds[ep_idx] if ep_idx < len(ep_seeds) else None
        if ep_seed is not None:
            reset_out = env.reset(seed=int(ep_seed))
        else:
            reset_out = env.reset()

        obs = deepcopy(reset_out[0] if isinstance(reset_out, tuple) else reset_out)
        done = False

        if return_bd_traj:
            _unwrapped = env.unwrapped
            ep_x_traj.append(float(_unwrapped.data.qpos[0]))
            ep_xpos_traj.append(np.asarray(_unwrapped.data.xpos, dtype=np.float32).copy())

        for _ in range(max_steps):
            if render:
                env.render()

            action = policy(obs)
            n_obs, reward, terminated, truncated, info = unpack_step(env.step(action), env)

            _term_arr = np.asarray(terminated)
            _trunc_arr = np.asarray(truncated)
            _is_batched = _term_arr.ndim > 0 or (
                isinstance(info, dict) and '_done_per_lane' in info
            )
            if _is_batched:
                _done_arr = np.asarray(
                    info.get('_done_per_lane', _term_arr | _trunc_arr), dtype=bool
                )
                done = bool(_done_arr.all())
                _rew_arr = np.asarray(reward, dtype=np.float32)
                score += float(_rew_arr.sum())
                steps += int(_term_arr.size)
                if memory is not None:
                    # Termination zeros the TD mask even when truncation is also reported.
                    _done_bool_batch = _term_arr.astype(np.float32)
                    memory.add_batch(
                        np.asarray(obs, dtype=np.float32),
                        np.asarray(n_obs, dtype=np.float32),
                        np.asarray(action, dtype=np.float32),
                        _rew_arr,
                        _done_bool_batch,
                    )
                done_bool = _done_arr
            else:
                done = bool(terminated) or bool(truncated)
                _rew_scalar = float(reward)
                score += _rew_scalar
                steps += 1
                _done_bool = replay_terminal(terminated, truncated)
                if memory is not None:
                    memory.add((obs, n_obs, action, _rew_scalar, _done_bool))
                done_bool = _done_bool


            if return_rollout:
                _rew_for_rollout = float(np.asarray(reward).mean()) if np.asarray(reward).ndim > 0 else float(reward)
                _done_for_rollout = float(np.asarray(done_bool).mean()) if np.asarray(done_bool).ndim > 0 else float(done_bool)
                ep_rollout.append((
                    np.asarray(obs, dtype=np.float32),
                    np.asarray(action, dtype=np.float32),
                    np.asarray(n_obs, dtype=np.float32),
                    _rew_for_rollout,
                    _done_for_rollout,
                ))

            if return_bd_traj:
                _unwrapped = env.unwrapped
                ep_x_traj.append(float(_unwrapped.data.qpos[0]))
                ep_xpos_traj.append(np.asarray(_unwrapped.data.xpos, dtype=np.float32).copy())

            obs = n_obs
            if done:
                break

        scores.append(float(score))
        if return_rollout:
            rollout_datas.append(ep_rollout)
        if return_bd_traj:
            x_traj_list.append(ep_x_traj)
            xpos_traj_list.append(ep_xpos_traj)


    if step_counter is not None:
        # Report episodes separately; env.step already counts their interactions.
        step_counter.add_eval_call(1)
        step_counter.add_eval_episodes(len(scores))

    if return_rollout or return_bd_traj:
        rewards_arr = np.asarray(scores, dtype=np.float32)
        return rewards_arr, rollout_datas, None, x_traj_list, xpos_traj_list

    mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
    return mean_score, int(steps)
