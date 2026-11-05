import os
import random
import numpy as np
import torch


def set_global_seeds(seed: int, deterministic: bool = True):
    seed = int(seed)
    deterministic = bool(deterministic)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    except Exception:
        pass


def seed_env(env, seed: int, episode_offset: int = 0):
    base = int(seed) & 0xFFFFFFFF
    offs = int(episode_offset) & 0xFFFFFFFF
    ep_seed = (base ^ ((offs << 1) | (offs >> 31))) & 0xFFFFFFFF
    return env.reset(seed=ep_seed)