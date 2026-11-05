import logging
from typing import Dict, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import CriticTD3

logger = logging.getLogger(__name__)


class PerIndividualCriticBank:
    """One dedicated CriticTD3 per archive individual, with an LRU eviction cap."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 config, max_individuals: int = 256,
                 lr: float = 1e-3, device: str = 'cuda',
                 gamma: float = 0.99,
                 policy_noise: float = 0.2, noise_clip: float = 0.5):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.config = config
        self.max_individuals = int(max_individuals)
        self.lr = float(lr)
        self.device_str = device
        self.gamma = float(gamma)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)

        self.individuals: Dict[int, Dict[str, Any]] = {}
        self._tick: Dict[int, int] = {}
        self._global_tick: int = 0

    def _evict_if_needed(self):
        if len(self.individuals) <= self.max_individuals:
            return
        n_evict = len(self.individuals) - self.max_individuals
        order = sorted(self._tick.items(), key=lambda kv: kv[1])
        for idx, _ in order[:n_evict]:
            self.remove(idx)

    def has(self, idx: int) -> bool:
        return int(idx) in self.individuals

    def register(self, idx: int, base_critic: Optional[CriticTD3] = None) -> None:
        idx = int(idx)
        self._global_tick += 1
        self._tick[idx] = self._global_tick
        if idx in self.individuals:
            return
        critic = CriticTD3(self.state_dim, self.action_dim, self.max_action, self.config)
        if base_critic is not None:
            try:
                critic.load_state_dict(base_critic.state_dict())
            except Exception:
                pass
        critic.to(self.device_str)
        opt = torch.optim.Adam(critic.parameters(), lr=self.lr)
        self.individuals[idx] = {"critic": critic, "opt": opt}
        self._evict_if_needed()

    def remove(self, idx: int) -> None:
        idx = int(idx)
        self.individuals.pop(idx, None)
        self._tick.pop(idx, None)

    def get_state(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        entry = self.individuals.get(int(idx))
        if entry is None:
            return None
        return {k: v.detach().cpu().clone() for k, v in entry["critic"].state_dict().items()}

    def get_critic(self, idx: int, device: Optional[str] = None) -> Optional[CriticTD3]:
        entry = self.individuals.get(int(idx))
        if entry is None:
            return None
        c = entry["critic"]
        if device is not None:
            c = c.to(device)
        return c.eval()


class NoveltyCriticBank:
    """Independent state/descriptor heads; absent archive targets default to one."""

    def __init__(self, state_dim: int, bd_dim: int = 2, n_bins: int = 50,
                 n_hidden: int = 128, K: int = 4, lr: float = 3e-4,
                 device: str = 'cuda'):
        self.state_dim = int(state_dim)
        self.bd_dim = int(bd_dim)
        self.n_bins = int(n_bins)
        self.K = int(K)
        self.device_str = device
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + bd_dim, n_hidden), nn.ReLU(),
                nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                nn.Linear(n_hidden, 1),
            ).to(device)
            for _ in range(K)
        ])
        self.opts = [torch.optim.Adam(net.parameters(), lr=float(lr))
                     for net in self.nets]

    @staticmethod
    def _target_distance(bd_batch: np.ndarray,
                          archive_bds: Optional[np.ndarray]) -> np.ndarray:
        if archive_bds is None or len(archive_bds) == 0:
            return np.ones(bd_batch.shape[0], dtype=np.float32)
        diffs = bd_batch[:, None, :] - archive_bds[None, :, :]
        d = np.linalg.norm(diffs, axis=-1)
        return d.min(axis=1).astype(np.float32)

    def _archive_bds(self, external_archive) -> Optional[np.ndarray]:
        if not external_archive:
            return None
        from adaptive_archive_admission import canonicalize_gait_bd
        rows = []
        for it in external_archive:
            bd_raw = it[2] if len(it) > 2 else None
            bd_arr, _ = canonicalize_gait_bd(bd_raw)
            if bd_arr is not None and np.all(np.isfinite(bd_arr)):
                rows.append(np.asarray(bd_arr, dtype=np.float64))
        if not rows:
            return None
        return np.stack(rows, axis=0)

    def update(self, replay_buffer, external_archive, actor, batch_size: int = 256) -> None:
        """Update all heads with Adam on the same sampled state/descriptor batch."""
        if replay_buffer is None or not hasattr(replay_buffer, 'size') or replay_buffer.size() < batch_size:
            return
        try:
            s, _, _, _, _ = replay_buffer.sample(batch_size)
        except Exception:
            return
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device_str)
        bd_t = torch.rand(batch_size, self.bd_dim, device=self.device_str)
        target = torch.as_tensor(
            self._target_distance(bd_t.detach().cpu().numpy(),
                                  self._archive_bds(external_archive)),
            dtype=torch.float32, device=self.device_str,
        ).view(-1, 1)
        inp = torch.cat([s_t, bd_t], dim=-1)
        for net, opt in zip(self.nets, self.opts):
            net.train()
            pred = net(inp)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def novelty_uncertainty(self, states: np.ndarray,
                            candidate_bds: np.ndarray) -> Optional[np.ndarray]:
        """Return the sample standard deviation across predictor heads."""
        if states is None or candidate_bds is None:
            return None
        s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device_str)
        bd_t = torch.as_tensor(candidate_bds, dtype=torch.float32, device=self.device_str)
        inp = torch.cat([s_t, bd_t], dim=-1)
        preds = torch.stack([net(inp).squeeze(-1) for net in self.nets], dim=0)
        return preds.std(dim=0).detach().cpu().numpy()


class SharedCriticBank:
    """Shared, diverse and individual critics providing branch learning and disagreement."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 config, device: str = 'cuda',
                 pib_max_individuals: int = 256):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.config = config
        self.device_str = device

        self.shared_critic = CriticTD3(state_dim, action_dim, max_action, config).to(device)
        self.shared_opt = torch.optim.Adam(
            self.shared_critic.parameters(),
            lr=float(getattr(config, 'critic_lr', 1e-3)),
        )
        self._profiles: Dict[int, np.ndarray] = {}
        self._private: Dict[int, Dict[str, torch.Tensor]] = {}

        # Copy the shared critic, then randomize hidden weights with per-critic seeds.
        n_ensemble = int(getattr(config, 'mask_critic_ensemble_size', 2))
        self.diverse_critics: List[CriticTD3] = []
        self.diverse_opts: List[torch.optim.Adam] = []
        if n_ensemble > 0:
            base_seed = int(getattr(config, 'seed', 42))
            _saved_seed = torch.initial_seed()
            for k in range(n_ensemble):
                c = CriticTD3(state_dim, action_dim, max_action, config).to(device)
                c.load_state_dict(self.shared_critic.state_dict())
                torch.manual_seed(base_seed + 1000 + k)
                with torch.no_grad():
                    c.l2.weight.data.normal_(0, 0.1)
                    c.l5.weight.data.normal_(0, 0.1)
                    c.l2.bias.data.zero_()
                    c.l5.bias.data.zero_()
                opt = torch.optim.Adam(c.parameters(),
                                       lr=float(getattr(config, 'critic_lr', 1e-3)))
                self.diverse_critics.append(c)
                self.diverse_opts.append(opt)
            torch.manual_seed(int(_saved_seed))

        self.pib = PerIndividualCriticBank(
            state_dim=state_dim, action_dim=action_dim, max_action=max_action,
            config=config, max_individuals=pib_max_individuals,
            lr=float(getattr(config, 'critic_lr', 1e-3)), device=device,
        )

        self.use_novelty_critic = bool(getattr(config, 'use_novelty_critic', False))
        self.novelty_bank: Optional[NoveltyCriticBank] = None
        if self.use_novelty_critic:
            try:
                self.novelty_bank = NoveltyCriticBank(
                    state_dim=state_dim,
                    bd_dim=int(getattr(config, 'novelty_critic_bd_dim', 2)),
                    n_bins=int(getattr(config, 'novelty_critic_n_bins',
                                       int(getattr(config, 'qd_n_bins_per_dim', 50)))),
                    K=int(getattr(config, 'novelty_critic_n_heads', 4)),
                    lr=float(getattr(config, 'novelty_critic_lr', 3e-4)),
                    device=device,
                )
            except Exception as e:
                logger.debug(f"NoveltyCriticBank init failed: {e}")
                self.novelty_bank = None

    def _td3_gradient_step(self, memory, actor, critic, opt, batch_size: int) -> Optional[float]:
        """One TD3 critic gradient step on a batch drawn from `memory`.

        Returns the loss value, or None when the batch would be empty.
        """
        bs = int(batch_size)
        if hasattr(memory, 'size'):
            bs = min(bs, int(memory.size()))
        if bs <= 0:
            return None
        s, ns, a, r, done = memory.sample(bs)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device_str)
        ns_t = torch.as_tensor(ns, dtype=torch.float32, device=self.device_str)
        a_t = torch.as_tensor(a, dtype=torch.float32, device=self.device_str)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device_str).view(-1, 1)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device_str).view(-1, 1)
        with torch.no_grad():
            na = actor(ns_t)
            noise = (torch.randn_like(na) * 0.2).clamp(-0.5, 0.5)
            na = (na + noise).clamp(-self.max_action, self.max_action)
            tq1, tq2 = critic(ns_t, na)
            target = r_t + 0.99 * (1.0 - done_t) * torch.min(tq1, tq2)
        q1, q2 = critic(s_t, a_t)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    def update_per_generation(self, memory, actor, batch_size: int = 256) -> List[float]:
        """Bootstrap the shared critic from `memory` (and diverse critics if available)."""
        try:
            loss = self._td3_gradient_step(
                memory, actor, self.shared_critic, self.shared_opt, batch_size)
            if loss is None:
                return []
        except Exception as e:
            logger.debug(f"Shared critic update failed: {e}")
            return []

        if self.diverse_critics and hasattr(memory, 'size') and memory.size() > batch_size:
            try:
                config = self.config
                ensemble_subset = float(getattr(config, 'mask_critic_ensemble_subset_ratio', 0.7))
                ensemble_steps = int(getattr(config, 'mask_critic_ensemble_steps', 30))
                for c_idx, (c, opt) in enumerate(zip(self.diverse_critics, self.diverse_opts)):
                    c.train()
                    for _ in range(ensemble_steps):
                        # Return the last ensemble loss if updated, otherwise the shared-critic loss.
                        loss = self._td3_gradient_step(
                            memory, actor, c, opt, int(batch_size * ensemble_subset))
                        if loss is None:
                            break
            except Exception as e:
                logger.debug(f"Diverse critic training failed: {e}")

        if self.novelty_bank is not None:
            try:
                archive = getattr(self, 'external_archive', None)
                self.novelty_bank.update(memory, archive, actor, batch_size=batch_size)
            except Exception as e:
                logger.debug(f"NoveltyCriticBank update failed: {e}")

        return [loss]

    def register_individual_critic(self, idx: int,
                                   base_critic: Optional[CriticTD3] = None) -> None:
        self.pib.register(int(idx), base_critic=base_critic)

    def get_critic(self, idx: int, device: Optional[str] = None) -> Optional[CriticTD3]:
        """Return per-individual critic (lazy-register from shared_critic)."""
        idx = int(idx)
        if not self.pib.has(idx):
            self.pib.register(idx, base_critic=self.shared_critic)
        return self.pib.get_critic(idx, device or self.device_str)

    def get_state(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        idx = int(idx)
        if not self.pib.has(idx):
            return None
        return self.pib.get_state(idx)

    def compute_uncertainty_from_buffer(self, buffer, actor, n_samples: int = 64) -> float:
        """Q-disagreement across shared + diverse critics on the candidate's
        own replay states. Sigmoid-scaled to [0, 1].
        """
        try:
            if buffer is None or not hasattr(buffer, 'size') or buffer.size() < n_samples:
                return 0.0
            actor_dev = actor.to(self.device_str).eval() if hasattr(actor, 'to') else actor
            s, _, _, _, _ = buffer.sample(int(n_samples))
            s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device_str)
            with torch.no_grad():
                acts = actor_dev(s_t)
                acts = torch.clamp(acts, -self.max_action, self.max_action)
                qs = []
                q1, q2 = self.shared_critic(s_t, acts)
                qs.append(((q1 + q2) / 2).squeeze(-1))
                for c in self.diverse_critics:
                    q1, q2 = c(s_t, acts)
                    qs.append(((q1 + q2) / 2).squeeze(-1))
                if len(qs) < 2:
                    q_std = (qs[0] - qs[0]).std().item()
                else:
                    qs_stack = torch.stack(qs, dim=0)
                    q_std = qs_stack.std(dim=0).mean().item()
            return float(1.0 / (1.0 + np.exp(-q_std * 5.0)))
        except Exception as exc:
            logger.debug(f"compute_uncertainty_from_buffer failed: {exc}")
            return 0.5

    def clone_shared_critic(self, lr: Optional[float] = None,
                            device: Optional[str] = None) -> CriticTD3:
        """deepcopy shared critic (used by mask-phase candidate evaluators)."""
        from networks import safe_deepcopy
        dev = device or self.device_str
        critic = safe_deepcopy(self.shared_critic, device_target=dev)
        if not hasattr(critic, 'optimizer') or critic.optimizer is None:
            critic.optimizer = torch.optim.Adam(
                critic.parameters(),
                lr=float(lr if lr is not None else getattr(self.config, 'critic_lr', 1e-3)),
            )
        return critic

    def compute_novelty_signal_from_buffer(self, buffer, candidate_bd: np.ndarray,
                                            actor=None, n_samples: int = 64) -> float:
        """BD-novelty uncertainty in [0, 1]; 0 if novelty bank is disabled."""
        if self.novelty_bank is None or buffer is None:
            return 0.0
        if not hasattr(buffer, 'size') or buffer.size() < n_samples:
            return 0.0
        try:
            s, _, _, _, _ = buffer.sample(int(n_samples))
            bd = np.asarray(candidate_bd, dtype=np.float64).reshape(1, -1)
            bd_tiled = np.tile(bd, (s.shape[0], 1))
            unc = self.novelty_bank.novelty_uncertainty(s, bd_tiled)
            if unc is None or unc.size == 0:
                return 0.0
            return float(1.0 / (1.0 + np.exp(-float(unc.mean()) * 5.0)))
        except Exception as exc:
            logger.debug(f"compute_novelty_signal_from_buffer failed: {exc}")
            return 0.5

    def has(self, idx: int) -> bool:
        """Whether a per-individual critic is registered for `idx`."""
        return self.pib.has(int(idx))

    def remove(self, idx: int) -> None:
        self.pib.remove(int(idx))

    def sync_with_archive(self, archive: list) -> None:
        valid_idx = set(range(len(archive)))
        for idx in list(self.pib.individuals.keys()):
            if idx not in valid_idx:
                self.pib.remove(idx)

    def archive_coverage_novelty(self) -> float:
        """Mean pairwise L2 distance between per-individual critic profiles."""
        if not self._profiles or len(self._profiles) < 2:
            return 0.0
        arr = np.stack([np.asarray(p, dtype=np.float32).flatten()
                        for p in self._profiles.values()], axis=0)
        diffs = arr[:, None, :] - arr[None, :, :]
        dist = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices_from(dist, k=1)
        return float(dist[iu].mean()) if iu[0].size > 0 else 0.0

    @property
    def profiles(self) -> Dict[int, np.ndarray]:
        return getattr(self, '_profiles', {})

    @profiles.setter
    def profiles(self, value):
        self._profiles = value

    @property
    def private(self) -> Dict[int, Dict[str, torch.Tensor]]:
        return getattr(self, '_private', {})

    @private.setter
    def private(self, value):
        self._private = value
