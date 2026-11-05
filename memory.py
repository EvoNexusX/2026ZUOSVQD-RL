import numpy as np
import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class Memory:
    def __init__(self, memory_size, state_dim, action_dim):
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        zeros = torch.zeros
        if USE_CUDA:
            device_ = torch.device("cuda")
        else:
            device_ = torch.device("cpu")
        self.states = zeros(self.memory_size, self.state_dim, device=device_)
        self.actions = zeros(self.memory_size, self.action_dim, device=device_)
        self.n_states = zeros(self.memory_size, self.state_dim, device=device_)
        self.rewards = zeros(self.memory_size, 1, device=device_)
        self.dones = zeros(self.memory_size, 1, device=device_)

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def reset(self):
        self.pos = 0
        self.full = False

    def get_pos(self):
        return self.pos

    def add(self, datum):
        state, n_state, action, reward, done = datum
        device_ = self.states.device
        self.states[self.pos] = torch.as_tensor(state, dtype=torch.float32, device=device_)
        self.n_states[self.pos] = torch.as_tensor(n_state, dtype=torch.float32, device=device_)
        self.actions[self.pos] = torch.as_tensor(action, dtype=torch.float32, device=device_)
        self.rewards[self.pos] = torch.as_tensor([reward], dtype=torch.float32, device=device_)
        self.dones[self.pos] = torch.as_tensor([done], dtype=torch.float32, device=device_)
        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def add_batch(self, states, n_states, actions, rewards, dones):
        n = int(np.asarray(states).shape[0])
        if n == 0:
            return
        device_ = self.states.device
        states_t = torch.as_tensor(np.array(states, dtype=np.float32, copy=True), dtype=torch.float32, device=device_)
        n_states_t = torch.as_tensor(np.array(n_states, dtype=np.float32, copy=True), dtype=torch.float32, device=device_)
        actions_t = torch.as_tensor(np.array(actions, dtype=np.float32, copy=True), dtype=torch.float32, device=device_)
        rewards_t = torch.as_tensor(np.array(rewards, dtype=np.float32, copy=True), dtype=torch.float32, device=device_).reshape(-1, 1)
        dones_t = torch.as_tensor(np.array(dones, dtype=np.float32, copy=True), dtype=torch.float32, device=device_).reshape(-1, 1)

        end_pos = self.pos + n
        if end_pos <= self.memory_size:
            self.states[self.pos:end_pos] = states_t
            self.n_states[self.pos:end_pos] = n_states_t
            self.actions[self.pos:end_pos] = actions_t
            self.rewards[self.pos:end_pos] = rewards_t
            self.dones[self.pos:end_pos] = dones_t
            self.pos = end_pos
            if self.pos == self.memory_size:
                self.full = True
                self.pos = 0
        else:
            first = self.memory_size - self.pos
            self.states[self.pos:] = states_t[:first]
            self.n_states[self.pos:] = n_states_t[:first]
            self.actions[self.pos:] = actions_t[:first]
            self.rewards[self.pos:] = rewards_t[:first]
            self.dones[self.pos:] = dones_t[:first]
            rest = n - first
            self.states[:rest] = states_t[first:]
            self.n_states[:rest] = n_states_t[first:]
            self.actions[:rest] = actions_t[first:]
            self.rewards[:rest] = rewards_t[first:]
            self.dones[:rest] = dones_t[first:]
            self.pos = rest
            self.full = True

    def sample(self, batch_size):
        upper_bound = self.memory_size if self.full else self.pos
        device_ = self.states.device
        batch_inds = torch.tensor(
            np.random.randint(0, upper_bound, size=batch_size),
            dtype=torch.long, device=device_)
        return (self.states[batch_inds],
                self.n_states[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.dones[batch_inds])

    def repeat(self, start_pos, end_pos):
        if start_pos <= end_pos:
            for i in range(start_pos, end_pos):
                self._copy_slot(i)
        else:
            for i in range(start_pos, self.memory_size):
                self._copy_slot(i)
            for i in range(end_pos):
                self._copy_slot(i)

    def _copy_slot(self, i):
        self.states[self.pos] = self.states[i].clone()
        self.n_states[self.pos] = self.n_states[i].clone()
        self.actions[self.pos] = self.actions[i].clone()
        self.rewards[self.pos] = self.rewards[i].clone()
        self.dones[self.pos] = self.dones[i].clone()
        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0


class MaskedReplay:
    """Replay namespaces keyed by mask_id; set_mask selects the active namespace."""

    def __init__(self, base_buffer_cls, memory_size, state_dim, action_dim,
                 base_kwargs=None, default_mask_id="default", max_buffers=5):
        assert base_buffer_cls is not None, "base_buffer_cls is required"
        self._base_cls = base_buffer_cls
        self._base_kwargs = dict(base_kwargs or {})
        self._memory_size = int(memory_size)
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)
        self._buffers = {}
        self._current_id = None
        self._max_buffers = max_buffers
        self._protected_ids = set()
        self.set_mask(default_mask_id)

    def _ensure_buffer(self, mask_id):
        if mask_id not in self._buffers:
            max_buffers = int(getattr(self, '_max_buffers', 5))
            if len(self._buffers) >= max_buffers:
                to_remove = None
                for k in self._buffers.keys():
                    if k != self._current_id and k not in self._protected_ids:
                        to_remove = k
                        break
                if to_remove is not None:
                    buf = self._buffers.pop(to_remove, None)
                    if buf is not None:
                        for attr in ('states', 'actions', 'n_states', 'rewards', 'dones'):
                            if hasattr(buf, attr):
                                delattr(buf, attr)
                        del buf
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            self._buffers[mask_id] = self._base_cls(
                self._memory_size, self._state_dim, self._action_dim, **self._base_kwargs
            )
        return self._buffers[mask_id]

    def set_mask(self, mask_id):
        if not isinstance(mask_id, str):
            mask_id = str(mask_id)
        self._current_id = mask_id
        self._ensure_buffer(mask_id)


    def get_buffer(self, mask_id):
        return self._buffers.get(mask_id, None)

    def protect_ids(self, ids):
        if ids is None:
            return
        self._protected_ids.update(str(x) for x in ids if x is not None)

    def release_ids(self, ids):
        if ids is None:
            return
        for x in ids:
            if x is not None:
                self._protected_ids.discard(str(x))

    def add(self, datum):
        return self._ensure_buffer(self._current_id).add(datum)

    def sample(self, batch_size):
        return self._ensure_buffer(self._current_id).sample(batch_size)

    def size(self):
        return self._ensure_buffer(self._current_id).size()

    def get_pos(self):
        buf = self._ensure_buffer(self._current_id)
        if hasattr(buf, 'get_pos'):
            return buf.get_pos()
        return buf.size()

    def reset(self):
        buf = self._ensure_buffer(self._current_id)
        if hasattr(buf, 'reset'):
            buf.reset()


    def clear_except(self, keep_ids):
        if keep_ids is None:
            keep_ids = set()
        keep = set(map(str, keep_ids)) | set(getattr(self, '_protected_ids', set()))
        to_del = [k for k in self._buffers.keys() if k not in keep]
        for k in to_del:
            try:
                buf = self._buffers.pop(k)
                del buf
            except Exception:
                pass
        if self._current_id not in self._buffers:
            self._current_id = None
            next_id = next(iter(keep)) if len(keep) > 0 else "default"
            self.set_mask(str(next_id))
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


class MixedMemory:
    """Sample sub-buffers by ratio and concatenate using the Memory sample interface."""

    def __init__(self, buffers, ratio=None):
        assert len(buffers) > 0, "MixedMemory needs at least one buffer"
        self.buffers = [b for b in buffers if b is not None]
        if not self.buffers:
            raise ValueError("MixedMemory: all buffers are None")
        self.n = len(self.buffers)
        if ratio is None:
            self.ratio = [1.0 / self.n] * self.n
        else:
            assert len(ratio) == len(buffers), (
                f"ratio length {len(ratio)} does not match buffers length {len(buffers)}"
            )
            total = sum(ratio)
            self.ratio = [r / total for r in ratio]
            self.ratio = [r for r, b in zip(self.ratio, buffers) if b is not None]

    def size(self):
        return sum(b.size() for b in self.buffers if hasattr(b, 'size'))

    def add(self, datum):
        if self.buffers:
            self.buffers[0].add(datum)

    def sample(self, batch_size):
        available = []
        for buf, ratio in zip(self.buffers, self.ratio):
            if buf is None or not hasattr(buf, 'sample'):
                continue
            actual_size = buf.size() if hasattr(buf, 'size') else 0
            if actual_size <= 0:
                continue
            available.append((buf, float(ratio)))

        if not available:
            raise RuntimeError("MixedMemory: all buffers are empty")

        if len(available) == 1:
            return available[0][0].sample(batch_size)

        total_ratio = sum(r for _, r in available)
        if total_ratio <= 0:
            ratios = [1.0 / len(available)] * len(available)
        else:
            ratios = [r / total_ratio for _, r in available]

        sizes = [max(1, int(batch_size * r)) for r in ratios]
        diff = int(batch_size) - sum(sizes)
        sizes[0] += diff

        batches = []
        for (buf, _), sz in zip(available, sizes):
            if sz <= 0:
                continue
            try:
                batch = buf.sample(sz)
                if batch is None or len(batch) < 5:
                    continue
                batches.append(batch)
            except Exception:
                continue

        if not batches:
            raise RuntimeError("MixedMemory: all buffers failed to sample")
        if len(batches) == 1:
            batch = batches[0]
            if batch[0].shape[0] == batch_size:
                return batch
            return available[0][0].sample(batch_size)

        states = torch.cat([b[0] for b in batches], dim=0)
        n_states = torch.cat([b[1] for b in batches], dim=0)
        actions = torch.cat([b[2] for b in batches], dim=0)
        rewards = torch.cat([b[3] for b in batches], dim=0)
        dones = torch.cat([b[4] for b in batches], dim=0)
        return (states, n_states, actions, rewards, dones)

def snapshot_replay_buffer(buf):
    """Copy replay and ring cursors to CPU before later namespace eviction can remove them."""
    if buf is None:
        return None
    if not all(hasattr(buf, name) for name in
               ('states', 'actions', 'n_states', 'rewards', 'dones')):
        return None
    try:
        return {
            'states': buf.states.detach().cpu().clone(),
            'actions': buf.actions.detach().cpu().clone(),
            'n_states': buf.n_states.detach().cpu().clone(),
            'rewards': buf.rewards.detach().cpu().clone(),
            'dones': buf.dones.detach().cpu().clone(),
            'pos': int(getattr(buf, 'pos', 0)),
            'left_ptr': int(getattr(buf, 'left_ptr', 0)),
            'full': bool(getattr(buf, 'full', False)),
            'memory_size': int(getattr(buf, 'memory_size', buf.states.shape[0])),
        }
    except Exception:
        return None
