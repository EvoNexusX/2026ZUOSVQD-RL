import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CUDA_GRAPH_ENABLED = os.environ.get("FT_CUDA_GRAPH", "1") == "1" and torch.cuda.is_available()


class RLNN(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super().__init__(state_dim, action_dim, 1)
        hidden_size = int(getattr(args, 'hidden_layer_size', 256))
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)
        if args.layer_norm:
            self.n1 = nn.LayerNorm(hidden_size)
            self.n2 = nn.LayerNorm(hidden_size)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)
        if args.layer_norm:
            self.n4 = nn.LayerNorm(hidden_size)
            self.n5 = nn.LayerNorm(hidden_size)
        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

    def _forward_impl(self, x, u):
        if not self.layer_norm:
            x1 = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x1 = F.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)
        else:
            x1 = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = F.leaky_relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)
        if not self.layer_norm:
            x2 = F.leaky_relu(self.l4(torch.cat([x, u], 1)))
            x2 = F.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)
        else:
            x2 = F.leaky_relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = F.leaky_relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)
        return x1, x2

    def forward(self, x, u):
        return self._forward_impl(x, u)

    def update(self, memory, batch_size, actor_t, critic_t):
        states, n_states, actions, rewards, dones = memory.sample(batch_size)
        noise = np.clip(
            np.random.normal(0, self.policy_noise, size=(batch_size, self.action_dim)),
            -self.noise_clip, self.noise_clip
        )
        noise_t = torch.tensor(noise, dtype=torch.float32, device=device)
        n_actions = actor_t(n_states) + noise_t
        n_actions = n_actions.clamp(-self.max_action, self.max_action)
        with torch.no_grad():
            q1_t, q2_t = critic_t(n_states, n_actions)
            target_Q = torch.min(q1_t, q2_t)
            target_Q = rewards + (1 - dones) * self.discount * target_Q
        current_Q1, current_Q2 = self(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class Actor(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args, hidden_layers=None, masks=None):
        super().__init__(state_dim, action_dim, max_action)
        self.args = args
        self.hidden_layers = list(hidden_layers or [])
        self.layer_norm = hasattr(args, 'layer_norm') and args.layer_norm
        self.masks = masks

        self.activation_function = nn.LeakyReLU(0.01)

        layers = []
        prev_dim = int(state_dim)
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, int(action_dim)))
        self.layers = nn.ModuleList(layers)

        self.to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=getattr(args, 'actor_lr', 1e-3))
        self.tau = getattr(args, 'tau', 0.005)
        self.discount = getattr(args, 'discount', 0.99)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)

    def _forward_impl(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_function(x)
        x = self.layers[-1](x)
        x = self.max_action * torch.tanh(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def update(self, memory, batch_size, critic, actor_t, lambda_explore: float = 0.0):
        states, _, _, _, _ = memory.sample(batch_size)
        q_pair = critic(states, self(states))
        if isinstance(q_pair, tuple):
            q1 = q_pair[0]
            q2 = q_pair[1] if len(q_pair) > 1 else q1
            q_min = torch.min(q1, q2)
        else:
            q_min = q1 = q2 = q_pair
        if float(lambda_explore) > 0.0:
            q_std = (q1 - q2).abs().mean() * 0.5
            actor_loss = -(q_min + float(lambda_explore) * q_std).mean()
        else:
            actor_loss = -q_min.mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class _ActorCUDAGraph:
    """Fixed-shape (1, state_dim) actor forward CUDA Graph cache."""

    def __init__(self, actor: "Actor", state_dim: int):
        self.actor = actor
        self.state_dim = int(state_dim)
        self.device = next(actor.parameters()).device
        self._graph = None
        self._static_in = None
        self._static_out = None
        self._built = False

    def _build(self):
        actor = self.actor
        actor.eval()
        self._static_in = torch.zeros(1, self.state_dim, device=self.device, dtype=torch.float32)
        forward_fn = getattr(actor, '_forward_impl', actor.forward)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = forward_fn(self._static_in)
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_out = forward_fn(self._static_in)
        self._built = True

    def __call__(self, state_np_or_tensor):
        if not self._built:
            self._build()
        if isinstance(state_np_or_tensor, np.ndarray):
            self._static_in[0].copy_(torch.as_tensor(state_np_or_tensor, dtype=torch.float32, device=self.device))
        else:
            self._static_in[0].copy_(state_np_or_tensor.to(self.device))
        self._graph.replay()
        return self._static_out[0].detach().cpu().numpy()


def get_actor_cuda_graph(actor: "Actor"):
    if not _CUDA_GRAPH_ENABLED:
        return None
    if getattr(actor, '_cuda_graph_cache', None) is None:
        actor._cuda_graph_cache = _ActorCUDAGraph(actor, actor.state_dim)
    return actor._cuda_graph_cache


def safe_deepcopy(model: nn.Module, device_target=None) -> nn.Module:
    """deepcopy that skips _cuda_graph_cache (CUDAGraph cannot pickle)."""
    saved = getattr(model, '_cuda_graph_cache', None)
    if saved is not None:
        model._cuda_graph_cache = None
    try:
        from copy import deepcopy as _dc
        out = _dc(model)
    finally:
        if saved is not None:
            model._cuda_graph_cache = saved
    out._cuda_graph_cache = None
    if device_target is not None:
        out = out.to(device_target)
    return out
