"""Critic-field direction injection (CFDI) for actor update diversification."""
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x):
    if torch.is_tensor(x):
        return x.to(device).float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def _unflatten_to_actor(actor, flat):
    """Split a flat tensor into per-parameter chunks matching `actor.parameters()`."""
    if not torch.is_tensor(flat):
        flat = torch.as_tensor(flat, dtype=torch.float32, device=device)
    flat = flat.to(device).float()
    parts = []
    cursor = 0
    for p in actor.parameters():
        n = int(np.prod(p.shape)) if p.dim() > 0 else 1
        chunk = flat[cursor: cursor + n].reshape(p.shape)
        parts.append(chunk)
        cursor += n
    return parts


def _build_param_mask(actor, masks=None):
    """Convert per-layer masks (or actor.masks) into a flat 0/1 vector aligned
    with `actor.parameters()`. Biases are always trainable; dense actors get all-ones.
    """
    masks = masks if masks is not None else getattr(actor, 'masks', None)
    flat_parts = []
    mask_iter = iter(masks) if masks is not None else None
    for layer in actor.modules():
        if isinstance(layer, nn.Linear):
            w_shape = layer.weight.shape
            m = None
            if mask_iter is not None:
                try:
                    m = next(mask_iter)
                except StopIteration:
                    m = None
            if m is None:
                w_mask = torch.ones(w_shape, dtype=torch.float32, device=device)
            else:
                if not torch.is_tensor(m):
                    m = torch.as_tensor(np.asarray(m), dtype=torch.float32)
                w_mask = m.to(device).float()
                if w_mask.shape != w_shape:
                    if int(np.prod(w_mask.shape)) == int(np.prod(w_shape)):
                        w_mask = w_mask.reshape(w_shape)
                    else:
                        w_mask = torch.ones(w_shape, dtype=torch.float32, device=device)
            flat_parts.append(w_mask.reshape(-1))
            if layer.bias is not None:
                flat_parts.append(torch.ones(layer.bias.shape, dtype=torch.float32, device=device).reshape(-1))
    if not flat_parts:
        return torch.ones(int(sum(int(np.prod(p.shape))
                                   for p in actor.parameters())),
                          dtype=torch.float32, device=device)
    return torch.cat(flat_parts)


def compute_critic_actor_direction(actor, critic, replay_batch, mask=None,
                                    grad_clip_norm=1.0, eps=1e-8):
    """Δθ_i^Q = ∇Q: gradient of critic Q w.r.t. actor params, aligned to the
    structural mask, clipped, and unit-normalised.
    """
    if actor is None or critic is None or replay_batch is None:
        return None
    states = replay_batch[0]
    if not torch.is_tensor(states):
        states = _to_tensor(states)
    else:
        states = states.to(device).float()
    if states.dim() == 1:
        states = states.unsqueeze(0)

    actor_was_training = actor.training
    critic_was_training = critic.training
    actor.eval()
    critic.eval()
    try:
        actions = actor(states)
        out = critic(states, actions)
        q1 = out[0] if isinstance(out, tuple) else out
        actor_loss = -q1.mean()
        grads = torch.autograd.grad(
            actor_loss, list(actor.parameters()),
            retain_graph=False, create_graph=False, allow_unused=True,
        )
        delta_parts = []
        for g, p in zip(grads, actor.parameters()):
            if g is None:
                delta_parts.append(torch.zeros_like(p).reshape(-1))
            else:
                delta_parts.append((-g).reshape(-1))
        delta = torch.cat(delta_parts)

        param_mask = _build_param_mask(actor, mask)
        if param_mask.numel() != delta.numel():
            param_mask = torch.ones_like(delta)
        delta = delta * param_mask

        norm = float(delta.norm().item())
        if norm > float(grad_clip_norm) and norm > 0:
            delta = delta * (float(grad_clip_norm) / (norm + float(eps)))

        out_norm = float(delta.norm().item()) + float(eps)
        delta_normed = (delta / out_norm).detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(delta_normed)):
            delta_normed = np.nan_to_num(delta_normed, nan=0.0, posinf=0.0, neginf=0.0)
        return delta_normed
    except Exception as e:
        logger.debug(f"compute_critic_actor_direction failed: {e}")
        return None
    finally:
        if actor_was_training:
            actor.train()
        if critic_was_training:
            critic.train()


def apply_flat_direction_to_actor(actor, flat_direction, mask=None,
                                  alpha=1.0, noise_std=0.0, reapply_mask=True):
    """Apply α · Δθ (+ optional Gaussian noise) to actor params; re-mask afterwards."""
    if actor is None or flat_direction is None:
        return
    if not torch.is_tensor(flat_direction):
        flat_t = torch.as_tensor(flat_direction, dtype=torch.float32, device=device)
    else:
        flat_t = flat_direction.to(device).float()
    parts = _unflatten_to_actor(actor, flat_t)
    actor.eval()
    param_mask_flat = _build_param_mask(actor, mask)
    if param_mask_flat.numel() != flat_t.numel():
        param_mask_flat = torch.ones_like(flat_t)
    cursor = 0
    with torch.no_grad():
        for p, d in zip(actor.parameters(), parts):
            n = d.numel()
            m_chunk = param_mask_flat[cursor: cursor + n].reshape(p.shape)
            cursor += n
            step = alpha * d
            if noise_std and float(noise_std) > 0.0:
                step = step + float(noise_std) * torch.randn_like(step)
            step = step * m_chunk
            p.data.add_(step.to(p.device))
        if reapply_mask and mask is not None:
            try:
                linear_layers = [m for m in getattr(actor, 'layers', []) if isinstance(m, nn.Linear)]
                for i, layer in enumerate(linear_layers):
                    if i >= len(mask):
                        continue
                    m = mask[i]
                    if not torch.is_tensor(m):
                        m = torch.as_tensor(np.asarray(m), dtype=torch.float32)
                    m = m.to(layer.weight.device).float()
                    if m.shape == layer.weight.shape:
                        layer.weight.data.mul_(m)
            except Exception:
                pass


def compute_non_redundant_direction(direction, neighbor_directions, eps=1e-8, clip_norm=1.0):
    """Project `direction` away from each neighbour direction in Gram-Schmidt order."""
    if direction is None:
        return None
    if not torch.is_tensor(direction):
        d = torch.as_tensor(np.asarray(direction, dtype=np.float32), device=device)
    else:
        d = direction.to(device).float()
    if d.dim() > 1:
        d = d.reshape(-1)
    d_norm = float(d.norm().item()) + float(eps)
    if d_norm <= float(eps):
        return d.detach().cpu().numpy().astype(np.float32)
    d_unit = d / d_norm

    perp = d.clone()
    if not neighbor_directions:
        out_norm = float(perp.norm().item()) + float(eps)
        return (perp / out_norm).detach().cpu().numpy().astype(np.float32)

    for g in neighbor_directions:
        if g is None:
            continue
        if not torch.is_tensor(g):
            g_t = torch.as_tensor(np.asarray(g, dtype=np.float32), device=device)
        else:
            g_t = g.to(device).float()
        if g_t.dim() > 1:
            g_t = g_t.reshape(-1)
        if g_t.numel() != perp.numel():
            continue
        g_norm = float(g_t.norm().item()) + float(eps)
        if g_norm <= float(eps):
            continue
        g_unit = g_t / g_norm
        coef = float(torch.dot(perp, g_unit).item())
        perp = perp - coef * g_unit
        if float(perp.norm().item()) <= float(eps):
            perp = torch.zeros_like(d)
            break

    pn = float(perp.norm().item())
    if pn <= float(eps):
        return d_unit.detach().cpu().numpy().astype(np.float32)
    if pn > float(clip_norm) and clip_norm > 0:
        perp = perp * (float(clip_norm) / (pn + float(eps)))
    out_norm = float(perp.norm().item()) + float(eps)
    return (perp / out_norm).detach().cpu().numpy().astype(np.float32)


def get_archive_neighbors(archive, target_idx, k=5, by='behavior'):
    """K-NN by BD distance; optionally restrict to similar-sparsity entries."""
    if not archive or target_idx is None or target_idx < 0 or target_idx >= len(archive):
        return []
    target = archive[target_idx]
    target_bc = target[2] if len(target) > 2 else None
    if target_bc is None:
        return []
    if torch.is_tensor(target_bc):
        target_bc = target_bc.detach().cpu().numpy()
    target_bc = np.asarray(target_bc, dtype=np.float32).flatten()
    # Archive layout: actor, reward, BD, masks, sparsity, inference metrics, critic, ...
    target_sp = float(target[4]) if len(target) > 4 and target[4] is not None else 0.0
    pairs = []
    for i, e in enumerate(archive):
        if i == target_idx:
            continue
        bc = e[2] if len(e) > 2 else None
        if bc is None:
            continue
        if torch.is_tensor(bc):
            bc = bc.detach().cpu().numpy()
        bc = np.asarray(bc, dtype=np.float32).flatten()
        if bc.size != target_bc.size or bc.size == 0:
            continue
        d = float(np.linalg.norm(bc - target_bc))
        sp = float(e[4]) if len(e) > 4 and e[4] is not None else 0.0
        if by == 'sparsity' and abs(sp - target_sp) > 0.15:
            continue
        pairs.append((d, i, e))
    pairs.sort(key=lambda x: x[0])
    out = []
    for d, i, e in pairs[: max(0, int(k))]:
        out.append((int(i), float(d), e))
    return out


def inject_critic_field_direction(actor, critic, critic_bank, memory,
                                  mask=None, archive=None, agent_idx=None,
                                  n_neighbors=3, alpha=0.05, noise_std=0.0,
                                  batch_size=256, grad_clip_norm=1.0) -> int:
    """One-step CFDI: compute critic direction, optionally orthogonalize against
    archive neighbours, then inject into actor parameters.
    Returns 1 on success, 0 on failure.
    """
    try:
        bs = min(int(batch_size), int(memory.size())) if hasattr(memory, 'size') else int(batch_size)
        if bs <= 0:
            return 0
        batch = memory.sample(bs)
        direction = compute_critic_actor_direction(
            actor=actor, critic=critic, replay_batch=batch,
            mask=mask, grad_clip_norm=grad_clip_norm,
        )
        if direction is None:
            return 0

        neighbor_dirs: List[np.ndarray] = []
        if archive is not None and agent_idx is not None and 0 <= agent_idx < len(archive):
            try:
                neighbors = get_archive_neighbors(archive, agent_idx, k=n_neighbors)
                for _, _, n_entry in neighbors:
                    n_critic = n_entry[6] if len(n_entry) > 6 else None
                    if n_critic is None or not isinstance(n_critic, nn.Module):
                        continue
                    try:
                        nb = memory.sample(min(int(batch_size), int(memory.size())))
                        nd = compute_critic_actor_direction(
                            actor=actor, critic=n_critic,
                            replay_batch=nb, mask=None,
                        )
                        if nd is not None:
                            neighbor_dirs.append(nd)
                    except Exception:
                        continue
            except Exception:
                pass

        if neighbor_dirs:
            direction = compute_non_redundant_direction(
                direction, neighbor_dirs, eps=1e-6, clip_norm=1.0,
            )

        apply_flat_direction_to_actor(
            actor=actor, flat_direction=direction, mask=mask,
            alpha=float(alpha), noise_std=float(noise_std), reapply_mask=True,
        )
        return 1
    except Exception as e:
        logger.debug(f"inject_critic_field_direction failed: {e}")
        return 0
