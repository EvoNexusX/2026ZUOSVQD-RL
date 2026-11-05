import torch
import torch.nn as nn
import numpy as np
from typing import List

from mask_config import get_mask_threshold
from networks import Actor


def _resolve_actor_dims(actor) -> tuple:
    input_dim = getattr(actor, 'state_dim', None)
    output_dim = getattr(actor, 'action_dim', None)
    if input_dim is not None and output_dim is not None:
        return int(input_dim), int(output_dim)
    linears = [l for l in getattr(actor, 'layers', []) if isinstance(l, nn.Linear)]
    if not linears:
        raise TypeError(
            "Cannot determine actor input/output dimensions; provide "
            "'input_dim'/'output_dim' or an actor exposing state_dim/action_dim."
        )
    return int(input_dim or linears[0].in_features), int(output_dim or linears[-1].out_features)


def build_structurally_pruned_actor(actor, masks, input_dim=None, output_dim=None,
                                    threshold=None):
    """Build a structurally-pruned Actor by collapsing zero-output channels.

    `masks` must already be binary 2-D weight masks aligned with each Linear
    layer's ``weight`` (see ``neuron_masks_to_weight_masks``); they are used as
    given rather than re-binarised.
    """
    if actor is None:
        raise ValueError("actor cannot be empty")
    if masks is None:
        return actor
    if threshold is None:
        threshold = get_mask_threshold()

    try:
        device = next(actor.parameters()).device
    except Exception:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if input_dim is None or output_dim is None:
        input_dim, output_dim = _resolve_actor_dims(actor)
    input_dim, output_dim = int(input_dim), int(output_dim)
    bin_masks = masks

    linear_layers: List[nn.Linear] = []
    for lyr in getattr(actor, 'layers', []):
        if isinstance(lyr, nn.Linear):
            linear_layers.append(lyr)
    if not linear_layers:
        return actor

    kept_out_indices_per_layer: List[np.ndarray] = []
    for li, lyr in enumerate(linear_layers):
        W = lyr.weight.detach().cpu().numpy()
        in_dim_l, out_dim_l = W.shape[1], W.shape[0]
        m = bin_masks[li] if li < len(bin_masks) else np.ones((in_dim_l, out_dim_l), dtype=np.float32)
        m = np.asarray(m)
        if m.shape != (in_dim_l, out_dim_l):
            if m.T.shape == (in_dim_l, out_dim_l):
                m = m.T
            else:
                m = np.ones((in_dim_l, out_dim_l), dtype=np.float32)
        col_means = np.mean(m, axis=0)
        keep_cols = np.where(col_means >= threshold)[0]
        if keep_cols.size == 0 and out_dim_l > 0:
            col_norms = np.linalg.norm(W, axis=1)
            keep_cols = np.array([int(np.argmax(col_norms))], dtype=np.int64)
        kept_out_indices_per_layer.append(keep_cols.astype(np.int64))

    pruned_hidden_sizes: List[int] = [
        int(keep_cols.size) for keep_cols in kept_out_indices_per_layer[:-1]
    ]

    new_actor = Actor(
        state_dim=input_dim, action_dim=output_dim,
        max_action=getattr(actor, 'max_action', 1.0),
        args=getattr(actor, 'args', None) or getattr(actor, 'config', None) or type('A', (), {})(),
        hidden_layers=pruned_hidden_sizes,
    )
    new_actor = new_actor.to(device)

    with torch.no_grad():
        src0: nn.Linear = linear_layers[0]
        dst0: nn.Linear = new_actor.layers[0]
        idx_out0 = kept_out_indices_per_layer[0]
        W0 = src0.weight.detach().cpu().numpy()[idx_out0, :]
        b0 = src0.bias.detach().cpu().numpy()[idx_out0] if src0.bias is not None else None
        dst0.weight.copy_(torch.as_tensor(W0, dtype=torch.float32))
        if b0 is not None and dst0.bias is not None:
            dst0.bias.copy_(torch.as_tensor(b0, dtype=torch.float32))

        for li in range(1, len(pruned_hidden_sizes)):
            src = linear_layers[li]
            dst = new_actor.layers[li]
            idx_in = kept_out_indices_per_layer[li - 1]
            idx_out = kept_out_indices_per_layer[li]
            W = src.weight.detach().cpu().numpy()[idx_out[:, None], idx_in].reshape(idx_out.size, idx_in.size)
            b = src.bias.detach().cpu().numpy()[idx_out] if src.bias is not None else None
            dst.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
            if b is not None and dst.bias is not None:
                dst.bias.copy_(torch.as_tensor(b, dtype=torch.float32))

        src_last = linear_layers[-1]
        dst_last = new_actor.layers[-1]
        idx_in_last = (kept_out_indices_per_layer[-2]
                       if len(kept_out_indices_per_layer) >= 2
                       else np.arange(src_last.weight.shape[1], dtype=np.int64))
        Wlast = src_last.weight.detach().cpu().numpy()[:, idx_in_last]
        blast = src_last.bias.detach().cpu().numpy() if src_last.bias is not None else None
        dst_last.weight.copy_(torch.as_tensor(Wlast, dtype=torch.float32))
        if blast is not None and dst_last.bias is not None:
            dst_last.bias.copy_(torch.as_tensor(blast, dtype=torch.float32))

    return new_actor
