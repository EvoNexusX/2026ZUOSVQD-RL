from typing import List, Optional
import torch.nn as nn


def _linear_shapes(network):
    """[(in, out), ...] of the network's Linear layers, in forward order."""
    try:
        layers = [m for m in network.modules() if isinstance(m, nn.Linear)]
    except Exception:
        return []
    return [(int(l.in_features), int(l.out_features)) for l in layers]


def _params_from_shapes(shapes) -> int:
    """Weight + bias parameter count of an MLP with the given layer shapes."""
    total = 0
    for in_f, out_f in shapes:
        total += in_f * out_f + out_f
    return int(total)


def measure_pruned_actor(network, input_dim: int, output_dim: int,
                         dense_hidden: Optional[List[int]] = None,
                         max_sparsity: float = 0.90):
    """Measure compact actor parameters and FLOPs against the configured dense actor.
    Parameter sparsity is 1 - compact_params / dense_params; mask zeros are not used."""
    shapes = _linear_shapes(network)
    if not shapes:
        return 0.0, {'flops_dense': 0, 'flops_sparse': 0}

    n_hidden = max(0, len(shapes) - 1)
    if dense_hidden is None:
        dense_hidden = []
    dense_hidden = [int(h) for h in dense_hidden]
    if len(dense_hidden) < n_hidden:
        # Without a compatible dense reference, report the archived network as dense.
        dense_hidden = dense_hidden + [shapes[0][0]] * (n_hidden - len(dense_hidden))
    dense_shapes = [(int(input_dim), dense_hidden[0])] if n_hidden else []
    for i in range(n_hidden):
        out_ref = dense_hidden[i + 1] if i + 1 < len(dense_hidden) else int(output_dim)
        dense_shapes.append((dense_shapes[-1][1] if dense_shapes else int(input_dim), out_ref))
    if not dense_shapes:
        dense_shapes = shapes
    dense_shapes[-1] = (dense_shapes[-1][0], int(output_dim))

    pruned_params = _params_from_shapes(shapes)
    dense_params = _params_from_shapes(dense_shapes)
    sparsity = 0.0
    if dense_params > 0:
        sparsity = 1.0 - (float(pruned_params) / float(dense_params))
    sparsity = float(min(max(sparsity, 0.0), float(max_sparsity)))

    def _flops(sh):
        return int(sum(2 * i * o + o for i, o in sh))

    flops_dense = _flops(dense_shapes)
    flops_sparse = _flops(shapes)
    if flops_dense > 0:
        flops_sparse = int(min(flops_sparse, flops_dense))
    return sparsity, {
        'flops_dense': int(flops_dense),
        'flops_sparse': int(flops_sparse),
    }
