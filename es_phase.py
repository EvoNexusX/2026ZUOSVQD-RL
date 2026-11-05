import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_actor_params(actor, params, param_mask=None, masked_param_indices=None):
    with torch.no_grad():
        flat = torch.nn.utils.parameters_to_vector(actor.parameters())
        if device.type == 'cuda' and not flat.is_cuda:
            flat = flat.to(device)
        if isinstance(params, np.ndarray):
            vec = torch.from_numpy(params).to(flat.dtype).to(flat.device)
        elif torch.is_tensor(params):
            vec = params.to(flat.device, dtype=flat.dtype)
        else:
            vec = torch.as_tensor(params, dtype=flat.dtype, device=flat.device)

        if param_mask is not None and masked_param_indices is not None:
            idx = torch.as_tensor(masked_param_indices, dtype=torch.long, device=flat.device)
            flat.index_copy_(0, idx, vec)
        else:
            flat.copy_(vec)

        torch.nn.utils.vector_to_parameters(flat, actor.parameters())


def get_actor_params(actor, param_mask=None, masked_param_indices=None) -> np.ndarray:
    with torch.no_grad():
        flat = torch.nn.utils.parameters_to_vector(actor.parameters())
        if param_mask is not None and masked_param_indices is not None:
            idx = torch.as_tensor(masked_param_indices, dtype=torch.long, device=flat.device)
            sub = flat.index_select(0, idx)
            return sub.detach().cpu().numpy()
        return flat.detach().cpu().numpy()


def apply_es_mean_to_actor(actor, es_optimizer, param_mask=None,
                           masked_param_indices=None, full_param_template=None):
    mu = es_optimizer.mu
    if param_mask is not None and masked_param_indices is not None:
        with torch.no_grad():
            idx = torch.as_tensor(masked_param_indices, dtype=torch.long, device=device)
            tmpl = torch.as_tensor(full_param_template, dtype=torch.float32, device=device)
            buf = tmpl.clone()
            vec = torch.from_numpy(mu).to(buf.device, dtype=buf.dtype)
            buf.index_copy_(0, idx, vec)
            torch.nn.utils.vector_to_parameters(buf, actor.parameters())
    else:
        set_actor_params(actor, mu)
    actor.to(device)
