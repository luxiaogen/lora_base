"""Training-only input statistics and a soft penalty on effective P conflict updates."""
import logging
import torch
from utils.p_region_diagnostic import preserve_diagnostic_state


def update_basis(module, moment, rank, layer, seed):
    # Equal task weight; snapshots are not claimed to be in an unchanged feature space.
    total = getattr(module, "_p_input_moment", torch.zeros_like(moment)) + moment
    module._p_input_moment = total.detach()
    rank = min(rank, total.shape[0])
    values, vectors = torch.linalg.eigh(total)
    module._p_input_basis = vectors[:, -rank:].detach()
    generator = torch.Generator(device="cpu").manual_seed(seed + layer)
    random = torch.randn(total.shape[0], rank, generator=generator)
    module._p_random_basis = torch.linalg.qr(random, mode="reduced").Q.to(total)
    return float(values[-rank:].sum() / values.sum().clamp_min(1e-12))


def collect_input_bases(network, modules, loader, device, rank=32, seed=1993):
    moments = [torch.zeros(m.qkv.in_features, m.qkv.in_features, device=device) for m in modules]
    counts = [0] * len(modules)
    handles = []

    def hook(index):
        def collect(_module, inputs):
            x = inputs[0].detach().float()
            # Fixed 16 token positions, including CLS; no random subsampling.
            positions = torch.linspace(0, x.shape[1] - 1, min(16, x.shape[1]), device=x.device).long()
            x = x[:, positions].reshape(-1, x.shape[-1])
            moments[index].add_(x.T @ x)
            counts[index] += x.shape[0]
        return collect

    with preserve_diagnostic_state(network):
        try:
            for i, module in enumerate(modules):
                handles.append(module.register_forward_pre_hook(hook(i)))
            for _, images, _ in loader:
                network.interface(images.to(device))
        finally:
            for handle in handles:
                handle.remove()
        for i, module in enumerate(modules):
            if not counts[i]:
                raise ValueError("Empty training input statistics")
            coverage = update_basis(module, moments[i] / counts[i], rank, i, seed)
            logging.info("P-input basis layer=%d rank=%d tokens=%d cumulative_energy=%.6f source=current_train", i, module._p_input_basis.shape[1], counts[i], coverage)


def conflict_subspace_loss(module, mode):
    if mode in ("none", "baseline") or module.cur_task == 0 or not hasattr(module, "_p_input_basis"):
        return None
    unit = module.P_lora[module.cur_task]
    if unit is None:
        return None
    raw = unit.B_weight @ unit.A_weight
    ratio, _ = module._conflict_parameters()
    _, mask = module._merge_base_and_conflict(raw, isolated=True, conflict_ratio=ratio)
    # Match the training forward: gamma is applied AFTER masking the unscaled BA.
    component = module.plora_gamma * module._safe_delta(raw, isolated=True) * mask.detach()
    basis = module._p_input_basis if mode == "old" else module._p_random_basis
    # Frobenius sum per layer; learner averages layers. No adaptive normalization.
    return (component @ basis.to(component)).square().sum()
