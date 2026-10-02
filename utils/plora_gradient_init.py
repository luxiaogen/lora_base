"""Frozen P-A from current gradients and existing weights, without activation memory."""
import hashlib
import json
import logging
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def gradient_a_basis(gradient, wpre, residual, original_a, mode):
    """Change the input span, preserving A A^T (scale and conditioning)."""
    gradient, wpre, residual = [value.detach().float() for value in (gradient, wpre, residual)]
    rank, dim = original_a.shape
    if mode == 'random':
        replacement = original_a.detach().clone()  # Exact existing Kaiming control.
        directions = torch.linalg.qr(replacement.float().T, mode='reduced').Q.T
    else:
        _, singular, vectors = torch.linalg.svd(gradient, full_matrices=False)
        score = singular.square()
        if mode == 'weight_prior':
            # Unit-mean directional energies keep the two weight priors scale-independent.
            pre_energy = (wpre @ vectors.T).square().sum(0) / wpre.square().sum().clamp_min(1e-12)
            old_energy = (residual @ vectors.T).square().sum(0) / residual.square().sum().clamp_min(1e-12)
            score = score / (1 + dim * (pre_energy + old_energy))
        selected = score.argsort(descending=True, stable=True)[:rank]
        # Small double-precision QR/frame products avoid TF32 rounding of A's Gram.
        directions = torch.linalg.qr(vectors[selected].double().T, mode='reduced').Q.T
        pivots = directions.abs().argmax(1, keepdim=True)
        directions = directions * directions.gather(1, pivots).sign()
        left, scale, _ = torch.linalg.svd(original_a.detach().double(), full_matrices=False)
        replacement = ((left * scale) @ directions).to(original_a)
    record = {}
    for name, matrix in (('gradient', gradient), ('wpre', wpre), ('history', residual)):
        record[name + '_energy_fraction'] = float(
            (matrix @ directions.float().T).square().sum() / matrix.square().sum().clamp_min(1e-12))
    original = original_a.detach().double()
    new = replacement.double()
    record['row_norm_max_error'] = float((new.norm(dim=1) - original.norm(dim=1)).abs().max())
    record['a_gram_relative_error'] = float(
        (new @ new.T - original @ original.T).norm() / (original @ original.T).norm().clamp_min(1e-12))
    return replacement, record


def initialize_gradient_a(network, modules, dataset, device, task, mode, known_classes, loss_fn,
                          batch_size=48, batches=4, seed=1993):
    """Probe classification gradients only; never step an optimizer or retain features."""
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:batch_size * batches]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=0, generator=generator)
    modes = [(module, module.training) for module in network.modules()]
    weights = [module.qkv.weight for module in modules]
    flags = [weight.requires_grad for weight in weights]
    gradients = [torch.zeros_like(weight, dtype=torch.float32) for weight in weights]
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = sorted({param.device.index for param in network.parameters() if param.is_cuda})
    samples = 0
    try:
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            network.eval()
            for weight in weights:
                weight.requires_grad_(True)
            for _, inputs, targets in loader:
                targets = targets.to(device) - known_classes
                loss = loss_fn(network(inputs.to(device))['logits'], targets)
                values = torch.autograd.grad(loss, weights)
                for total, value in zip(gradients, values):
                    total.add_(value.detach().float(), alpha=len(targets))
                samples += len(targets)
    finally:
        for weight, flag in zip(weights, flags):
            weight.requires_grad_(flag)
        for module, training in modes:
            module.training = training
        random.setstate(py_state)
        np.random.set_state(np_state)

    records = []
    with torch.no_grad():
        for layer, (module, gradient) in enumerate(zip(modules, gradients)):
            gradient.div_(samples)
            plastic_gradient = gradient * (1 - module.general_mask.to(gradient))
            unit = module.P_lora[task]
            replacement, record = gradient_a_basis(plastic_gradient, module.pretrained_weight,
                module.qkv.weight.detach() - module.pretrained_weight, unit.A.weight, mode)
            if mode != 'random':
                unit.A.weight.copy_(replacement)
            record.update(task=task, layer=layer, mode=mode, images=samples, rank=unit.A.weight.shape[0],
                source='current_train_classification_gradient',
                sample_hash=hashlib.sha256(indices.numpy().tobytes()).hexdigest()[:16],
                masked_gradient_energy_fraction=float(
                    plastic_gradient.square().sum() / gradient.square().sum().clamp_min(1e-12)),
                a_trainable=unit.A.weight.requires_grad, b_norm=float(unit.B.weight.norm()))
            logging.info('PGradientAInit %s', json.dumps(record))
            records.append(record)
    logging.info('PGradientAInit completed: task=%s mode=%s seconds=%.3f',
                 task, mode, time.perf_counter() - started)
    return records


@torch.no_grad()
def basis_update_metrics(a, raw, effective, wpre, residual):
    """Weight-space proxies, not old-data sensitivity or a safety guarantee."""
    directions = torch.linalg.qr(a.detach().float().T, mode='reduced').Q.T
    row = dict(raw_norm=float(raw.norm()), effective_norm=float(effective.norm()))
    for name, delta in (('raw', raw.float()), ('effective', effective.float())):
        row[name + '_space_escape'] = float(
            (delta - (delta @ directions.T) @ directions).norm() / delta.norm().clamp_min(1e-12))
        gram = delta.T @ delta / delta.square().sum().clamp_min(1e-12)
        for prior_name, prior in (('wpre', wpre.float()), ('history', residual.float())):
            prior_gram = prior.T @ prior / prior.square().sum().clamp_min(1e-12)
            row[prior_name + '_overlap_' + name] = float((gram * prior_gram).sum())
    return row


@torch.no_grad()
def log_basis_update(module, raw, effective, task, mode):
    row = basis_update_metrics(module.P_lora[task].A.weight, raw, effective,
        module.pretrained_weight, module.qkv.weight.detach() - module.pretrained_weight)
    row.update(task=task, layer=int(module.layer_idx), mode=mode, source='actual_merge_P_delta')
    logging.info('PGradientAUpdate %s', json.dumps(row))
