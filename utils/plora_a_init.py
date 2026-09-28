"""Current-task activation directions for frozen P-A; no historical statistics."""
import hashlib
import json
import logging
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def select_a_basis(moment, original_a, mode, seed):
    """Match each original Kaiming row norm, changing only its direction."""
    rank, dim = original_a.shape
    if mode == 'activation':
        _, vectors = torch.linalg.eigh(moment.detach().cpu().double())
        directions = vectors[:, -rank:].flip(1).T
    else:  # random_orthogonal control, independent of the training RNG
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.randn(dim, rank, generator=generator, dtype=torch.float64)
        directions = torch.linalg.qr(matrix, mode='reduced').Q.T
    # Canonical signs make eigensolver/QR sign ambiguity irrelevant.
    pivots = directions.abs().argmax(dim=1, keepdim=True)
    directions = directions * directions.gather(1, pivots).sign()
    norms = original_a.detach().cpu().double().norm(dim=1, keepdim=True)
    return (directions * norms).to(original_a)


@torch.no_grad()
def initialize_plora_a(network, modules, dataset, device, task, mode,
                       batch_size=48, batches=4, seed=1993):
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:batch_size * batches]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=0, generator=generator)
    modes = [(m, m.training) for m in network.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    moments, counts, handles = {}, {}, []

    def capture(layer):
        def hook(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            gram = x.T @ x
            if layer not in moments:
                moments[layer], counts[layer] = gram, len(x)
            else:
                moments[layer].add_(gram)
                counts[layer] += len(x)
        return hook

    try:
        with torch.random.fork_rng(devices=devices):
            network.eval()
            for layer, module in enumerate(modules):
                handles.append(module.register_forward_pre_hook(capture(layer)))
            for _, inputs, _ in loader:
                network.extract_vector(inputs.to(device), task_id=task)
    finally:
        for handle in handles:
            handle.remove()
        random.setstate(py_state)
        np.random.set_state(np_state)
        for module, training in modes:
            module.training = training

    records = []
    for layer, module in enumerate(modules):
        unit = module.P_lora[task]
        original = unit.A.weight.detach().clone()
        moment = (moments.pop(layer) / counts[layer]).cpu().double()
        replacement = select_a_basis(moment, original, mode, seed + layer)
        unit.A.weight.copy_(replacement)
        directions = torch.nn.functional.normalize(replacement.cpu().double(), dim=1)
        captured = ((directions @ moment) * directions).sum() / moment.trace().clamp_min(1e-12)
        record = dict(task=task, layer=layer, mode=mode, source='current_train',
                      images=len(indices), tokens=counts[layer], rank=len(original),
                      sample_hash=hashlib.sha256(indices.numpy().tobytes()).hexdigest()[:16],
                      input_energy_fraction=captured.item(),
                      row_norm_max_error=(replacement.norm(dim=1) - original.norm(dim=1)).abs().max().item(),
                      a_trainable=unit.A.weight.requires_grad, b_norm=unit.B.weight.norm().item())
        logging.info('P-A initialization %s', json.dumps(record))
        records.append(record)
    logging.info('P-A initialization completed: task=%s mode=%s seconds=%.3f',
                 task, mode, time.perf_counter() - started)
    return records
