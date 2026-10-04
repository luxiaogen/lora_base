"""Current-train feedback for limited P permissions; no historical data buffer."""
from contextlib import contextmanager
import hashlib
import json
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.dualmask_core_audit import round_robin_indices, stage_cost, _logits_stats


@torch.no_grad()
def choose_release(utility, extra, protect, mode, seed):
    result = torch.zeros_like(protect, dtype=torch.bool)
    for projection, (score, change, allowed, out) in enumerate(zip(
            utility.chunk(3), extra.chunk(3), protect.chunk(3), result.chunk(3))):
        valid = allowed.bool().flatten().nonzero(as_tuple=True)[0]
        positives = int((score.flatten()[valid] > 0).sum())
        count = min(int(.1 * valid.numel()), positives)
        if not count:
            continue
        if mode == 'random':
            generator = torch.Generator().manual_seed(seed + 1009 * projection)
            order = torch.randperm(valid.numel(), generator=generator)[:count].to(valid.device)
        else:
            values = score.flatten()[valid] if mode == 'benefit' else change.abs().flatten()[valid]
            order = values.argsort(descending=True, stable=True)[:count]
        out.flatten()[valid[order]] = True
    return result


def permission_release_gate(delta, protect, release, strength, conflict, match):
    base = (1 - protect) * conflict
    gate = ((1 - protect) + (1 - strength) * release.to(delta) * protect) * conflict
    scales = delta.new_ones(3)
    if match:
        with torch.no_grad():
            reference = (delta.detach().float() * base).reshape(3, -1).norm(dim=1)
            candidate = (delta.detach().float() * gate).reshape(3, -1).norm(dim=1)
            scales = torch.where(candidate > 0, reference / candidate.clamp_min(1e-30),
                                 torch.ones_like(candidate)).clamp(max=1)
        gate = gate * scales.repeat_interleave(delta.shape[0] // 3).unsqueeze(1).to(gate)
    return gate, scales


@contextmanager
def probe_state(network, modules, device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in network.modules()]
    flags = [(module.qkv.weight, module.qkv.weight.requires_grad) for module in modules]
    disabled = [module._p_release_disabled for module in modules]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            network.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for weight, trainable in flags:
            weight.requires_grad_(trainable)
        for module, training in modes:
            module.training = training
        for module, value in zip(modules, disabled):
            module._p_release_disabled = value


def _loader(dataset, indices, batch_size):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                      num_workers=0, generator=torch.Generator().manual_seed(1993))


def _probe_losses(network, loader, device, known, loss_fn):
    total, count = 0., 0
    with torch.no_grad():
        for _, inputs, labels in loader:
            labels = labels.to(device) - known
            total += float(loss_fn(network(inputs.to(device))['logits'], labels)) * len(labels)
            count += len(labels)
    return total / count


def refresh_release(learner, dataset, epoch, loss_fn):
    network = learner._network.module if isinstance(learner._network, torch.nn.DataParallel) else learner._network
    modules = list(learner._iter_lora_modules())
    indices = round_robin_indices(dataset.labels, range(learner._known_classes, learner._total_classes), 320)
    chosen, check = indices[:160], indices[160:]
    loader = _loader(dataset, chosen, learner.batch_size)
    weights = [module.qkv.weight for module in modules]
    gradients = [torch.zeros_like(weight, dtype=torch.float32) for weight in weights]
    samples = 0
    with stage_cost(learner, 'release_probe'), probe_state(network, modules, learner._device):
        for module in modules:
            module._p_release_disabled = True
            module.qkv.weight.requires_grad_(True)
        with torch.enable_grad():
            for _, inputs, labels in loader:
                labels = labels.to(learner._device) - learner._known_classes
                loss = loss_fn(network(inputs.to(learner._device))['logits'], labels)
                values = torch.autograd.grad(loss, weights)
                for total, value in zip(gradients, values):
                    total.add_(value.detach().float(), alpha=len(labels))
                samples += len(labels)
        with torch.no_grad():
            for layer, (module, gradient) in enumerate(zip(modules, gradients)):
                gradient.div_(samples)
                unit = module.P_lora[learner._cur_task]
                delta = unit.B_weight @ unit.A_weight
                protect = module._p_protect_mask().to(delta)
                _, _, selected = module._p_permission_delta(delta)
                conflict = 1 - module._conflict_parameters()[1] * selected.to(delta)
                extra = module.plora_gamma * delta * protect * (1 - module.effective_protect_strength) * conflict
                utility = -gradient * extra
                seed = int(learner.args['seed']) + 10007 * learner._cur_task + 9176 * layer + 53 * epoch
                mode = learner.args['p_permission_release']
                mask = choose_release(utility, extra, protect, mode, seed)
                module.p_permission_release_mask = mask
                logging.info('PPermissionRelease %s', json.dumps(dict(task=learner._cur_task,
                    epoch=epoch, layer=layer, mode=mode, source='current_train_classification_gradient',
                    position=learner.args.get('p_permission_position', 'wpre'), samples=samples,
                    sample_sha256=hashlib.sha256(np.asarray(chosen, dtype=np.int64).tobytes()).hexdigest(),
                    protected=[int(p.sum()) for p in protect.chunk(3)],
                    released=[int(p.sum()) for p in mask.chunk(3)],
                    positive_utility=[int(((u > 0) & p.bool()).sum()) for u, p in zip(utility.chunk(3), protect.chunk(3))],
                    selected_utility=float(utility[mask].sum()),
                    norm_match=learner.args.get('p_permission_norm_match', True))))
            for module in modules:
                module.qkv.weight.requires_grad_(False)
        if check:
            check_loader = _loader(dataset, check, learner.batch_size)
            before = _probe_losses(network, check_loader, learner._device, learner._known_classes, loss_fn)
            for module in modules:
                module._p_release_disabled = False
            after = _probe_losses(network, check_loader, learner._device, learner._known_classes, loss_fn)
            logging.info('PPermissionTrainDiagnostic %s', json.dumps(dict(task=learner._cur_task,
                epoch=epoch, source='disjoint_current_train_probe_not_holdout', count=len(check),
                reference_loss=before, released_loss=after, loss_change=after-before)))


def report_release(learner, dataset, epoch):
    if learner._cur_task not in (1, 5, 9) or epoch not in (1, 5, 10, 20):
        return
    network = learner._network.module if isinstance(learner._network, torch.nn.DataParallel) else learner._network
    modules = list(learner._iter_lora_modules())
    indices = round_robin_indices(dataset.labels, range(learner._known_classes), 512)
    indices += round_robin_indices(dataset.labels, range(learner._known_classes, learner._total_classes), 128)
    totals = {name: dict(count=0, reference_correct=0, released_correct=0, changed=0,
                        reference_margin=0., released_margin=0.) for name in ('old', 'new')}
    with stage_cost(learner, 'mechanism_diagnostic'), probe_state(network, modules, learner._device), torch.no_grad():
        for _, inputs, labels in _loader(dataset, indices, learner.batch_size):
            inputs, labels = inputs.to(learner._device), labels.to(learner._device)
            predictions = []
            for disabled in (True, False):
                for module in modules:
                    module._p_release_disabled = disabled
                predictions.append(_logits_stats(network.interface(inputs), labels))
            for name, selected in (('old', labels < learner._known_classes), ('new', labels >= learner._known_classes)):
                first, second = predictions
                row = totals[name]
                row['count'] += int(selected.sum())
                row['reference_correct'] += int((first[0][selected] == labels[selected]).sum())
                row['released_correct'] += int((second[0][selected] == labels[selected]).sum())
                row['changed'] += int((first[0][selected] != second[0][selected]).sum())
                row['reference_margin'] += float(first[1][selected].sum())
                row['released_margin'] += float(second[1][selected].sum())
        for partition, row in totals.items():
            count = row['count']
            if count:
                for key in ('reference_margin', 'released_margin'):
                    row[key] /= count
                row['reference_accuracy'] = 100 * row['reference_correct'] / count
                row['released_accuracy'] = 100 * row['released_correct'] / count
            logging.info('PPermissionTestDiagnostic %s', json.dumps(dict(task=learner._cur_task,
                epoch=epoch, partition=partition, source='test_only_fixed_order', scope='all_seen', **row)))
