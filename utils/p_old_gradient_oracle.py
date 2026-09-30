"""Privileged old-training-data reference, not an exemplar-free method."""
from contextlib import contextmanager
import hashlib
import json
import logging
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from utils.stage_audit import collect_logits, stage_metrics


def build_oracle_loaders(data_manager, known_classes, total_classes, seed, batch_size):
    dataset = data_manager.get_dataset(np.arange(total_classes), source='train', mode='test')
    labels = np.asarray(dataset.labels)
    selector, old_probe, new_probe = [], [], []
    for label in range(total_classes):
        positions = np.flatnonzero(labels == label)
        if label < known_classes:
            selector.extend(positions[:8:2].tolist())
            old_probe.extend(positions[1:8:2].tolist())
        else:
            new_probe.extend(positions[:4].tolist())
    return tuple(DataLoader(Subset(dataset, ids), batch_size=batch_size,
                           shuffle=(index == 0), num_workers=0,
                           generator=torch.Generator().manual_seed(seed + index))
                 for index, ids in enumerate((selector, old_probe, new_probe)))


@contextmanager
def oracle_eval(network):
    modes = [(module, module.training) for module in network.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            network.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training


def project_old_gradient(steps, gradients):
    dot = sum((step * gradient).sum() for step, gradient in zip(steps, gradients))
    norm_sq = sum(gradient.square().sum() for gradient in gradients)
    coefficient = dot.clamp_min(0) / norm_sq.clamp_min(1e-12)
    return [step - coefficient * gradient for step, gradient in zip(steps, gradients)]


@torch.no_grad()
def copy_p(params, values):
    for param, value in zip(params, values):
        param.copy_(value)


def apply_oracle_step(network, snapshots, old_batch, new_batch, loss_function, scale):
    """Correct the actual P SGD displacement; S/heads/momentum are untouched.

    All comparisons hold S/heads at their common post-SGD state. Acceptance
    checks actual losses on the selector old batch and the current new batch.
    This is not a guarantee about all old data or the whole optimizer step.
    """
    model = network.module if isinstance(network, torch.nn.DataParallel) else network
    params, before = zip(*snapshots)
    reference = [param.detach().clone() for param in params]
    selected = reference
    old_inputs, old_targets = old_batch
    new_inputs, new_targets = new_batch

    def measure():
        with torch.no_grad():
            old_logits = model.interface(old_inputs)
            new_logits = network(new_inputs)['logits']
            return dict(old_loss=float(F.cross_entropy(scale * old_logits, old_targets)),
                        new_loss=float(loss_function(new_logits, new_targets)),
                        old_accuracy=float(old_logits.argmax(1).eq(old_targets).float().mean() * 100),
                        new_local_accuracy=float(new_logits.argmax(1).eq(new_targets).float().mean() * 100))

    with oracle_eval(network):
        try:
            copy_p(params, before)
            old_loss = F.cross_entropy(scale * model.interface(old_inputs), old_targets)
            gradients = torch.autograd.grad(old_loss, params, allow_unused=True)
            gradients = [torch.zeros_like(param) if grad is None else grad.detach()
                         for param, grad in zip(params, gradients)]
            before_metrics = measure()
            steps = [end - start for end, start in zip(reference, before)]
            dot = float(sum((step * grad).sum() for step, grad in zip(steps, gradients)))
            proposal = project_old_gradient(steps, gradients)
            projected = [start + step for start, step in zip(before, proposal)] if dot > 0 else reference
            copy_p(params, reference)
            reference_metrics = measure()
            copy_p(params, projected)
            projected_metrics = measure()
            changed = any(not torch.equal(a, b) for a, b in zip(reference, projected))
            applied = (changed
                       and projected_metrics['old_loss'] <= before_metrics['old_loss'] + 1e-6
                       and projected_metrics['new_loss'] < before_metrics['new_loss'] - 1e-7)
            if applied:
                selected = projected
            return dict(applied=applied, gradient_conflict=dot > 0,
                        predicted_old_change=dot,
                        projected_predicted_old_change=float(sum((step * grad).sum()
                                                               for step, grad in zip(proposal, gradients))),
                        reference_b_step_norm=float(sum(x.square().sum() for x in steps).sqrt()),
                        selected_b_step_norm=float(sum((end - start).square().sum()
                                                      for end, start in zip(selected, before)).sqrt()),
                        before_p=before_metrics, reference=reference_metrics,
                        projected=projected_metrics,
                        selected=projected_metrics if applied else reference_metrics)
        finally:
            copy_p(params, selected)


def independent_probe(network, snapshots, reference, selected, loaders, device, known_classes):
    params = [param for param, _ in snapshots]
    results = {}
    try:
        for name, values in (('reference', reference), ('selected', selected)):
            copy_p(params, values)
            results[name] = {}
            for group, loader in zip(('old', 'new'), loaders):
                logits, targets, _ = collect_logits(network, loader, device)
                results[name][group] = stage_metrics(logits, targets, known_classes)[group]
    finally:
        copy_p(params, selected)
    return results


class OldGradientOracle:
    def __init__(self, data_manager, known_classes, total_classes, seed, batch_size):
        self.known_classes = known_classes
        self.loaders = build_oracle_loaders(data_manager, known_classes, total_classes, seed, batch_size)
        self.iterator = None
        self.counts = dict(sampled=0, gradient_conflicts=0, applied=0)
        logging.info('POldGradientOracleData %s', json.dumps(dict(
            source='old_train_privileged_oracle', formal_exemplar_free=False,
            samples=dict(zip(('old_selector', 'old_probe', 'new_probe'),
                             (len(loader.dataset) for loader in self.loaders))),
            sample_sha256=hashlib.sha256(json.dumps([loader.dataset.indices for loader in self.loaders]).encode()).hexdigest(),
            old_selector_probe_disjoint=True, probe_used_for_selection=False,
            probe_is_unseen_holdout=False, checkpoints=False, dense_history_components=False)))

    def step(self, network, snapshots, inputs, targets, loss_function, scale, task, epoch, batch):
        if self.iterator is None:
            self.iterator = iter(self.loaders[0])
        try:
            _, old_inputs, old_targets = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loaders[0])
            _, old_inputs, old_targets = next(self.iterator)
        device = inputs.device
        reference = [param.detach().clone() for param, _ in snapshots]
        row = apply_oracle_step(network, snapshots, (old_inputs.to(device), old_targets.to(device)),
                                (inputs, targets), loss_function, scale)
        self.counts['sampled'] += 1
        self.counts['gradient_conflicts'] += int(row['gradient_conflict'])
        self.counts['applied'] += int(row['applied'])
        if batch == 0:
            logging.info('POldGradientOracleStep %s', json.dumps(dict(
                task=task, epoch=epoch + 1, batch=batch + 1, source='train_selector',
                formal_exemplar_free=False, **row), allow_nan=False))
            if epoch in (0, 9, 19):
                selected = [param.detach().clone() for param, _ in snapshots]
                probe = independent_probe(network, snapshots, reference, selected,
                                          self.loaders[1:], device, self.known_classes)
                logging.info('POldGradientOracleProbe %s', json.dumps(dict(
                    task=task, epoch=epoch + 1, batch=batch + 1,
                    source='disjoint_train_probe_not_unseen_holdout',
                    used_for_selection=False, metrics=probe), allow_nan=False))

    def summary(self, task, epoch):
        logging.info('POldGradientOracleSummary %s', json.dumps(dict(
            task=task, epoch=epoch + 1, formal_exemplar_free=False, **self.counts)))
        self.counts = dict(sampled=0, gradient_conflicts=0, applied=0)
