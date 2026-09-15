"""Evaluation-only interventions on already merged, suppressed P contributions."""
from contextlib import contextmanager
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConflictComponent(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('delta', torch.empty(0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + 'delta' in state_dict:
            self.delta = self.delta.new_empty(state_dict[prefix + 'delta'].shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@contextmanager
def conflict_weights(network, weights):
    """Temporarily apply per-image weights; never overwrite merged parameters."""
    modules = [m for m in network.modules() if hasattr(m, '_p_conflict_weights')]
    previous = [m._p_conflict_weights for m in modules]
    try:
        for module in modules:
            module._p_conflict_weights = weights
        yield
    finally:
        for module, value in zip(modules, previous):
            module._p_conflict_weights = value


@torch.no_grad()
def conservative_weights(task_probs):
    """Blend from baseline ones toward soft routing only when task evidence is sharp."""
    n = task_probs.shape[1]
    if n == 1:
        return torch.ones_like(task_probs)
    entropy = -(task_probs * task_probs.clamp_min(torch.finfo(task_probs.dtype).tiny).log()).sum(1)
    confidence = (1 - entropy / math.log(n)).clamp(0, 1).unsqueeze(1)
    return 1 - confidence * (1 - task_probs)


@torch.no_grad()
def diagnostic_logits(network, images, scale, true_tasks):
    """Oracle changes the features, not the set of candidate classes."""
    n = network.numtask
    with conflict_weights(network, None):
        baseline = network.interface(images)
    task_probs = (baseline * scale).softmax(1).reshape(len(images), n, network.class_num).sum(2)
    predicted_tasks = baseline.argmax(1) // network.class_num
    weights = {
        'ones': torch.ones_like(task_probs),
        'uniform': torch.full_like(task_probs, 1.0 / n),
        'soft': task_probs,
        'conservative': conservative_weights(task_probs),
        'predicted_onehot': F.one_hot(predicted_tasks, num_classes=n).to(task_probs),
        'oracle': F.one_hot(true_tasks, num_classes=n).to(task_probs),
    }
    outputs = {}
    for mode, values in weights.items():
        with conflict_weights(network, values):
            outputs[mode] = network.interface(images)
    torch.testing.assert_close(outputs['ones'], baseline, atol=1e-6, rtol=1e-5)
    return outputs, weights


def evaluate_p_conflict(network, loader, device, scale):
    """Read-only, post-CA diagnostic. No results feed training or normal predictions."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(m, m.training) for m in network.modules()]
    generators = {g for g in (getattr(loader, 'generator', None),
                              getattr(loader.sampler, 'generator', None)) if g is not None}
    generator_states = {g: g.get_state() for g in generators}
    cuda_devices = sorted({p.device.index for p in network.parameters() if p.device.type == 'cuda'})
    predictions = {mode: [] for mode in ('ones', 'uniform', 'soft', 'conservative', 'predicted_onehot', 'oracle')}
    labels, weight_sums = [], {}
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            network.eval()
            for _, images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                outputs, weights = diagnostic_logits(network, images, scale, targets // network.class_num)
                labels.append(targets.cpu())
                for mode, logits in outputs.items():
                    predictions[mode].append(logits.argmax(1).cpu())
                    weight_sums[mode] = weight_sums.get(mode, 0) + weights[mode].sum(0).cpu()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for generator, state in generator_states.items():
            generator.set_state(state)
        for module, training in modes:
            module.training = training
    targets = torch.cat(labels)
    predictions = {mode: torch.cat(values) for mode, values in predictions.items()}
    baseline_correct = predictions['ones'] == targets
    true_task = targets // network.class_num
    first_task_correct = predictions['ones'] // network.class_num == true_task
    n = network.numtask
    old = true_task < n - 1
    report = {}
    for mode, pred in predictions.items():
        correct = pred == targets
        pred_task = pred // network.class_num
        confusion = torch.bincount(true_task * n + pred_task, minlength=n*n).reshape(n, n)
        report[mode] = {
            'oracle_only': mode == 'oracle',
            'total': correct.double().mean().item() * 100,
            'old': correct[old].double().mean().item() * 100 if old.any() else None,
            'new': correct[~old].double().mean().item() * 100 if (~old).any() else None,
            'task_prediction': (pred_task == true_task).double().mean().item() * 100,
            'per_task': [correct[true_task == t].double().mean().item() * 100
                         if (true_task == t).any() else None for t in range(n)],
            'corrected': int((correct & ~baseline_correct).sum()),
            'broken': int((~correct & baseline_correct).sum()),
            'task_confusion_counts': confusion.tolist(),
            'mean_weights': (weight_sums[mode] / len(targets)).tolist(),
        }
        if mode == 'predicted_onehot':
            report[mode]['first_pass_task_accuracy'] = first_task_correct.double().mean().item() * 100
            report[mode]['by_first_pass_task'] = {}
            for name, group in [('correct', first_task_correct), ('wrong', ~first_task_correct)]:
                report[mode]['by_first_pass_task'][name] = {
                    'samples': int(group.sum()),
                    'corrected': int((group & correct & ~baseline_correct).sum()),
                    'broken': int((group & ~correct & baseline_correct).sum()),
                    'final_task_correct': int((group & (pred_task == true_task)).sum()),
                }
    return report


def summarize_p_conflict(reports):
    """Mean stage accuracy and forgetting through the latest partial task."""
    summary = {}
    for mode in reports[-1]:
        current = reports[-1][mode]['per_task']
        forgetting = None
        if len(reports) == len(current) and len(current) > 1:
            old_tasks = range(len(current) - 1)
            drops = [max(report[mode]['per_task'][i] for report in reports[i:]) - current[i]
                     for i in old_tasks]
            forgetting = sum(drops) / len(drops)
        summary[mode] = {
            'average': sum(report[mode]['total'] for report in reports) / len(reports),
            'forgetting': forgetting,
        }
    return summary
