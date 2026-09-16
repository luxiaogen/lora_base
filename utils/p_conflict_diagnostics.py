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
def conditional_onehot_weights(task_probs, predicted_tasks, margin_threshold):
    """Use predicted one-hot only when the top-two task probabilities are close."""
    n = task_probs.shape[1]
    if n == 1:
        return torch.ones_like(task_probs)
    top_two = task_probs.topk(2, dim=1).values
    margin = top_two[:, 0] - top_two[:, 1]
    ambiguous = margin < float(margin_threshold)
    predicted = F.one_hot(predicted_tasks, num_classes=n).to(task_probs)
    return torch.where(ambiguous.unsqueeze(1), predicted, torch.ones_like(task_probs))


@torch.no_grad()
def conditional_blend_weights(task_probs, predicted_tasks, margin_threshold):
    """Continuously blend from ones to predicted one-hot below the margin threshold."""
    n = task_probs.shape[1]
    threshold = float(margin_threshold)
    if n == 1 or threshold <= 0:
        return torch.ones_like(task_probs)
    top_two = task_probs.topk(2, dim=1).values
    margin = top_two[:, 0] - top_two[:, 1]
    strength = (1 - margin / threshold).clamp(0, 1).unsqueeze(1)
    predicted = F.one_hot(predicted_tasks, num_classes=n).to(task_probs)
    return 1 + strength * (predicted - 1)


@torch.no_grad()
def select_top2_counterfactual(baseline_logits, candidate_logits, candidate_tasks, scale, class_num):
    """Accept the best candidate only when it beats the original task margin."""
    batch, candidate_count, classes = candidate_logits.shape
    n = classes // class_num
    baseline_probs = (baseline_logits * scale).softmax(1).reshape(batch, n, class_num).sum(2)
    baseline_top2 = baseline_probs.topk(2, dim=1).values
    baseline_margin = baseline_top2[:, 0] - baseline_top2[:, 1]
    candidate_probs = (candidate_logits.reshape(-1, classes) * scale).softmax(1)
    candidate_probs = candidate_probs.reshape(batch, candidate_count, n, class_num).sum(3)
    own = candidate_probs.gather(2, candidate_tasks.unsqueeze(2)).squeeze(2)
    candidate_mask = F.one_hot(candidate_tasks, num_classes=n).bool()
    other = candidate_probs.masked_fill(candidate_mask, float('-inf')).max(2).values
    best_score, best_index = (own - other).max(1)
    rows = torch.arange(batch, device=baseline_logits.device)
    selected_logits = candidate_logits[rows, best_index]
    selected_tasks = candidate_tasks[rows, best_index]
    accepted = best_score > baseline_margin
    output = torch.where(accepted.unsqueeze(1), selected_logits, baseline_logits)
    return output, accepted, selected_tasks


@torch.no_grad()
def diagnostic_logits(network, images, scale, true_tasks, margin_threshold=0.1):
    """Oracle changes the features, not the set of candidate classes."""
    n = network.numtask
    with conflict_weights(network, None):
        baseline = network.interface(images)
    task_probs = (baseline * scale).softmax(1).reshape(len(images), n, network.class_num).sum(2)
    predicted_tasks = baseline.argmax(1) // network.class_num
    predicted_onehot = F.one_hot(predicted_tasks, num_classes=n).to(task_probs)
    true_onehot = F.one_hot(true_tasks, num_classes=n).to(task_probs)
    if n == 1:
        low_margin = torch.zeros(len(images), dtype=torch.bool, device=images.device)
    else:
        top_two = task_probs.topk(2, dim=1).values
        low_margin = top_two[:, 0] - top_two[:, 1] < float(margin_threshold)
    weights = {
        'ones': torch.ones_like(task_probs),
        'uniform': torch.full_like(task_probs, 1.0 / n),
        'soft': task_probs,
        'conservative': conservative_weights(task_probs),
        'predicted_onehot': predicted_onehot,
        'conditional_onehot': conditional_onehot_weights(task_probs, predicted_tasks, margin_threshold),
        'conditional_blend': conditional_blend_weights(task_probs, predicted_tasks, margin_threshold),
        'conditional_oracle': torch.where(low_margin.unsqueeze(1), true_onehot, torch.ones_like(task_probs)),
        'high_confidence_oracle': torch.where(low_margin.unsqueeze(1), torch.ones_like(task_probs), true_onehot),
        'oracle': true_onehot,
    }
    outputs = {'ones': baseline}
    for mode, values in weights.items():
        if mode == 'ones':
            continue
        with conflict_weights(network, values):
            logits = network.interface(images)
        if mode in {'conditional_onehot', 'conditional_blend', 'conditional_oracle',
                    'high_confidence_oracle'}:
            selected = values.ne(1).any(1).unsqueeze(1)
            logits = torch.where(selected, logits, baseline)
        outputs[mode] = logits
    counterfactual_weights = torch.ones_like(task_probs)
    counterfactual_logits = baseline.clone()
    if n > 1 and low_margin.any():
        ambiguous = low_margin.nonzero(as_tuple=False).squeeze(1)
        candidate_tasks = task_probs[ambiguous].topk(2, dim=1).indices
        candidate_logits = []
        for index in range(2):
            candidate_weights = F.one_hot(candidate_tasks[:, index], num_classes=n).to(task_probs)
            with conflict_weights(network, candidate_weights):
                candidate_logits.append(network.interface(images[ambiguous]))
        candidate_logits = torch.stack(candidate_logits, dim=1)
        selected_logits, accepted, selected_tasks = select_top2_counterfactual(
            baseline[ambiguous], candidate_logits, candidate_tasks, scale, network.class_num)
        counterfactual_logits[ambiguous] = selected_logits
        accepted_weights = torch.ones(len(ambiguous), n, device=images.device, dtype=task_probs.dtype)
        accepted_weights[accepted] = F.one_hot(
            selected_tasks[accepted], num_classes=n).to(task_probs)
        counterfactual_weights[ambiguous] = accepted_weights
    weights['top2_counterfactual'] = counterfactual_weights
    outputs['top2_counterfactual'] = counterfactual_logits
    return outputs, weights


def _distribution(values):
    if not values.numel():
        return {'count': 0, 'mean': None, 'q25': None, 'median': None, 'q75': None}
    values = values.double()
    quantiles = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype))
    return {
        'count': int(values.numel()),
        'mean': float(values.mean()),
        'q25': float(quantiles[0]),
        'median': float(quantiles[1]),
        'q75': float(quantiles[2]),
    }


def evaluate_p_conflict(network, loader, device, scale, margin_threshold=0.1):
    """Read-only, post-CA diagnostic. No results feed training or normal predictions."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(m, m.training) for m in network.modules()]
    generators = {g for g in (getattr(loader, 'generator', None),
                              getattr(loader.sampler, 'generator', None)) if g is not None}
    generator_states = {g: g.get_state() for g in generators}
    cuda_devices = sorted({p.device.index for p in network.parameters() if p.device.type == 'cuda'})
    prediction_modes = ('ones', 'uniform', 'soft', 'conservative', 'predicted_onehot',
                        'conditional_onehot', 'conditional_blend', 'conditional_oracle',
                        'high_confidence_oracle', 'top2_counterfactual', 'oracle')
    predictions = {mode: [] for mode in prediction_modes}
    labels, weight_sums = [], {}
    task_margins, task_entropies = [], []
    conditional_gates, counterfactual_gates, blend_strengths = [], [], []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            network.eval()
            for _, images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                outputs, weights = diagnostic_logits(
                    network, images, scale, targets // network.class_num, margin_threshold)
                labels.append(targets.cpu())
                task_probs = weights['soft']
                if network.numtask == 1:
                    margin = torch.ones(len(images), device=images.device)
                    entropy = torch.zeros(len(images), device=images.device)
                else:
                    top_two = task_probs.topk(2, dim=1).values
                    margin = top_two[:, 0] - top_two[:, 1]
                    entropy = -(task_probs * task_probs.clamp_min(
                        torch.finfo(task_probs.dtype).tiny).log()).sum(1) / math.log(network.numtask)
                task_margins.append(margin.cpu())
                task_entropies.append(entropy.cpu())
                conditional_gates.append(weights['conditional_onehot'].ne(1).any(1).cpu())
                counterfactual_gates.append(weights['top2_counterfactual'].ne(1).any(1).cpu())
                blend_strengths.append((1 - weights['conditional_blend'].min(1).values).cpu())
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
    task_margins = torch.cat(task_margins)
    task_entropies = torch.cat(task_entropies)
    conditional_gates = torch.cat(conditional_gates)
    counterfactual_gates = torch.cat(counterfactual_gates)
    blend_strengths = torch.cat(blend_strengths)
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
            'oracle_only': mode in {'conditional_oracle', 'high_confidence_oracle', 'oracle'},
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
    hard_correct = predictions['predicted_onehot'] == targets
    effect_groups = {
        'corrected': hard_correct & ~baseline_correct,
        'broken': ~hard_correct & baseline_correct,
    }
    report['predicted_onehot']['first_pass_evidence'] = {
        name: {
            'task_margin': _distribution(task_margins[group]),
            'normalized_entropy': _distribution(task_entropies[group]),
            'conditional_gate_rate': float(conditional_gates[group].double().mean() * 100)
            if group.any() else None,
        }
        for name, group in effect_groups.items()
    }
    conditional_correct = predictions['conditional_onehot'] == targets
    report['conditional_onehot'].update({
        'margin_threshold': float(margin_threshold),
        'selected_samples': int(conditional_gates.sum()),
        'selected_rate': float(conditional_gates.double().mean() * 100),
        'selected_corrected': int((conditional_gates & conditional_correct & ~baseline_correct).sum()),
        'selected_broken': int((conditional_gates & ~conditional_correct & baseline_correct).sum()),
    })
    for mode, selected in (
            ('conditional_oracle', conditional_gates),
            ('high_confidence_oracle', ~conditional_gates)):
        correct = predictions[mode] == targets
        report[mode].update({
            'margin_threshold': float(margin_threshold),
            'selected_samples': int(selected.sum()),
            'selected_rate': float(selected.double().mean() * 100),
            'selected_corrected': int((selected & correct & ~baseline_correct).sum()),
            'selected_broken': int((selected & ~correct & baseline_correct).sum()),
        })
    blend_active = blend_strengths > 0
    report['conditional_blend'].update({
        'margin_threshold': float(margin_threshold),
        'active_samples': int(blend_active.sum()),
        'active_rate': float(blend_active.double().mean() * 100),
        'mean_strength': float(blend_strengths.double().mean()),
    })
    counterfactual_correct = predictions['top2_counterfactual'] == targets
    evaluated = task_margins < float(margin_threshold)
    report['top2_counterfactual'].update({
        'margin_threshold': float(margin_threshold),
        'evaluated_samples': int(evaluated.sum()),
        'evaluated_rate': float(evaluated.double().mean() * 100),
        'accepted_samples': int(counterfactual_gates.sum()),
        'accepted_rate': float(counterfactual_gates.double().mean() * 100),
        'accepted_corrected': int((counterfactual_gates & counterfactual_correct & ~baseline_correct).sum()),
        'accepted_broken': int((counterfactual_gates & ~counterfactual_correct & baseline_correct).sum()),
    })
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
