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
def counterfactual_evidence(baseline_logits, candidate_logits, candidate_tasks, scale, class_num):
    """Measure how each candidate changes its own task evidence and competitors."""
    batch, candidate_count, classes = candidate_logits.shape
    n = classes // class_num
    baseline_probs = (baseline_logits * scale).softmax(1).reshape(batch, n, class_num).sum(2)
    candidate_probs = (candidate_logits.reshape(-1, classes) * scale).softmax(1)
    candidate_probs = candidate_probs.reshape(batch, candidate_count, n, class_num).sum(3)
    baseline_task_max = baseline_logits.reshape(batch, n, class_num).max(2).values
    candidate_task_max = candidate_logits.reshape(
        batch, candidate_count, n, class_num).max(3).values
    candidate_mask = F.one_hot(candidate_tasks, num_classes=n).bool()
    baseline_probs = baseline_probs.unsqueeze(1).expand(-1, candidate_count, -1)
    baseline_own = baseline_probs.gather(2, candidate_tasks.unsqueeze(2)).squeeze(2)
    baseline_other = baseline_probs.masked_fill(candidate_mask, float('-inf')).max(2).values
    own = candidate_probs.gather(2, candidate_tasks.unsqueeze(2)).squeeze(2)
    other = candidate_probs.masked_fill(candidate_mask, float('-inf')).max(2).values
    baseline_top_class = baseline_task_max.gather(1, candidate_tasks)
    candidate_top_class = candidate_task_max.gather(
        2, candidate_tasks.unsqueeze(2)).squeeze(2)
    absolute_margin = own - other
    return {
        'absolute_margin': absolute_margin,
        'margin_gain': absolute_margin - (baseline_own - baseline_other),
        'own_gain': own - baseline_own,
        'other_change': other - baseline_other,
        'top_class_gain': candidate_top_class - baseline_top_class,
    }


@torch.no_grad()
def select_counterfactual(baseline_logits, candidate_logits, candidate_tasks, scale, class_num,
                          score_mode='absolute_margin'):
    """Select a candidate and retain the original logits when its evidence does not improve."""
    batch = len(baseline_logits)
    evidence = counterfactual_evidence(
        baseline_logits, candidate_logits, candidate_tasks, scale, class_num)
    scores = evidence[score_mode]
    best_score, best_index = scores.max(1)
    rows = torch.arange(batch, device=baseline_logits.device)
    selected_logits = candidate_logits[rows, best_index]
    selected_tasks = candidate_tasks[rows, best_index]
    if score_mode == 'absolute_margin':
        n = baseline_logits.shape[1] // class_num
        baseline_probs = (baseline_logits * scale).softmax(1).reshape(batch, n, class_num).sum(2)
        baseline_top2 = baseline_probs.topk(2, dim=1).values
        accepted = best_score > baseline_top2[:, 0] - baseline_top2[:, 1]
    else:
        accepted = best_score > 0
    output = torch.where(accepted.unsqueeze(1), selected_logits, baseline_logits)
    selected_evidence = {name: values[rows, best_index] for name, values in evidence.items()}
    return output, accepted, selected_tasks, selected_evidence


@torch.no_grad()
def select_top2_counterfactual(baseline_logits, candidate_logits, candidate_tasks, scale, class_num):
    """Keep the original Top-2 selection behavior for matched comparisons."""
    output, accepted, selected_tasks, _ = select_counterfactual(
        baseline_logits, candidate_logits, candidate_tasks, scale, class_num)
    return output, accepted, selected_tasks


@torch.no_grad()
def diagnostic_logits(network, images, scale, true_tasks, margin_threshold=0.1,
                      return_details=False):
    """Oracle changes the features, not the set of candidate classes."""
    n = network.numtask
    with conflict_weights(network, None):
        baseline = network.interface(images)
    task_probs = (baseline * scale).softmax(1).reshape(len(images), n, network.class_num).sum(2)
    task_logits = baseline.reshape(len(images), n, network.class_num).max(2).values
    predicted_tasks = baseline.argmax(1) // network.class_num
    predicted_onehot = F.one_hot(predicted_tasks, num_classes=n).to(task_probs)
    true_onehot = F.one_hot(true_tasks, num_classes=n).to(task_probs)
    if n == 1:
        low_margin = torch.zeros(len(images), dtype=torch.bool, device=images.device)
        true_task_in_top2 = torch.ones(len(images), dtype=torch.bool, device=images.device)
    else:
        top_two = task_probs.topk(2, dim=1)
        low_margin = top_two.values[:, 0] - top_two.values[:, 1] < float(margin_threshold)
        true_task_in_top2 = top_two.indices.eq(true_tasks.unsqueeze(1)).any(1)
    top2_oracle_active = low_margin & true_task_in_top2
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
        'top2_task_oracle': torch.where(
            top2_oracle_active.unsqueeze(1), true_onehot, torch.ones_like(task_probs)),
        'oracle': true_onehot,
    }
    outputs = {'ones': baseline}
    for mode, values in weights.items():
        if mode == 'ones':
            continue
        with conflict_weights(network, values):
            logits = network.interface(images)
        if mode in {'conditional_onehot', 'conditional_blend', 'conditional_oracle',
                    'high_confidence_oracle', 'top2_task_oracle'}:
            selected = values.ne(1).any(1).unsqueeze(1)
            logits = torch.where(selected, logits, baseline)
        outputs[mode] = logits
    counterfactual_modes = {
        'top2_counterfactual': 'absolute_margin',
        'union_counterfactual': 'absolute_margin',
        'union_delta_margin': 'margin_gain',
        'union_own_gain': 'own_gain',
        'union_top_class_gain': 'top_class_gain',
    }
    details = {}
    for mode in counterfactual_modes:
        weights[mode] = torch.ones_like(task_probs)
        outputs[mode] = baseline.clone()
        details[mode] = {
            'accepted': torch.zeros(len(images), dtype=torch.bool, device=images.device),
            **{
                name: torch.full((len(images),), float('nan'), device=images.device,
                                 dtype=task_probs.dtype)
                for name in ('absolute_margin', 'margin_gain', 'own_gain', 'other_change',
                             'top_class_gain')
            },
        }
    weights['union_task_oracle'] = torch.ones_like(task_probs)
    outputs['union_task_oracle'] = baseline.clone()
    if n > 1 and low_margin.any():
        ambiguous = low_margin.nonzero(as_tuple=False).squeeze(1)
        probability_candidates = task_probs[ambiguous].topk(2, dim=1).indices
        max_logit_candidates = task_logits[ambiguous].topk(2, dim=1).indices
        candidate_tasks = torch.cat((probability_candidates, max_logit_candidates), dim=1)
        candidate_logits = []
        for index in range(candidate_tasks.shape[1]):
            candidate_weights = F.one_hot(candidate_tasks[:, index], num_classes=n).to(task_probs)
            with conflict_weights(network, candidate_weights):
                candidate_logits.append(network.interface(images[ambiguous]))
        candidate_logits = torch.stack(candidate_logits, dim=1)
        candidate_sets = {
            'top2_counterfactual': (candidate_tasks[:, :2], candidate_logits[:, :2]),
            'union_counterfactual': (candidate_tasks, candidate_logits),
            'union_delta_margin': (candidate_tasks, candidate_logits),
            'union_own_gain': (candidate_tasks, candidate_logits),
            'union_top_class_gain': (candidate_tasks, candidate_logits),
        }
        for mode, score_mode in counterfactual_modes.items():
            mode_tasks, mode_logits = candidate_sets[mode]
            selected_logits, accepted, selected_tasks, selected_evidence = select_counterfactual(
                baseline[ambiguous], mode_logits, mode_tasks, scale, network.class_num,
                score_mode=score_mode)
            outputs[mode][ambiguous] = selected_logits
            accepted_weights = torch.ones(
                len(ambiguous), n, device=images.device, dtype=task_probs.dtype)
            accepted_weights[accepted] = F.one_hot(
                selected_tasks[accepted], num_classes=n).to(task_probs)
            weights[mode][ambiguous] = accepted_weights
            details[mode]['accepted'][ambiguous] = accepted
            for name, values in selected_evidence.items():
                details[mode][name][ambiguous] = values

        true_candidates = candidate_tasks.eq(true_tasks[ambiguous].unsqueeze(1))
        covered = true_candidates.any(1)
        match_index = true_candidates.to(torch.int64).argmax(1)
        rows = torch.arange(len(ambiguous), device=images.device)
        selected = candidate_logits[rows, match_index]
        outputs['union_task_oracle'][ambiguous[covered]] = selected[covered]
        weights['union_task_oracle'][ambiguous[covered]] = true_onehot[ambiguous[covered]]
    if return_details:
        return outputs, weights, details
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
                        'high_confidence_oracle', 'top2_counterfactual',
                        'union_counterfactual', 'union_delta_margin', 'union_own_gain',
                        'union_top_class_gain',
                        'top2_task_oracle', 'union_task_oracle', 'oracle')
    predictions = {mode: [] for mode in prediction_modes}
    labels, weight_sums = [], {}
    task_margins, task_entropies = [], []
    conditional_gates, blend_strengths = [], []
    counterfactual_modes = ('top2_counterfactual', 'union_counterfactual',
                            'union_delta_margin', 'union_own_gain', 'union_top_class_gain')
    counterfactual_gates = {mode: [] for mode in counterfactual_modes}
    coverage_names = ('prob_top2', 'prob_top3', 'prob_top5', 'maxlogit_top2', 'union_top2')
    candidate_coverages = {name: [] for name in coverage_names}
    candidate_counts = {name: [] for name in coverage_names}
    evidence_names = ('absolute_margin', 'margin_gain', 'own_gain', 'other_change',
                      'top_class_gain')
    selection_evidence = {
        mode: {name: [] for name in evidence_names} for mode in counterfactual_modes
    }
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            network.eval()
            for _, images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                outputs, weights, details = diagnostic_logits(
                    network, images, scale, targets // network.class_num, margin_threshold,
                    return_details=True)
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
                for mode in counterfactual_modes:
                    counterfactual_gates[mode].append(details[mode]['accepted'].cpu())
                    for name in evidence_names:
                        selection_evidence[mode][name].append(details[mode][name].cpu())
                true_tasks = targets // network.class_num
                if network.numtask == 1:
                    for name in coverage_names:
                        candidate_coverages[name].append(torch.ones(len(images), dtype=torch.bool))
                        candidate_counts[name].append(torch.ones(len(images), dtype=torch.long))
                else:
                    max_logits = outputs['ones'].reshape(
                        len(images), network.numtask, network.class_num).max(2).values
                    pools = {
                        'prob_top2': task_probs.topk(2, dim=1).indices,
                        'prob_top3': task_probs.topk(min(3, network.numtask), dim=1).indices,
                        'prob_top5': task_probs.topk(min(5, network.numtask), dim=1).indices,
                        'maxlogit_top2': max_logits.topk(2, dim=1).indices,
                    }
                    pools['union_top2'] = torch.cat(
                        (pools['prob_top2'], pools['maxlogit_top2']), dim=1)
                    for name, candidates in pools.items():
                        covered = candidates.eq(true_tasks.unsqueeze(1)).any(1)
                        candidate_coverages[name].append(covered.cpu())
                        unique = torch.ones_like(candidates, dtype=torch.bool)
                        for column in range(1, candidates.shape[1]):
                            unique[:, column] = ~candidates[:, :column].eq(
                                candidates[:, column].unsqueeze(1)).any(1)
                        candidate_counts[name].append(unique.sum(1).cpu())
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
    counterfactual_gates = {
        mode: torch.cat(values) for mode, values in counterfactual_gates.items()
    }
    candidate_coverages = {
        name: torch.cat(values) for name, values in candidate_coverages.items()
    }
    candidate_counts = {name: torch.cat(values) for name, values in candidate_counts.items()}
    selection_evidence = {
        mode: {name: torch.cat(values) for name, values in evidence.items()}
        for mode, evidence in selection_evidence.items()
    }
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
            'oracle_only': mode in {
                'conditional_oracle', 'high_confidence_oracle', 'top2_task_oracle',
                'union_task_oracle', 'oracle'},
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
    evaluated = task_margins < float(margin_threshold)
    evaluated_wrong = evaluated & ~first_task_correct
    for mode in counterfactual_modes:
        gate = counterfactual_gates[mode]
        correct = predictions[mode] == targets
        corrected = gate & correct & ~baseline_correct
        broken = gate & ~correct & baseline_correct
        report[mode].update({
            'margin_threshold': float(margin_threshold),
            'evaluated_samples': int(evaluated.sum()),
            'evaluated_rate': float(evaluated.double().mean() * 100),
            'accepted_samples': int(gate.sum()),
            'accepted_rate': float(gate.double().mean() * 100),
            'accepted_corrected': int(corrected.sum()),
            'accepted_broken': int(broken.sum()),
            'selected_evidence': {
                group_name: {
                    name: _distribution(selection_evidence[mode][name][group])
                    for name in evidence_names
                }
                for group_name, group in (('corrected', corrected), ('broken', broken))
            },
        })

    coverage_report = {}
    for name, coverage in candidate_coverages.items():
        covered = evaluated & coverage
        wrong_covered = evaluated_wrong & coverage
        coverage_report[name] = {
            'mean_candidate_count': float(candidate_counts[name][evaluated].double().mean())
            if evaluated.any() else None,
            'max_candidate_count': int(candidate_counts[name][evaluated].max())
            if evaluated.any() else None,
            'evaluated_samples': int(evaluated.sum()),
            'covered_samples': int(covered.sum()),
            'covered_rate': float(covered.double().sum() / evaluated.sum() * 100)
            if evaluated.any() else None,
            'first_pass_wrong_samples': int(evaluated_wrong.sum()),
            'first_pass_wrong_covered_samples': int(wrong_covered.sum()),
            'first_pass_wrong_covered_rate': float(
                wrong_covered.double().sum() / evaluated_wrong.sum() * 100)
            if evaluated_wrong.any() else None,
        }
    report['ones']['candidate_coverage'] = coverage_report

    for mode, coverage_name, label in (
            ('top2_task_oracle', 'prob_top2', 'top2'),
            ('union_task_oracle', 'union_top2', 'union')):
        coverage = candidate_coverages[coverage_name]
        covered = evaluated & coverage
        wrong_covered = evaluated_wrong & coverage
        oracle_correct = predictions[mode] == targets
        report[mode].update({
            'margin_threshold': float(margin_threshold),
            'evaluated_samples': int(evaluated.sum()),
            f'true_task_in_{label}_samples': int(covered.sum()),
            f'true_task_in_{label}_rate': float(covered.double().sum() / evaluated.sum() * 100)
            if evaluated.any() else None,
            'first_pass_wrong_samples': int(evaluated_wrong.sum()),
            f'first_pass_wrong_true_task_in_{label}_samples': int(wrong_covered.sum()),
            f'first_pass_wrong_true_task_in_{label}_rate': float(
                wrong_covered.double().sum() / evaluated_wrong.sum() * 100)
            if evaluated_wrong.any() else None,
            'selected_corrected': int((covered & oracle_correct & ~baseline_correct).sum()),
            'selected_broken': int((covered & ~oracle_correct & baseline_correct).sum()),
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
