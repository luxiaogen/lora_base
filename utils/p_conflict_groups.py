"""Post-merge, read-only ablations of each layer's Q/K/V P-conflict component.

Positive margin change means removing the group helps on the measured samples.
Test labels are used for reporting only, never to select gates or predictions.
"""
from contextlib import contextmanager

import numpy as np
import torch

from utils.p_region_diagnostic import collect_logits, describe, preserve_diagnostic_state, summarize


def balanced_indices(labels, per_class=4):
    labels = np.asarray(labels)
    selected = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        positions = np.linspace(0, len(indices)-1, min(per_class, len(indices)), dtype=int)
        selected.extend(indices[positions].tolist())
    return selected


@contextmanager
def remove_projection(module, projection):
    weight = module.qkv.weight
    width = weight.shape[0] // 3
    rows = slice(projection * width, (projection + 1) * width)
    original = weight[rows].detach().clone()
    try:
        with torch.no_grad():
            weight[rows].sub_(module._p_conflict_component[rows].to(weight))
        yield
    finally:
        with torch.no_grad():
            weight[rows].copy_(original)


def class_margin(logits, labels):
    wrong = logits.clone()
    wrong.scatter_(1, labels[:, None], -torch.inf)
    return logits.gather(1, labels[:, None]).squeeze(1) - wrong.max(1).values


def compare_groups(network, modules, loader, device, task_sizes):
    with preserve_diagnostic_state(network):
        base, labels, indices = collect_logits(network, loader, device)
        base_margin = class_margin(base, labels)
        report = {'baseline': summarize(base, base, labels, task_sizes),
                  'sample_indices': indices.tolist(), 'groups': []}
        for layer, module in enumerate(modules):
            component = getattr(module, '_p_conflict_component', None)
            if component is None:
                continue
            for projection, name in enumerate(('Q', 'K', 'V')):
                part = component.chunk(3, dim=0)[projection]
                norm = part.norm().item()
                if norm == 0:
                    candidate = base
                else:
                    with remove_projection(module, projection):
                        candidate, other_labels, other_indices = collect_logits(network, loader, device)
                    if not torch.equal(labels, other_labels) or not torch.equal(indices, other_indices):
                        raise ValueError('Group diagnostic requires deterministic sample order')
                metrics = summarize(base, candidate, labels, task_sizes)
                change = class_margin(candidate, labels) - base_margin
                old = labels < sum(task_sizes[:-1])
                for key, selected in (('all', torch.ones_like(old)), ('old', old), ('new', ~old)):
                    metrics[key]['global_margin_change'] = describe(change[selected])
                report['groups'].append({'layer': layer, 'projection': name, 'removed_norm': norm,
                                         'nonzero_coordinates': int(part.count_nonzero()), 'metrics': metrics})
        restored, restored_labels, restored_indices = collect_logits(network, loader, device)
        if not torch.equal(labels, restored_labels) or not torch.equal(indices, restored_indices):
            raise ValueError('Group diagnostic sample order changed during restoration check')
        diff = (restored - base).abs().max().item()
        report['restored_max_abs_logit_diff'] = diff
        if not torch.allclose(restored, base, atol=1e-6, rtol=0):
            raise RuntimeError(f'Group diagnostic changed baseline logits: {diff}')
        return report
