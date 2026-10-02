"""Post-training gate interventions on fixed test inputs, never used by training.

No old inputs, logits, features or weights are retained after this report.
This diagnoses gate choices at one trained state, not optimizer-step causality.
"""
import hashlib
import json
import logging
import random

import numpy as np
import torch

from utils.p_functional_score import gram_risk, qk_risk
from utils.stage_audit import stage_metrics


MODES = ('coordinate', 'wpre_product', 'wpre_input', 'wpre_output', 'wpre_qk', 'task_qk')


def diagnostic_indices(labels, known_classes, per_partition):
    labels = np.asarray(labels)
    result = []
    for old in (True, False):
        classes = np.unique(labels[(labels < known_classes) == old])
        if len(classes) > per_partition:
            classes = classes[np.linspace(0, len(classes) - 1, per_partition, dtype=int)]
        buckets = [np.flatnonzero(labels == value).tolist() for value in classes]
        order = [bucket[i] for i in range(max(map(len, buckets), default=0))
                 for bucket in buckets if i < len(bucket)]
        result.extend(order[:per_partition])
    return result


@torch.no_grad()
def layer_proxy_metrics(layer, state):
    base = state['base'].float()
    safe = base * state['gate']
    anchor = layer.pretrained_weight.detach().float().reshape_as(base)
    start = layer.qkv.weight.detach().float().reshape_as(base)
    s = layer.S_lora[layer.cur_task]
    context = layer._safe_delta(layer.slora_gamma * (s.B_weight @ s.A_weight), False).reshape_as(base).float()
    wpre_context = start - anchor + context
    bias = None if layer.qkv.bias is None else layer.qkv.bias.detach().float().reshape(3, layer.dim)
    row = dict(layer=int(layer.layer_idx),
        selected_k=state['selected'].sum((-2, -1)).tolist(),
        reference_k=state['reference'].sum((-2, -1)).tolist(),
        strength=state['strengths'].tolist(),
        reference_removed_norm=(base * state['reference'] * state['reference_strength']).norm(dim=(-2, -1)).tolist(),
        actual_removed_norm=(base - safe).norm(dim=(-2, -1)).tolist())
    for name, delta in (('ungated', base), ('gated', safe)):
        row[name] = dict(
            wpre_input=gram_risk(anchor, delta, wpre_context, 'input')[0].mean().item(),
            wpre_output=gram_risk(anchor, delta, wpre_context, 'output')[0].mean().item(),
            wpre_qk=qk_risk(anchor, delta, wpre_context, layer.num_heads, bias)[0].item(),
            task_qk=qk_risk(start, delta, context, layer.num_heads, bias)[0].item())
    return row


def report_score_counterfactuals(network, dataset, device, task, known_classes, per_partition=64):
    layers = [module for module in network.modules() if hasattr(module, '_p_direction_gate')]
    modes = [(module, module.training) for module in network.modules()]
    saved = [(layer, layer.args.get('p_direction_score', 'off'), layer._p_direction_state,
              layer._p_score_diagnostic_gate) for layer in layers]
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    rows = []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            network.eval()
            indices = diagnostic_indices(dataset.labels, known_classes, per_partition)
            samples = [dataset[index] for index in indices]
            targets = torch.tensor([int(sample[2]) for sample in samples])
            inputs = torch.stack([sample[1] for sample in samples])
            digest = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes() + targets.numpy().tobytes()).hexdigest()
            reference_logits = torch.cat([network.interface(batch.to(device)).float().cpu()
                                          for batch in inputs.split(32)])
            for mode in MODES:
                telemetry = []
                for layer in layers:
                    layer._p_score_diagnostic_gate = None
                    layer.args['p_direction_score'] = mode
                    p = layer.P_lora[task]
                    raw = layer.plora_gamma * (p.B_weight @ p.A_weight)
                    gate, selected = layer._p_direction_gate(raw)
                    telemetry.append(layer_proxy_metrics(layer, layer._p_direction_state))
                    layer._p_score_diagnostic_gate = (gate, selected)
                logits = torch.cat([network.interface(batch.to(device)).float().cpu()
                                    for batch in inputs.split(32)])
                errors = [abs(a - b) / max(b, 1e-12) for layer in telemetry
                          for a, b in zip(layer['actual_removed_norm'], layer['reference_removed_norm'])]
                metrics = stage_metrics(logits, targets, known_classes, reference_logits)
                metrics = {group: {key: values[key] for key in ('n', 'accuracy', 'margin', 'corrected', 'broken')}
                           for group, values in metrics.items()}
                row = dict(task=int(task), trained_mode=saved[0][1], applied_mode=mode,
                    reference_mode=saved[0][1],
                    source='test_report_only_fixed_weights', sample_sha256=digest,
                    metrics=metrics,
                    max_relative_removed_norm_error=max(errors, default=0), layers=telemetry)
                logging.info('PScoreCounterfactual %s', json.dumps(row, allow_nan=False))
                rows.append(row)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        for layer, mode, state, override in saved:
            layer.args['p_direction_score'] = mode
            layer._p_direction_state = state
            layer._p_score_diagnostic_gate = override
        for module, training in modes:
            module.training = training
    return rows
