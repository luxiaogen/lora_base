"""P-only post-SGD direction screen; every diagnostic uses the actual gate.

The proposal right-preconditions the SGD displacement by a damped A Gram
inverse. It is a finite candidate search, NOT a constrained optimum or a
guarantee of old-class safety. Momentum remains the original SGD accumulator.
"""
import json
import logging
import random

import numpy as np
import torch


def projection_norms(value):
    return torch.stack([part.float().norm() for part in value.chunk(3, dim=0)])


def split_effective_step(raw_before, raw_after, gate_before, gate_after, gamma):
    fixed = gamma * gate_before * (raw_after - raw_before)
    switch = gamma * (gate_after - gate_before) * raw_after
    return fixed, switch


def gram_preconditioner(a):
    gram = a.float() @ a.float().t()
    scale = gram.diag().mean().clamp_min(1e-12)
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + .01 * scale * eye, scale * eye).to(a)


@torch.no_grad()
def propose_steps(module, before_b, after_b, task_gradient, inverse):
    """Choose norm-matched and conflict-bounded proposals on the SAME SGD step.

    Matching is per Q/K/V, on actual dynamic-gate effective increments: at
    least 98% and at most 100.01% of the original norm. Conflict comparison
    uses the union of before/reference/candidate masks for BOTH updates.
    """
    unit = module.P_lora[module.cur_task]
    a, gamma = unit.A_weight.detach(), float(module.plora_gamma)
    raw0 = before_b @ a
    safe0, gate0, mask0 = module._safe_delta(raw0, True, return_details=True)

    def evaluate(b):
        raw = b @ a
        safe, gate, mask = module._safe_delta(raw, True, return_details=True)
        effective = gamma * (safe - safe0)
        fixed, switch = split_effective_step(raw0, raw, gate0, gate, gamma)
        return dict(b=b, effective=effective, fixed=fixed, switch=switch,
                    mask=mask.bool(), gain=float(-(task_gradient * (b - before_b)).sum()))

    reference = evaluate(after_b)
    reference['mix'] = 0.0
    choices = {'norm': reference, 'conflict': reference}
    reference_norm = projection_norms(reference['effective'])
    fixed_norm = projection_norms(reference['fixed'])
    step = after_b - before_b
    proposal = step @ inverse
    for mix in (1.0, .5, .25):
        candidate_step = (1 - mix) * step + mix * proposal
        candidate_fixed = gamma * gate0 * (candidate_step @ a)
        scales = fixed_norm / projection_norms(candidate_fixed).clamp_min(1e-12)
        candidate_step = candidate_step * scales.repeat_interleave(step.shape[0] // 3)[:, None]
        candidate = evaluate(before_b + candidate_step)
        candidate['mix'] = mix
        norms = projection_norms(candidate['effective'])
        matched = bool(((norms <= reference_norm * 1.0001 + 1e-12)
                        & (norms >= reference_norm * .98 - 1e-12)).all())
        gain_floor = max(reference['gain'], 0.0) + abs(reference['gain']) * 1e-6 + 1e-12
        if not matched or candidate['gain'] <= gain_floor:
            continue
        if candidate['gain'] > choices['norm']['gain']:
            choices['norm'] = candidate
        union = mask0.bool() | reference['mask'] | candidate['mask']
        bounded = bool((projection_norms(candidate['effective'] * union)
                        <= projection_norms(reference['effective'] * union) * 1.0001 + 1e-12).all())
        if bounded and candidate['gain'] > choices['conflict']['gain']:
            choices['conflict'] = candidate
    return reference, choices, mask0.bool()


def prepare_step(modules, task_loss):
    modules = [m for m in modules if m.P_lora[m.cur_task] is not None]
    params = [m.P_lora[m.cur_task].B_weight for m in modules]
    gradients = torch.autograd.grad(task_loss, params, retain_graph=True)
    return [(m, p.detach().clone(), g.detach()) for m, p, g in zip(modules, params, gradients)]


@torch.no_grad()
def finish_step(snapshots, mode, epoch, batch, detailed, probe_records=None):
    counts = dict(layers=0, norm_accepted=0, conflict_accepted=0, applied=0)
    for module, before, gradient in snapshots:
        unit = module.P_lora[module.cur_task]
        cache = getattr(module, '_p_step_gram_cache', None)
        if cache is None or cache[0] != module.cur_task:
            cache = (module.cur_task, gram_preconditioner(unit.A_weight.detach()))
            module._p_step_gram_cache = cache
        reference, choices, mask0 = propose_steps(
            module, before, unit.B_weight.detach().clone(), gradient, cache[1])
        selected = reference if mode == 'baseline' else choices[mode]
        # Never copy the reference back: preserve baseline/fallback bit for bit.
        if selected is not reference:
            unit.B_weight.copy_(selected['b'])
        if probe_records is not None:
            probe_records.append((unit.B_weight, reference['b'], choices['norm']['b'],
                                  choices['conflict']['b'], selected['b']))
        counts['layers'] += 1
        counts['norm_accepted'] += int(choices['norm'] is not reference)
        counts['conflict_accepted'] += int(choices['conflict'] is not reference)
        counts['applied'] += int(selected is not reference)
        if not detailed:
            continue
        for name, candidate in (('reference', reference), *choices.items()):
            union = mask0 | reference['mask'] | candidate['mask']
            for j, projection in enumerate(('Q', 'K', 'V')):
                ref = reference['effective'].chunk(3)[j]
                eff = candidate['effective'].chunk(3)[j]
                mask = union.chunk(3)[j]
                before_mask, after_mask = mask0.chunk(3)[j], candidate['mask'].chunk(3)[j]
                denominator = int((before_mask | after_mask).sum())
                cosine_denom = float(ref.norm() * eff.norm())
                row = dict(task=module.cur_task, epoch=epoch + 1, batch=batch + 1,
                           layer=module.layer_idx, branch='P', projection=projection,
                           mode=mode, proposal=name, accepted=candidate is not reference,
                           mix=candidate['mix'], fixed_gate_predicted_gain=candidate['gain'],
                           reference_fixed_gate_predicted_gain=reference['gain'],
                           effective_norm=float(eff.norm()), reference_norm=float(ref.norm()),
                           conflict_norm=float((eff * mask).norm()),
                           reference_conflict_norm=float((ref * mask).norm()),
                           fixed_norm=float(candidate['fixed'].chunk(3)[j].norm()),
                           switch_norm=float(candidate['switch'].chunk(3)[j].norm()),
                           mask_count=int(after_mask.sum()), reference_mask_count=int(reference['mask'].chunk(3)[j].sum()),
                           mask_jaccard=float((before_mask & after_mask).sum()) / denominator if denominator else 1.,
                           cosine=float((eff * ref).sum()) / cosine_denom if cosine_denom else None)
                logging.info('PStepDirection %s', json.dumps(row, allow_nan=False))
    return counts


def probe_directions(network, records, inputs, targets, loss_function):
    """Actual nonlinear loss on the SAME TRAINING batch; never validation.

    S and classifier stay at the common post-SGD state. Each proposal replaces
    all P B matrices together. Restore selected B, module modes and RNG even
    when evaluation fails. Probe results never select an update.
    """
    modes = [(m, m.training) for m in network.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    metrics = {}
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            network.eval()
            for index, name in enumerate(('reference', 'norm', 'conflict'), 1):
                for record in records:
                    record[0].copy_(record[index])
                logits = network(inputs)['logits']
                positive = logits.gather(1, targets[:, None]).squeeze(1)
                negative = logits.scatter(1, targets[:, None], float('-inf')).max(dim=1).values
                metrics[name] = dict(loss=float(loss_function(logits, targets)),
                                     accuracy=float((logits.argmax(1) == targets).float().mean() * 100),
                                     margin=float((positive - negative).mean()))
    finally:
        with torch.no_grad():
            for record in records:
                record[0].copy_(record[4])
        for module, training in modes:
            module.training = training
        random.setstate(py_state)
        np.random.set_state(np_state)
    return metrics
