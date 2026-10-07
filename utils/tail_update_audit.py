"""Read-only update telemetry at two consecutive steps; no images or feature cache."""
import json
import logging

import torch


@torch.no_grad()
def sample_updates(learner, epoch, step):
    task = learner._cur_task
    if task == 0:
        return
    if step == 0:
        learner._tail_previous_gates = {}
    previous = learner._tail_previous_gates
    device = learner._device
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    for module in learner._iter_lora_modules():
        for branch, isolated, gamma, unit in (
                ('S', False, module._shared_gamma(task), module.S_lora[task]),
                ('P', True, module.plora_gamma, module.P_lora[task])):
            if unit is None:
                continue
            delta = unit.B_weight @ unit.A_weight
            state = module._tail_update_state(delta, isolated)
            # O uses its original score/energy policy, not the soft-tail counterfactual.
            effective, gate, selected = module._safe_delta(delta, isolated, return_details=True)
            base, _ = module._merge_base_and_conflict(delta, isolated,
                module._conflict_parameters()[0], compute_conflict=False)
            threshold = float(state['threshold'])
            removed = (base - effective).float().norm()
            for projection, raw, before, actual, actual_gate, applied, reference in zip(('Q', 'K', 'V'),
                    delta.chunk(3), base.chunk(3), effective.chunk(3), gate.chunk(3), selected.chunk(3), state['selected'].chunk(3)):
                key = module.layer_idx, branch, projection
                # Fixed evenly spaced coordinates measure changes without retaining full gates.
                stride = max(1, actual_gate.numel() // 2048)
                sample = actual_gate.flatten()[::stride][:2048].float()
                old = previous.get(key)
                change = float((sample - old).abs().mean()) if old is not None else None
                switch = float(((sample < 1) != (old < 1)).float().mean()) if old is not None else None
                if step == 0:
                    previous[key] = sample.clone()
                logging.info('TailUpdate %s', json.dumps(dict(task=task, epoch=epoch, step=step,
                    layer=module.layer_idx, branch=branch, projection=projection,
                    rule=learner.args.get('dual_mask_update_rule', 'step'), threshold=threshold,
                    raw_norm=float((gamma * raw).float().norm()), base_norm=float((gamma * before).float().norm()),
                    effective_norm=float((gamma * actual).float().norm()),
                    removed_norm=float((gamma * (before - actual)).float().norm()),
                    selected_density=float(reference.float().mean()), applied_density=float(applied.float().mean()),
                    changed_nonzero_density=float(((before != actual) & (before != 0)).float().mean()),
                    layer_target_removed_norm=float(abs(gamma) * state['target_removed_norm']),
                    layer_actual_removed_norm=float(abs(gamma) * removed),
                    layer_match_residual=float(abs(gamma) * (removed - state['target_removed_norm']).abs()),
                    coefficient=float(state['coefficient']), gate_sample_change=change,
                    gate_sample_switch=switch, gate_sample_count=sample.numel(), peak_allocated_bytes=peak)))
    if step == 1:
        previous.clear()
