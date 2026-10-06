"""Read-only position diagnostics and measured costs; no training data buffer."""
from contextlib import contextmanager
import json
import logging
import random
import resource
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


@contextmanager
def stage_cost(learner, stage):
    enabled = learner.args.get('dual_mask_mechanism_audit', False)
    if not enabled:
        yield
        return
    device = learner._device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    before_diagnostic = getattr(learner, '_core_diagnostic_seconds', 0.0)
    started = time.perf_counter()
    try:
        yield
    finally:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        if stage in ('mechanism_diagnostic', 'update_telemetry', 'release_probe'):
            learner._core_diagnostic_seconds = before_diagnostic + seconds
        excluded = (getattr(learner, '_core_diagnostic_seconds', 0.0) - before_diagnostic
                    if stage == 'training' else 0.0)
        logging.info('CoreCost %s', json.dumps(dict(task=learner._cur_task, stage=stage,
            seconds=seconds - excluded, wall_seconds=seconds, excluded_diagnostic_seconds=excluded,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
            cpu_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                * (1 if sys.platform == 'darwin' else 1024))))


def round_robin_indices(labels, classes, limit):
    pools = [np.flatnonzero(np.asarray(labels) == category).tolist() for category in classes]
    result = []
    depth = 0
    while len(result) < limit:
        current = [pool[depth] for pool in pools if depth < len(pool)]
        if not current:
            break
        result.extend(current[:limit - len(result)])
        depth += 1
    return result


@contextmanager
def diagnostic_state(network, modules, device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in network.modules()]
    positions = [module._core_audit_position for module in modules]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            network.eval()
            with torch.no_grad():
                yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training
        for module, position in zip(modules, positions):
            module._core_audit_position = position


def _logits_stats(logits, labels):
    positive = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], float('-inf'))
    return logits.argmax(1), positive - competitors.max(1).values


def position_diagnostic(learner, dataset, epoch):
    if learner._cur_task not in (1, 5, 9) or epoch not in (1, 5, 10, 20):
        return
    network = learner._network.module if isinstance(learner._network, torch.nn.DataParallel) else learner._network
    modules = list(learner._iter_lora_modules())
    indices = round_robin_indices(dataset.labels, range(learner._known_classes), 512)
    indices += round_robin_indices(dataset.labels, range(learner._known_classes, learner._total_classes), 128)
    loader = DataLoader(Subset(dataset, indices), batch_size=learner.batch_size, shuffle=False,
                        num_workers=0, generator=torch.Generator().manual_seed(1993))
    with stage_cost(learner, 'mechanism_diagnostic'), diagnostic_state(network, modules, learner._device):
        totals = {name: dict(count=0, wpre_correct=0, permuted_correct=0, changed=0,
                            wpre_margin=0.0, permuted_margin=0.0) for name in ('old', 'new')}
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(learner._device), targets.to(learner._device)
            values = []
            for position in ('wpre', 'permuted'):
                for module in modules:
                    module._core_audit_position = position
                values.append(_logits_stats(network.interface(inputs), targets))
            for name, selected in (('old', targets < learner._known_classes),
                                   ('new', targets >= learner._known_classes)):
                count = int(selected.sum())
                first, second = values
                row = totals[name]
                row['count'] += count
                row['wpre_correct'] += int((first[0][selected] == targets[selected]).sum())
                row['permuted_correct'] += int((second[0][selected] == targets[selected]).sum())
                row['changed'] += int((first[0][selected] != second[0][selected]).sum())
                row['wpre_margin'] += float(first[1][selected].sum())
                row['permuted_margin'] += float(second[1][selected].sum())
        for name, row in totals.items():
            count = row['count']
            if count:
                row.update(wpre_accuracy=100 * row['wpre_correct'] / count,
                           permuted_accuracy=100 * row['permuted_correct'] / count,
                           wpre_margin=row['wpre_margin'] / count,
                           permuted_margin=row['permuted_margin'] / count)
            logging.info('CorePositionDiagnostic %s', json.dumps(dict(task=learner._cur_task,
                epoch=epoch, partition=name, norm_control='same_state_paired_min',
                scope='all_seen', source='test_only_fixed_order', **row)))


@torch.no_grad()
def epoch_updates(learner, epoch):
    from utils.protect_position import update_rows
    for module in learner._iter_lora_modules():
        task = learner._cur_task
        mode = learner.args.get('dual_mask_permission_mode', 'asymmetric')
        for branch, isolated, gamma, unit in (
                ('Single' if module._single_branch_active(task) else 'S', False,
                 module._shared_gamma(task), module.S_lora[task]),
                ('P', True, module.plora_gamma, module.P_lora[task])):
            if unit is None or (task == 0 and isolated):
                continue
            delta = unit.B_weight @ unit.A_weight
            effective, _, selected = module._safe_delta(delta, isolated, return_details=True)
            if task == 0:
                base = delta
            else:
                base, _ = module._merge_base_and_conflict(delta, isolated,
                    module._conflict_parameters()[0], compute_conflict=False)
            protect = module._p_protect_mask() if isolated else module.general_mask
            release = module.p_permission_release_mask if task > 0 and isolated else None
            allowed = None if release is None else ((1 - protect).bool() | release.bool())
            relative = task > 0 and module.dual_mask_conflict_score_mode in ('wpre_relative', 'task_relative')
            if relative:
                from models.attention import _exact_top_ratio_mask
                magnitude_mask = _exact_top_ratio_mask(delta.detach().abs(), module.dual_mask_conflict_ratio).bool()
            for row in update_rows(gamma * delta, gamma * base, gamma * effective,
                    selected, protect, allowed_mask=allowed, task=task, epoch=epoch,
                    layer=module.layer_idx, branch=branch, permission_mode=mode,
                    position_norm_match=learner.args.get('dual_mask_position_norm_match', 'off') if task else 'off',
                    p_permission_norm_match=learner.args.get('p_permission_norm_match', True) if release is not None else False,
                    position=(learner.args.get('p_permission_position', 'wpre') if isolated
                        and module.p_permission_protect is not None else learner.args.get('dual_mask_protect_position', 'wpre')) if task else 'wpre'):
                if relative:
                    index = ('Q', 'K', 'V').index(row['projection'])
                    first, second = selected.bool().chunk(3)[index], magnitude_mask.chunk(3)[index]
                    union = int((first | second).sum())
                    scale = module.relative_conflict_scale.chunk(3)[index]
                    row.update(score_mode=module.dual_mask_conflict_score_mode,
                               magnitude_mask_jaccard=int((first & second).sum()) / union if union else 1.0,
                               reference_row_rms_min=float(scale.min()), reference_row_rms_max=float(scale.max()))
                logging.info('CoreEpochUpdate %s', json.dumps(row))
            if task > 0 and isolated and learner.args.get('p_permission_release', 'off') != 'off':
                protect = module._p_protect_mask().to(delta)
                release = module.p_permission_release_mask
                if release is None:
                    release = torch.zeros_like(protect)
                reference = delta * (1 - protect) * (1 - module._conflict_parameters()[1] * selected)
                for projection, raw_part, ref, actual, released, protected in zip(('Q', 'K', 'V'),
                        delta.chunk(3), reference.chunk(3), effective.chunk(3), release.chunk(3), protect.chunk(3)):
                    logging.info('PPermissionUpdate %s', json.dumps(dict(task=task, epoch=epoch,
                        layer=module.layer_idx, projection=projection, released=int(released.sum()),
                        raw_norm=float(gamma * raw_part.norm()), reference_norm=float(gamma * ref.norm()),
                        effective_norm=float(gamma * actual.norm()),
                        protected_effective_norm=float(gamma * (actual * protected).norm()),
                        norm_match=learner.args.get('p_permission_norm_match', True))))


def storage_bytes(learner, stage):
    groups, seen = {}, set()
    def add(name, value):
        if torch.is_tensor(value):
            key = (str(value.device), value.data_ptr())
            if key not in seen:
                seen.add(key)
                groups[name] = groups.get(name, 0) + value.numel() * value.element_size()
        elif isinstance(value, dict):
            for item in value.values():
                add(name, item)
    for name, value in list(learner._network.named_parameters()) + list(learner._network.named_buffers()):
        if 'pretrained_weight' in name:
            group = 'wpre_qkv'
        elif 'lora.' in name.lower():
            group = 'temporary_adapters'
        elif 'classifier_pool' in name:
            group = 'classifier'
        elif any(key in name for key in ('importance', 'mask', 'core_reference', 'core_permuted', 'p_permission_protect')):
            group = 'importance_and_masks'
        else:
            group = 'encoder_and_other_buffers'
        add(group, value)
    for group, attr in (('ca_means', '_class_means'), ('ca_covariances', '_class_covs'),
                        ('wpre_class_prototypes', '_w0_class_means')):
        add(group, getattr(learner, attr, None))
    activity = {}
    if 'dual_mask_branch_layout' in learner.args:
        units = [unit for module in learner._iter_lora_modules()
                 for unit in (module.S_lora[learner._cur_task],
                              module.P_lora[learner._cur_task] if learner._cur_task else None)
                 if unit is not None]
        activity = dict(branch_layout=learner.args['dual_mask_branch_layout'],
            active_adapter_factor_parameters=sum(p.numel() for unit in units for p in unit.parameters()),
            trainable_adapter_parameters=sum(p.numel() for unit in units for p in unit.parameters() if p.requires_grad),
            trainable_network_parameters=sum(p.numel() for p in learner._network.parameters() if p.requires_grad))
    logging.info('CoreStorage %s', json.dumps(dict(task=learner._cur_task, stage=stage,
        groups_bytes=groups, total_tensor_bytes=sum(groups.values()),
        definition='unique live tensor views; excludes optimizer and transient workspaces', **activity)))
