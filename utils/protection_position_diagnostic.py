"""旧训练来源的只读位置诊断；不提供正式训练权限或新增损失。"""
from contextlib import contextmanager
import copy
import hashlib
import json
import logging
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from utils.protect_position import permute_protect_mask
from utils.task_start_prior import count_matched_mask, correlation, jaccard


POSITIONS = ('W', 'WR', 'WRperm', 'MC', 'Random')
TRANSIENT = ('_pending_safe_residual_deltas', '_last_safe_residual_loss',
             '_composed_forward_gate', '_p_direction_state', '_core_audit_position',
             '_prototype_audit_position', '_position_signal_s_override')


def split_indices(labels, known_classes, total_classes):
    """每个旧类前两张评分、接下来两张评估；新类只取两张评估。"""
    labels = np.asarray(labels)
    pools = [np.flatnonzero(labels == label).tolist() for label in range(total_classes)]
    def select(classes, start, end):
        return [pools[c][depth] for depth in range(start, end) for c in classes
                if depth < len(pools[c])]
    return dict(score=select(range(known_classes), 0, 2),
                old=select(range(known_classes), 2, 4),
                new=select(range(known_classes, total_classes), 0, 2))


@contextmanager
def preserve_state(network, modules):
    """求导不写 .grad；仍恢复缓冲、原梯度对象、模式及全部训练 RNG。"""
    modes = [(m, m.training) for m in network.modules()]
    parameters = [(p, p.requires_grad, p.grad,
                   p.grad.detach().clone() if p.grad is not None else None) for p in network.parameters()]
    buffers = [(b, b.detach().clone()) for b in network.buffers()]
    attributes = [(m, {n: (hasattr(m, n), copy.copy(getattr(m, n, None))) for n in TRANSIENT}) for m in modules]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            network.eval()
            yield
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
            for parameter, flag, gradient, original in parameters:
                parameter.requires_grad_(flag)
                parameter.grad = gradient
                if gradient is not None:
                    gradient.copy_(original)
        for module, training in modes:
            module.training = training
        for module, values in attributes:
            for name, (existed, value) in values.items():
                if existed:
                    setattr(module, name, value)
                elif hasattr(module, name):
                    delattr(module, name)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def sampled_fisher(network, modules, dataset, indices, known_classes, scale, seed):
    """固定旧类概率模型；两次标签采样分别累计逐样本梯度平方。"""
    weights = [module.qkv.weight for module in modules]
    scores = [[torch.zeros_like(weight, dtype=torch.float64 if weight.dtype == torch.float64 else torch.float32)
               for weight in weights] for _ in range(2)]
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    with preserve_state(network, modules), torch.enable_grad():
        for weight in weights:
            weight.requires_grad_(True)
        for number, index in enumerate(indices, 1):
            _, image, _ = dataset[index]
            logits = network.interface(image.unsqueeze(0).to(weights[0].device))[:, :known_classes]
            log_probability = F.log_softmax(scale * logits, dim=1)
            labels = torch.multinomial(log_probability.detach().exp().cpu()[0], 2,
                                       replacement=True, generator=generator)
            for draw, label in enumerate(labels):
                loss = -log_probability[0, int(label)]
                gradients = torch.autograd.grad(loss, weights, retain_graph=draw == 0)
                for score, gradient in zip(scores[draw], gradients):
                    score.add_(gradient.detach().to(score).square())
            if number % 32 == 0 or number == len(indices):
                logging.info('PositionSignalFisherProgress samples=%s/%s', number, len(indices))
    for draw in scores:
        for score in draw:
            score.div_(max(len(indices), 1))
    return scores


@torch.no_grad()
def position_masks(importance, residual, fisher, reference, seed, layer):
    from models.attention import _normalize_score
    pre = _normalize_score(importance)
    history = _normalize_score(residual.detach().abs())
    return dict(W=reference.bool().clone(),
        WR=count_matched_mask(pre + history, reference).bool(),
        WRperm=count_matched_mask(pre + permute_protect_mask(history, seed, layer), reference).bool(),
        MC=count_matched_mask(fisher, reference).bool(),
        Random=permute_protect_mask(reference, seed, layer).bool())


@torch.no_grad()
def matched_s_updates(value, masks, strength):
    updates = {name: value * (1 - strength * mask.to(value)) for name, mask in masks.items()}
    norms = {name: update.float().norm() for name, update in updates.items()}
    target = torch.stack(list(norms.values())).min()
    scales = {name: torch.where(norm > 0, target / norm.clamp_min(1e-30), torch.ones_like(norm))
              for name, norm in norms.items()}
    return ({name: update * scales[name].to(value) for name, update in updates.items()}, scales)


def _emit(marker, **row):
    logging.info('%s %s', marker, json.dumps(row))


def _network(learner):
    return learner._network.module if isinstance(learner._network, torch.nn.DataParallel) else learner._network


def prepare_positions(learner, dataset):
    """只缓存当前任务的评分和候选掩码；不改 general_mask/P 权限。"""
    started = time.perf_counter()
    network = _network(learner)
    modules = list(learner._iter_lora_modules())
    indices = split_indices(dataset.labels, learner._known_classes, learner._total_classes)
    seed = learner.args['seed']
    seed = int(seed[0] if isinstance(seed, (list, tuple)) else seed)
    fisher_started = time.perf_counter()
    draws = sampled_fisher(network, modules, dataset, indices['score'],
                           learner._known_classes, learner.scale, seed + 15485863 * learner._cur_task)
    fisher_seconds = time.perf_counter() - fisher_started
    layers = []
    for layer, (module, first, second) in enumerate(zip(modules, *draws)):
        fisher = (first + second) * .5
        residual = module.qkv.weight.detach() - module.pretrained_weight
        reference = module.general_mask.bool()
        masks = position_masks(module.w0_importance, residual, fisher, reference, seed, layer)
        _emit('PositionSignalFisher', task=learner._cur_task, layer=layer,
              samples=len(indices['score']), draws_per_image=2, scale=learner.scale,
              classifier_scope='previous_seen_classes', source='old_train_score_only',
              s_b_norm=float(module.S_lora[learner._cur_task].B_weight.detach().norm()),
              p_b_norm=float(module.P_lora[learner._cur_task].B_weight.detach().norm()),
              draw_correlation=correlation(first, second),
              draw_mask_jaccard=jaccard(count_matched_mask(first, reference), count_matched_mask(second, reference)),
              fisher_sum=float(fisher.double().sum()), residual_norm=float(residual.float().norm()))
        for name, mask in masks.items():
            _emit('PositionSignalMask', task=learner._cur_task, layer=layer, position=name,
                  qkv_counts=[int(x.sum()) for x in mask.chunk(3)],
                  reference_counts=[int(x.sum()) for x in reference.chunk(3)],
                  reference_jaccard=jaccard(reference, mask),
                  fisher_mass=float((fisher * mask).double().sum() / fisher.double().sum().clamp_min(1e-30)),
                  residual_mass=float((residual.abs() * mask).double().sum() / residual.abs().double().sum().clamp_min(1e-30)))
        from models.attention import _normalize_score
        for name, score in (('spectrum', _normalize_score(module.w0_importance)),
                            ('residual', _normalize_score(residual.abs()))):
            _emit('PositionSignalScore', task=learner._cur_task, layer=layer, source=name,
                  mass=float(score.double().sum()), quantiles=torch.quantile(score.float().flatten(),
                      torch.tensor([0., .25, .5, .75, .9, 1.], device=score.device)).tolist(),
                  fisher_correlation=correlation(score, fisher))
        layers.append(dict(fisher=fisher.cpu(), masks={n: m.cpu() for n, m in masks.items()}))
    learner._position_signal_context = dict(dataset=dataset, indices=indices, layers=layers)
    _emit('PositionSignalPrepared', task=learner._cur_task, source='train_only_no_training_effect',
          sample_counts={k: len(v) for k, v in indices.items()},
          classes_covered={k: sorted(set(int(dataset.labels[i]) for i in v)) for k, v in indices.items()},
          index_hashes={k: hashlib.sha256(np.asarray(v, dtype=np.int64).tobytes()).hexdigest() for k, v in indices.items()},
          score_old_disjoint=not bool(set(indices['score']) & set(indices['old'])),
          transform=repr(getattr(dataset, 'trsf', None)),
          fisher_seconds=fisher_seconds, seconds=time.perf_counter() - started,
          cache_bytes=sum(r['fisher'].numel() * r['fisher'].element_size() +
              sum(m.numel() * m.element_size() for m in r['masks'].values()) for r in layers))


@torch.no_grad()
def report_positions(learner, epoch):
    """同一状态只替换 S；P及正式训练掩码不动。"""
    started = time.perf_counter()
    context = learner._position_signal_context
    network = _network(learner)
    modules = list(learner._iter_lora_modules())
    with preserve_state(network, modules):
        overrides = {name: [] for name in POSITIONS}
        for layer, (module, record) in enumerate(zip(modules, context['layers'])):
            raw = module.S_lora[learner._cur_task].B_weight @ module.S_lora[learner._cur_task].A_weight
            _, _, selected = module._safe_delta(raw, False, return_details=True)
            beta = module._conflict_parameters()[1]
            value = raw * (1 - beta * selected)
            masks = {n: m.to(raw.device) for n, m in record['masks'].items()}
            updates, scales = matched_s_updates(value, masks, module.effective_protect_strength)
            gamma = module._shared_gamma(learner._cur_task)
            private = module.P_lora[learner._cur_task]
            p = module.plora_gamma * module._safe_delta(private.B_weight @ private.A_weight, True)
            risk = record['fisher'].to(value) * (gamma * value).square()
            risk_mask = count_matched_mask(risk, masks['W'])
            for name, update in updates.items():
                overrides[name].append(update)
                effective = gamma * update
                mass = float((risk * masks[name]).double().sum() / risk.double().sum().clamp_min(1e-30))
                density = float(masks[name].float().mean())
                _emit('PositionSignalUpdate', task=learner._cur_task, epoch=epoch, layer=layer,
                      position=name, alpha=module.effective_protect_strength, beta=float(beta),
                      scale=float(scales[name]), s_raw_norm=float((gamma * raw).float().norm()),
                      s_pre_protect_norm=float((gamma * value).float().norm()),
                      s_effective_norm=float(effective.float().norm()),
                      s_removed_norm=float((gamma * raw - effective).float().norm()),
                      p_effective_norm=float(p.float().norm()), net_effective_norm=float((effective + p).float().norm()),
                      sp_inner_product=float((effective.double() * p.double()).sum()),
                      risk_mass=mass, risk_mask_jaccard=jaccard(risk_mask, masks[name]),
                      risk_enrichment=mass / density if density else None)
        indices = context['indices']['old'] + context['indices']['new']
        loader = DataLoader(Subset(context['dataset'], indices), batch_size=learner.batch_size,
                            shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(1993))
        totals = {(name, part): dict(count=0, correct=0, loss=0., margin=0., changed=0, improved=0, worsened=0)
                  for name in POSITIONS + ('ActualF',) for part in ('old', 'new')}
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(learner._device), targets.to(learner._device)
            reference = None
            for name in POSITIONS + ('ActualF',):
                for index, module in enumerate(modules):
                    module._position_signal_s_override = None if name == 'ActualF' else overrides[name][index]
                logits = network.interface(inputs)
                prediction = logits.argmax(1)
                correct = prediction.eq(targets)
                if reference is None:
                    reference = prediction, correct
                positive = logits.gather(1, targets[:, None]).squeeze(1)
                competitors = logits.clone()
                competitors.scatter_(1, targets[:, None], -torch.inf)
                margin = positive - competitors.max(1).values
                losses = F.cross_entropy(learner.scale * logits, targets, reduction='none')
                for part, selected in (('old', targets < learner._known_classes), ('new', targets >= learner._known_classes)):
                    row = totals[name, part]
                    row['count'] += int(selected.sum())
                    row['correct'] += int(correct[selected].sum())
                    row['loss'] += float(losses[selected].sum())
                    row['margin'] += float(margin[selected].sum())
                    row['changed'] += int((prediction[selected] != reference[0][selected]).sum())
                    row['improved'] += int((correct & ~reference[1])[selected].sum())
                    row['worsened'] += int((~correct & reference[1])[selected].sum())
        for (name, part), row in totals.items():
            count = row['count']
            _emit('PositionSignalIntervention', task=learner._cur_task, epoch=epoch, position=name,
                  partition=part, source='independent_train_diagnostic', classifier_scope='all_seen',
                  prediction_reference='W_matched',
                  norm_control='actual_unmatched' if name == 'ActualF' else 's_only_five_layer_min',
                  count=count, accuracy=100 * row['correct'] / count if count else None,
                  loss=row['loss'] / count if count else None, margin=row['margin'] / count if count else None,
                  changed=row['changed'], improved=row['improved'], worsened=row['worsened'])
    _emit('PositionSignalCost', task=learner._cur_task, epoch=epoch, seconds=time.perf_counter() - started)
