"""Current-image W_pre supervision with branch-scoped auxiliary gradients."""
import logging
import random

import numpy as np
import torch
from torch.nn import functional as F


def teacher_features(network, inputs, anchor_context):
    """Run before the student forward; preserve its modes and random trajectory."""
    modes = [(module, module.training) for module in network.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [inputs.device.index] if inputs.is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices), anchor_context, torch.no_grad():
            network.eval()
            return network(inputs)['features'].detach()
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def all_seen_logits(features, heads):
    """Use the same normalized all-seen readout as ordinary inference."""
    with torch.no_grad():
        return torch.cat([F.linear(F.normalize(features.detach(), dim=-1),
                                   F.normalize(head.weight.detach(), dim=-1))
                          for head in heads], dim=1)


def selective_feature_loss(student, teacher, ridge_weight, global_labels,
                           selection='teacher_correct', student_logits=None,
                           normalization='selected', generator=None):
    """Known current training labels select teacher-correct images, never tests."""
    teacher = teacher.detach()
    prediction = (teacher @ ridge_weight.detach()).argmax(1)
    teacher_correct = prediction.eq(global_labels)
    rescue = (teacher_correct & student_logits.detach().argmax(1).ne(global_labels)
              if student_logits is not None else teacher_correct)
    selected = teacher_correct
    if selection == 'complement':
        selected = rescue
    elif selection == 'random_matched':
        # Use a private CPU generator, not the training/augmentation RNG stream.
        candidates = teacher_correct.nonzero(as_tuple=True)[0]
        order = torch.randperm(len(candidates), generator=generator)
        chosen = candidates[order[:int(rescue.sum())].to(candidates.device)]
        selected = torch.zeros_like(teacher_correct)
        selected[chosen] = True
    selected = selected.to(student.dtype)
    distance = 1. - F.cosine_similarity(student, teacher, dim=1)
    denominator = len(student) if normalization == 'batch' else selected.sum().clamp_min(1.)
    loss = (distance * selected).sum() / denominator
    metrics = dict(wpre_feature_loss=loss.detach(), wpre_selected_ratio=selected.mean().detach(),
                   wpre_teacher_correct_ratio=teacher_correct.float().mean().detach())
    if student_logits is not None:
        metrics['wpre_rescue_ratio'] = rescue.float().mean().detach()
    return loss, metrics


def log_s_gradients(task_loss, params, auxiliary_grads, task, epoch, batch, scope='s'):
    ordinary = torch.autograd.grad(task_loss, params, retain_graph=True, allow_unused=True)
    zero = task_loss.detach().new_zeros(())
    ce_sq, aux_sq, dot = zero.clone(), zero.clone(), zero.clone()
    for ce, aux in zip(ordinary, auxiliary_grads):
        if ce is not None:
            ce_sq += ce.detach().float().square().sum()
        if aux is not None:
            aux_sq += aux.detach().float().square().sum()
        if ce is not None and aux is not None:
            dot += (ce.detach().float() * aux.detach().float()).sum()
    ce_norm, aux_norm = ce_sq.sqrt(), aux_sq.sqrt()
    logging.info('WpreDistillGrad %s', dict(task=task, epoch=epoch + 1, batch=batch + 1,
        scope=f'current_{scope.upper()}_B_only', ce_norm=float(ce_norm), auxiliary_norm=float(aux_norm),
        ratio=float(aux_norm / ce_norm.clamp_min(1e-12)),
        cosine=float(dot / (ce_norm * aux_norm).clamp_min(1e-12))))
