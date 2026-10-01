"""Current-image W_pre supervision; auxiliary gradients are added to S only."""
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


def selective_feature_loss(student, teacher, ridge_weight, global_labels):
    """Known current training labels select teacher-correct images, never tests."""
    teacher = teacher.detach()
    prediction = (teacher @ ridge_weight.detach()).argmax(1)
    selected = prediction.eq(global_labels).to(student.dtype)
    distance = 1. - F.cosine_similarity(student, teacher, dim=1)
    loss = (distance * selected).sum() / selected.sum().clamp_min(1.)
    return loss, dict(wpre_feature_loss=loss.detach(), wpre_selected_ratio=selected.mean().detach())


def log_s_gradients(task_loss, params, auxiliary_grads, task, epoch, batch):
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
        scope='current_S_B_only', ce_norm=float(ce_norm), auxiliary_norm=float(aux_norm),
        ratio=float(aux_norm / ce_norm.clamp_min(1e-12)),
        cosine=float(dot / (ce_norm * aux_norm).clamp_min(1e-12))))
