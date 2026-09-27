"""Old-class output distillation on current-task images only."""
import copy

import torch
from torch.nn import functional as F


def frozen_teacher(network):
    teacher = copy.deepcopy(network)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def old_output_loss(features, old_weights, teacher_cosines, scale, temperature):
    student = F.linear(F.normalize(features, dim=1),
                       F.normalize(old_weights.detach(), dim=1))
    log_student = F.log_softmax(student.float() * scale / temperature, dim=1)
    teacher = F.softmax(teacher_cosines.detach().float() * scale / temperature, dim=1)
    return F.kl_div(log_student, teacher, reduction='batchmean') * temperature ** 2
