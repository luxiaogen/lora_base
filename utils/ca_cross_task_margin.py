"""Symmetric, class-balanced CA supervision against other-task negatives."""
import torch


def cross_task_margin(cosines, targets, class_tasks, margin):
    positive = cosines.gather(1, targets[:, None]).squeeze(1)
    other_task = class_tasks[None, :] != class_tasks[targets, None]
    negative = cosines.masked_fill(~other_task, float('-inf')).max(1).values
    violations = (margin + negative - positive).clamp_min(0.)
    return violations.mean(), (violations.detach() > 0).float().mean()
