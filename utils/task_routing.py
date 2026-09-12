"""Task-ID-free decisions derived from existing classifier logits."""

import torch


def predict_with_task_evidence(logits, task_sizes, mode="global"):
    """Return global class and inferred task IDs without using true task IDs."""
    mode = str(mode).lower()
    valid_modes = {"global", "centered_max", "top1_top2"}
    if mode not in valid_modes:
        raise ValueError(
            "classification_inference_mode must be one of "
            f"{sorted(valid_modes)}, got {mode!r}"
        )

    sizes = tuple(int(size) for size in task_sizes)
    if sum(sizes) != logits.shape[1]:
        raise ValueError("task_sizes must cover every logit column")
    class_tasks = torch.arange(len(sizes), device=logits.device).repeat_interleave(
        torch.tensor(sizes, device=logits.device)
    )

    if mode == "global":
        classes = logits.argmax(dim=1)
        return classes, class_tasks[classes]

    task_scores = []
    task_classes = []
    for task_logits in torch.split(logits, sizes, dim=1):
        top_values, top_classes = task_logits.max(dim=1)
        task_classes.append(top_classes)
        if mode == "centered_max":
            if task_logits.shape[1] < 2:
                raise ValueError("centered_max requires at least two classes per task")
            rest_mean = (task_logits.sum(dim=1) - top_values) / (task_logits.shape[1] - 1)
            task_scores.append(top_values - rest_mean)
        else:
            if task_logits.shape[1] < 2:
                raise ValueError("top1_top2 requires at least two classes per task")
            top_two = task_logits.topk(k=2, dim=1).values
            task_scores.append(top_two[:, 0] - top_two[:, 1])

    tasks = torch.stack(task_scores, dim=1).argmax(dim=1)
    local_classes = torch.stack(task_classes, dim=1).gather(1, tasks[:, None]).squeeze(1)
    offsets = torch.tensor((0, *torch.tensor(sizes).cumsum(dim=0)[:-1].tolist()), device=logits.device)
    return local_classes + offsets[tasks], tasks
