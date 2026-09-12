"""Task-ID-free decisions derived from existing classifier logits."""

import torch


def raw_task_score_components(logits, targets, task_sizes):
    """Collect raw per-head maxima using labels only for test-time analysis."""
    sizes = tuple(int(size) for size in task_sizes)
    if len(sizes) < 2:
        raise ValueError("raw task-score diagnostics require at least two tasks")
    if sum(sizes) != logits.shape[1]:
        raise ValueError("task_sizes must cover every logit column")

    task_scores = torch.stack([
        task_logits.max(dim=1).values
        for task_logits in torch.split(logits, sizes, dim=1)
    ], dim=1)
    class_tasks = torch.arange(len(sizes), device=logits.device).repeat_interleave(
        torch.tensor(sizes, device=logits.device)
    )
    target_tasks = class_tasks[targets]
    task_mask = torch.nn.functional.one_hot(target_tasks, len(sizes)).bool()
    correct = task_scores.gather(1, target_tasks[:, None]).squeeze(1)
    strongest_wrong = task_scores.masked_fill(task_mask, float("-inf")).max(dim=1).values
    return {
        "task_scores": task_scores,
        "target_tasks": target_tasks,
        "predicted_tasks": task_scores.argmax(dim=1),
        "correct": correct,
        "strongest_wrong": strongest_wrong,
        "all_wrong": task_scores.masked_select(~task_mask),
        "margin": correct - strongest_wrong,
    }


def score_distribution_auc(positive_scores, negative_scores):
    """Rank-based AUC with average ranks for tied raw scores."""
    positive_scores = positive_scores.detach().flatten().double().cpu()
    negative_scores = negative_scores.detach().flatten().double().cpu()
    scores = torch.cat((positive_scores, negative_scores))
    labels = torch.cat((
        torch.ones(len(positive_scores), dtype=torch.bool),
        torch.zeros(len(negative_scores), dtype=torch.bool),
    ))
    order = scores.argsort()
    sorted_scores = scores[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(dim=0).double()
    starts = ends - counts + 1
    average_ranks = ((starts + ends) / 2).repeat_interleave(counts)
    positive_rank_sum = average_ranks[labels[order]].sum()
    baseline = len(positive_scores) * (len(positive_scores) + 1) / 2
    return ((positive_rank_sum - baseline) / (
        len(positive_scores) * len(negative_scores)
    )).item()


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
