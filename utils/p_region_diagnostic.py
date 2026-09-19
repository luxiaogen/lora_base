"""Read-only norm-matched interventions on the current task's merged P update."""
from contextlib import contextmanager
import random
import numpy as np
import torch


def matched_removals(safe_delta, conflict_mask):
    conflict = safe_delta.detach().float() * conflict_mask
    nonconflict = safe_delta.detach().float() * (1 - conflict_mask)
    c_norm, u_norm = conflict.norm().item(), nonconflict.norm().item()
    target = 0.5 * min(c_norm, u_norm)
    c_scale = target / c_norm if c_norm else 0.0
    u_scale = target / u_norm if u_norm else 0.0
    return {"conflict": (conflict * c_scale).cpu(), "nonconflict": (nonconflict * u_scale).cpu()}, {
        "conflict_norm": c_norm, "nonconflict_norm": u_norm, "removed_norm": target,
        "conflict_fraction": c_scale, "nonconflict_fraction": u_scale}


@contextmanager
def preserve_diagnostic_state(network):
    py_state, np_state = random.getstate(), np.random.get_state()
    modes = [(module, module.training) for module in network.modules()]
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            network.eval()
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        for module, mode in modes:
            module.training = mode


@contextmanager
def temporary_removal(modules, mode):
    backups = []
    try:
        with torch.no_grad():
            for module in modules:
                removals = getattr(module, "_p_region_removals", None)
                if mode == "baseline" or removals is None:
                    continue
                weight = module.qkv.weight
                backups.append((weight, weight.detach().clone()))
                weight.sub_(removals[mode].to(weight))
        yield
    finally:
        with torch.no_grad():
            for weight, original in backups:
                weight.copy_(original)


def collect_logits(network, loader, device):
    logits, labels, indices = [], [], []
    for index, images, target in loader:
        logits.append(network.interface(images.to(device)).detach().cpu())
        labels.append(target.cpu())
        indices.append(index.cpu())
    return torch.cat(logits), torch.cat(labels), torch.cat(indices)


def margins(logits, labels, task_sizes):
    bounds = torch.tensor([0] + list(np.cumsum(task_sizes)))
    true_task = torch.bucketize(labels, bounds[1:], right=True)
    classes = torch.arange(logits.shape[1])[None, :]
    same = (classes >= bounds[true_task, None]) & (classes < bounds[true_task + 1, None])
    positive = logits.gather(1, labels[:, None]).squeeze(1)
    wrong = classes != labels[:, None]
    local = positive - logits.masked_fill(~(same & wrong), -torch.inf).max(1).values
    cross = positive - logits.masked_fill(same, -torch.inf).max(1).values
    return local, cross, true_task, torch.bucketize(logits.argmax(1), bounds[1:], right=True)


def describe(values):
    values = values[torch.isfinite(values)].float()
    if not values.numel():
        return None
    return {"mean": values.mean().item(), "median": values.median().item(),
            "q25": values.quantile(0.25).item(), "q75": values.quantile(0.75).item(),
            "positive_fraction": (values > 0).float().mean().item()}


def summarize(base, candidate, labels, task_sizes):
    old = labels < sum(task_sizes[:-1])
    base_ok, candidate_ok = base.argmax(1) == labels, candidate.argmax(1) == labels
    before, after = margins(base, labels, task_sizes), margins(candidate, labels, task_sizes)
    report = {}
    for name, selected in (("all", torch.ones_like(old)), ("old", old), ("new", ~old)):
        count = int(selected.sum())
        report[name] = {"count": count}
        if count:
            report[name].update({
                "accuracy": candidate_ok[selected].float().mean().item() * 100,
                "task_accuracy": (after[2][selected] == after[3][selected]).float().mean().item() * 100,
                "corrected": int((~base_ok & candidate_ok & selected).sum()),
                "broken": int((base_ok & ~candidate_ok & selected).sum()),
                "local_margin_change": describe((after[0] - before[0])[selected]),
                "cross_margin_change": describe((after[1] - before[1])[selected]),
            })
    return report


def compare_regions(network, modules, loader, device, task_sizes):
    with preserve_diagnostic_state(network):
        base, labels, indices = collect_logits(network, loader, device)
        reports = {"baseline": summarize(base, base, labels, task_sizes)}
        for mode in ("conflict", "nonconflict"):
            with temporary_removal(modules, mode):
                candidate, other_labels, other_indices = collect_logits(network, loader, device)
            if not torch.equal(labels, other_labels) or not torch.equal(indices, other_indices):
                raise ValueError("Diagnostic requires a deterministic, non-shuffled loader")
            reports[mode] = summarize(base, candidate, labels, task_sizes)
        return reports
