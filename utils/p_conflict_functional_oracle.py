"""Read-only functional diagnosis for grouped P-conflict updates."""
from contextlib import contextmanager
import math
import random

import numpy as np
import torch

from utils.p_region_diagnostic import collect_logits, preserve_diagnostic_state, summarize


PROJECTIONS = ("q", "k", "v")


def rank_slices(rank, groups):
    groups = max(1, min(int(groups), int(rank)))
    base, extra = divmod(int(rank), groups)
    result, start = [], 0
    for group in range(groups):
        stop = start + base + (group < extra)
        result.append(slice(start, stop))
        start = stop
    return result


def svd_energy_slices(singular_values, groups):
    """Split ordered singular directions into non-empty, near-equal energy groups."""
    rank = int(singular_values.numel())
    groups = max(1, min(int(groups), rank))
    energy = singular_values.detach().float().pow(2)
    total = float(energy.sum().item())
    if total == 0.0:
        return rank_slices(rank, groups)

    cumulative = torch.cumsum(energy, dim=0)
    boundaries = [0]
    for group in range(1, groups):
        target = group * total / groups
        boundary = int(torch.searchsorted(cumulative, target).item()) + 1
        boundary = max(boundaries[-1] + 1, boundary)
        boundary = min(rank - (groups - group), boundary)
        boundaries.append(boundary)
    boundaries.append(rank)
    return [slice(boundaries[i], boundaries[i + 1]) for i in range(groups)]


def prepare_svd_components(modules):
    """Create orthogonal SVD groups from the retained masked conflict update."""
    for module in modules:
        state = getattr(module, "_p_conflict_functional_state", None)
        if state is None or state.get("decomposition", "rank") != "svd":
            continue
        if "svd_components" in state:
            continue

        dim = module.qkv.weight.shape[1]
        components, energy_fractions = {}, {}
        max_diff = 0.0
        for projection_index, projection in enumerate(PROJECTIONS):
            rows = slice(projection_index * dim, (projection_index + 1) * dim)
            conflict = state["conflict_component"][rows].to(module.qkv.weight.device)
            u, singular_values, vh = torch.linalg.svd(conflict, full_matrices=False)
            slices = svd_energy_slices(singular_values, state["rank_groups"])
            projection_components = []
            total_energy = singular_values.float().pow(2).sum().clamp_min(1e-30)
            fractions = []
            for ranks in slices:
                component = (u[:, ranks] * singular_values[ranks]) @ vh[ranks]
                projection_components.append(component.detach().float().cpu())
                fractions.append(float(singular_values[ranks].float().pow(2).sum().div(total_energy).item()))
            reconstructed = torch.stack(
                [component.to(conflict.device) for component in projection_components]
            ).sum(dim=0)
            max_diff = max(max_diff, float((reconstructed - conflict.float()).abs().max().item()))
            components[projection] = projection_components
            energy_fractions[projection] = fractions
        state["svd_components"] = components
        state["svd_energy_fractions"] = energy_fractions
        state["svd_reconstruction_max_abs_diff"] = max_diff


def balanced_holdout_indices(labels, per_class=4):
    """Return disjoint deterministic selector/evaluator indices per class."""
    labels = np.asarray(labels)
    per_class = int(per_class)
    if per_class < 2:
        raise ValueError("per_class must be at least 2")
    selector, evaluator = [], []
    selector_count = per_class // 2
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)[:per_class]
        if len(indices) < per_class:
            raise ValueError("Each class needs at least per_class samples")
        selector.extend(indices[:selector_count].tolist())
        evaluator.extend(indices[selector_count:].tolist())
    return selector, evaluator


def component_specs(modules):
    specs = []
    for layer, module in enumerate(modules):
        state = getattr(module, "_p_conflict_functional_state", None)
        if state is None:
            continue
        for projection in PROJECTIONS:
            if state.get("decomposition", "rank") == "svd":
                groups = len(state["svd_components"][projection])
            else:
                groups = len(rank_slices(state["A"].shape[0], state["rank_groups"]))
            specs.extend((layer, projection, group) for group in range(groups))
    return specs


def materialize_component(module, spec):
    """Materialize one retained P-conflict contribution on the weight device."""
    _, projection, group = spec
    state = module._p_conflict_functional_state
    if state.get("decomposition", "rank") == "svd":
        return state["svd_components"][projection][group].to(module.qkv.weight.device)

    dim = module.qkv.weight.shape[1]
    projection_index = PROJECTIONS.index(projection)
    rows = slice(projection_index * dim, (projection_index + 1) * dim)
    ranks = rank_slices(state["A"].shape[0], state["rank_groups"])[group]
    device = module.qkv.weight.device
    a = state["A"][ranks].to(device=device)
    b = state["B"][rows, ranks].to(device=device)
    gate = state["gate"][rows].to(device=device)
    return float(state["gamma"]) * (b @ a) * gate


def reconstruct_conflict_component(module):
    prepare_svd_components([module])
    result = torch.zeros_like(module.qkv.weight, dtype=torch.float32)
    for spec in component_specs([module]):
        _, projection, _ = spec
        dim = module.qkv.weight.shape[1]
        projection_index = PROJECTIONS.index(projection)
        rows = slice(projection_index * dim, (projection_index + 1) * dim)
        local_spec = (0, projection, spec[2])
        result[rows].add_(materialize_component(module, local_spec).float())
    return result


@contextmanager
def temporary_plan(modules, plan):
    """Subtract a component plan and restore the exact affected weight rows."""
    grouped = {}
    for spec, scale in plan.items():
        if float(scale) == 0.0:
            continue
        layer, projection, _ = spec
        key = (int(layer), projection)
        component = materialize_component(modules[layer], spec) * float(scale)
        grouped[key] = grouped.get(key, 0.0) + component

    backups = []
    squared_norm = 0.0
    try:
        with torch.no_grad():
            for (layer, projection), removal in grouped.items():
                module = modules[layer]
                dim = module.qkv.weight.shape[1]
                start = PROJECTIONS.index(projection) * dim
                rows = slice(start, start + dim)
                backups.append((module.qkv.weight, rows, module.qkv.weight[rows].detach().clone()))
                module.qkv.weight[rows].sub_(removal.to(module.qkv.weight))
                squared_norm += float(removal.float().pow(2).sum().item())
        yield {"actual_combined_norm": math.sqrt(squared_norm)}
    finally:
        with torch.no_grad():
            for weight, rows, original in backups:
                weight[rows].copy_(original)


def energy_matched_plan(specs, norms, target_energy, order):
    """Match sum(scale^2 * component_norm^2) without amplifying components."""
    remaining = max(0.0, float(target_energy))
    plan = {}
    for spec in order:
        norm = float(norms[spec])
        energy = norm * norm
        if remaining <= 0.0:
            break
        if energy <= remaining:
            plan[spec] = 1.0
            remaining -= energy
        elif norm > 0.0:
            plan[spec] = math.sqrt(remaining) / norm
            remaining = 0.0
    return plan


def plan_energy(plan, norms):
    return sum((float(scale) * float(norms[spec])) ** 2 for spec, scale in plan.items())


def class_margin(logits, labels):
    positive = logits.gather(1, labels[:, None]).squeeze(1)
    classes = torch.arange(logits.shape[1])[None, :]
    negative = logits.masked_fill(classes == labels[:, None], -torch.inf).max(1).values
    return positive - negative


def _same_batch(labels, indices, other_labels, other_indices):
    if not torch.equal(labels, other_labels) or not torch.equal(indices, other_indices):
        raise ValueError("Diagnostic requires deterministic, non-shuffled loaders")


def _component_record(spec, gain, old, norm):
    old_gain = float(gain[old].mean().item()) if old.any() else float("nan")
    new_gain = float(gain[~old].mean().item()) if (~old).any() else float("nan")
    if old_gain > 0.0 and new_gain > 0.0:
        quadrant = "removal_helps_old_and_new"
    elif old_gain > 0.0:
        quadrant = "removal_helps_old_hurts_new"
    elif new_gain > 0.0:
        quadrant = "removal_hurts_old_helps_new"
    else:
        quadrant = "removal_hurts_old_and_new"
    return {
        "layer": spec[0],
        "projection": spec[1],
        "component_group": spec[2],
        "rank_group": spec[2],
        "component_norm": norm,
        "old_margin_gain": old_gain,
        "new_margin_gain": new_gain,
        "old_win_fraction": float((gain[old] > 0).float().mean().item()) if old.any() else None,
        "new_win_fraction": float((gain[~old] > 0).float().mean().item()) if (~old).any() else None,
        "quadrant": quadrant,
    }


def compare_functional_components(
        network,
        modules,
        selector_loader,
        evaluator_loader,
        device,
        task_sizes,
        random_seed=1993,
):
    """Select with labeled selector samples and evaluate on disjoint samples."""
    with preserve_diagnostic_state(network):
        prepare_svd_components(modules)
        specs = component_specs(modules)
        selector_base, selector_labels, selector_indices = collect_logits(network, selector_loader, device)
        evaluator_base, evaluator_labels, evaluator_indices = collect_logits(network, evaluator_loader, device)
        selector_margin = class_margin(selector_base, selector_labels)
        old = selector_labels < sum(task_sizes[:-1])
        records, norms = [], {}

        for spec in specs:
            with temporary_plan(modules, {spec: 1.0}):
                candidate, labels, indices = collect_logits(network, selector_loader, device)
            _same_batch(selector_labels, selector_indices, labels, indices)
            norm = float(materialize_component(modules[spec[0]], spec).float().norm().item())
            norms[spec] = norm
            records.append(_component_record(spec, class_margin(candidate, labels) - selector_margin, old, norm))

        selected = [
            (record["layer"], record["projection"], record["component_group"])
            for record in records
            if record["quadrant"] == "removal_helps_old_and_new"
        ]
        oracle_plan = {spec: 1.0 for spec in selected}
        target_energy = plan_energy(oracle_plan, norms)
        magnitude_order = sorted(specs, key=lambda spec: norms[spec], reverse=True)
        random_order = list(specs)
        random.Random(int(random_seed)).shuffle(random_order)
        plans = {
            "functional_oracle_holdout": oracle_plan,
            "magnitude_control": energy_matched_plan(specs, norms, target_energy, magnitude_order),
            "random_control": energy_matched_plan(specs, norms, target_energy, random_order),
        }

        modes = {"baseline": summarize(evaluator_base, evaluator_base, evaluator_labels, task_sizes)}
        budgets = {}
        for name, plan in plans.items():
            with temporary_plan(modules, plan) as intervention:
                candidate, labels, indices = collect_logits(network, evaluator_loader, device)
            _same_batch(evaluator_labels, evaluator_indices, labels, indices)
            modes[name] = summarize(evaluator_base, candidate, evaluator_labels, task_sizes)
            budgets[name] = {
                "components": len(plan),
                "component_energy": plan_energy(plan, norms),
                **intervention,
            }

        restored, labels, indices = collect_logits(network, evaluator_loader, device)
        _same_batch(evaluator_labels, evaluator_indices, labels, indices)
        quadrants = {}
        for record in records:
            quadrants[record["quadrant"]] = quadrants.get(record["quadrant"], 0) + 1
        top_joint = sorted(
            records,
            key=lambda record: record["old_margin_gain"] + record["new_margin_gain"],
            reverse=True,
        )[:12]
        return {
            "diagnostic_only": True,
            "uses_true_labels_for_selection": True,
            "selector_and_evaluator_disjoint": True,
            "decomposition": next(
                (
                    module._p_conflict_functional_state.get("decomposition", "rank")
                    for module in modules
                    if getattr(module, "_p_conflict_functional_state", None) is not None
                ),
                None,
            ),
            "components_total": len(specs),
            "layer_private_ranks": [
                int(module._p_conflict_functional_state.get("private_rank", 0))
                for module in modules
                if getattr(module, "_p_conflict_functional_state", None) is not None
            ],
            "layer_reconstruction_max_abs_diff": [
                float(module._p_conflict_functional_state["reconstruction_max_abs_diff"])
                for module in modules
                if getattr(module, "_p_conflict_functional_state", None) is not None
                and "reconstruction_max_abs_diff" in module._p_conflict_functional_state
            ],
            "layer_svd_reconstruction_max_abs_diff": [
                float(module._p_conflict_functional_state["svd_reconstruction_max_abs_diff"])
                for module in modules
                if getattr(module, "_p_conflict_functional_state", None) is not None
                and "svd_reconstruction_max_abs_diff" in module._p_conflict_functional_state
            ],
            "svd_energy_fractions": [
                module._p_conflict_functional_state["svd_energy_fractions"]
                for module in modules
                if getattr(module, "_p_conflict_functional_state", None) is not None
                and "svd_energy_fractions" in module._p_conflict_functional_state
            ],
            "quadrants": quadrants,
            "selected_components": [list(spec) for spec in selected],
            "top_joint_margin_gain": top_joint,
            "budgets": budgets,
            "modes": modes,
            "restored_max_abs_logit_diff": float((restored - evaluator_base).abs().max().item()),
        }
