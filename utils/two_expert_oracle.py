"""Read-only complementarity bound; label-free signals never select training updates."""
import csv
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F


def rank_auc(scores, positive):
    n_positive = int(positive.sum())
    n_negative = positive.numel() - n_positive
    if not n_positive or not n_negative:
        return None
    order = scores.argsort()
    _, counts = torch.unique_consecutive(scores[order], return_counts=True)
    ends = counts.cumsum(0)
    midranks = (ends - counts + 1 + ends).double() / 2
    ranks = torch.repeat_interleave(midranks, counts)
    rank_sum = float(ranks[positive[order]].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def expert_signals(base, anchor):
    """Both inputs are cosine scores over exactly the same all-seen classes."""
    base_pred, anchor_pred = base.argmax(1), anchor.argmax(1)
    base_top, anchor_top = base.topk(2, dim=1).values, anchor.topk(2, dim=1).values
    base_gap = base_top[:, 0] - base_top[:, 1]
    anchor_gap = anchor_top[:, 0] - anchor_top[:, 1]
    gather = lambda logits, predictions: logits.gather(1, predictions[:, None]).squeeze(1)
    return dict(margin_advantage=anchor_gap - base_gap,
                proposal_advantage=(gather(anchor, anchor_pred) - gather(anchor, base_pred)
                                    - gather(base, base_pred) + gather(base, anchor_pred)),
                base_uncertainty=-base_gap)


def expert_rows(base, anchor, targets, indices, class_ids, class_size):
    rows = dict(index=indices, target=targets,
                base_prediction=class_ids[base.argmax(1)],
                anchor_prediction=class_ids[anchor.argmax(1)],
                **expert_signals(base, anchor))
    # True task information is used ONLY for a separately labelled diagnostic ceiling.
    allowed = class_ids[None, :] // class_size == targets[:, None] // class_size
    rows['task_id_prediction'] = class_ids[base.masked_fill(~allowed, float('-inf')).argmax(1)]
    return rows


def collect_experts(network, loader, device, anchor_context, prototypes, class_ids, class_size):
    modes = [(module, module.training) for module in network.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    generator = getattr(loader, 'generator', None)
    generator_state = generator.get_state() if generator is not None else None
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    parts = []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            network.eval()
            normalized_prototypes = F.normalize(prototypes, dim=1)
            for indices, inputs, targets in loader:
                inputs = inputs.to(device)
                base = network.interface(inputs)[:, class_ids].detach().float().cpu()
                # Identical transformed inputs; restore the merged weights after this forward.
                with anchor_context():
                    features = network.extract_vector(inputs)
                    anchor = (F.normalize(features, dim=1) @ normalized_prototypes.T).float().cpu()
                parts.append(expert_rows(base, anchor, targets.cpu(), torch.as_tensor(indices).cpu(),
                                         class_ids.cpu(), class_size))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if generator_state is not None:
            generator.set_state(generator_state)
        for module, training in modes:
            module.training = training
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def expert_report(rows, known_classes):
    target = rows['target']
    base_correct = rows['base_prediction'].eq(target)
    anchor_correct = rows['anchor_prediction'].eq(target)
    disagree = rows['base_prediction'].ne(rows['anchor_prediction'])
    rescued = ~base_correct & anchor_correct
    harmed = base_correct & ~anchor_correct
    neutral = disagree & ~base_correct & ~anchor_correct
    groups = {}
    for name, mask in dict(total=torch.ones_like(base_correct), old=target < known_classes,
                           new=target >= known_classes).items():
        n = int(mask.sum())
        accuracy = lambda correct: 100. * int((correct & mask).sum()) / n if n else None
        row = dict(n=n, base_accuracy=accuracy(base_correct), anchor_accuracy=accuracy(anchor_correct),
                   oracle_accuracy=accuracy(base_correct | anchor_correct),
                   oracle_gain_pp=100. * int((rescued & mask).sum()) / n if n else None,
                   task_id_ceiling_accuracy=accuracy(rows['task_id_prediction'].eq(target)),
                   disagreements=int((disagree & mask).sum()), rescued=int((rescued & mask).sum()),
                   harmed=int((harmed & mask).sum()), neutral_disagreement=int((neutral & mask).sum()),
                   signals={})
        informative = mask & (rescued | harmed)
        for signal in ('margin_advantage', 'proposal_advantage', 'base_uncertainty'):
            values = rows[signal]
            metric = dict(rescue_vs_harm_auc=rank_auc(values[informative], rescued[informative]),
                          auc_n=int(informative.sum()))
            if signal != 'base_uncertainty':
                # Predeclared threshold zero; no fitting or selection on test labels.
                switch = disagree & (values > 0)
                prediction_correct = torch.where(switch, anchor_correct, base_correct)
                r, h = int((switch & rescued & mask).sum()), int((switch & harmed & mask).sum())
                metric.update(fixed_threshold=0., fixed_rule_accuracy=accuracy(prediction_correct),
                              selected_rescued=r, selected_harmed=h,
                              selected_neutral=int((switch & neutral & mask).sum()),
                              net_gain_pp=100. * (r - h) / n if n else None)
            row['signals'][signal] = metric
        groups[name] = row
    return dict(groups=groups)


def save_report(directory, task, source, rows, known_classes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f'task_{task:02d}_{source}'
    report = dict(task=task, source=source, class_scope='all_seen',
                  used_for_training=False, old_train_images_accessed=False,
                  oracle_uses_true_class_labels=True, signals_use_labels_or_task_id=False,
                  thresholds_fitted=False, probe_is_unseen_holdout=False,
                  prototype_source='immutable_W_pre_current_train_when_each_class_arrived',
                  auc_population='disagreements_with_exactly_one_expert_correct; excludes_both_wrong',
                  **expert_report(rows, known_classes))
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    columns = list(rows)
    with stem.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*(rows[key].tolist() for key in columns)))
    logging.info('TwoExpertOracle %s', json.dumps(report, allow_nan=False))
    return report
