"""Finite S/P B-step choices: privileged reference vs current-input proxies."""
import hashlib
import json
import logging

import torch
from torch.nn import functional as F

from utils.p_old_gradient_oracle import (
    build_oracle_loaders, copy_p, independent_probe, oracle_eval,
)


def candidate_coefficients(snapshots, scope):
    if scope == 'p':
        return [(1., 1.), (1., 0.), (1., .5), (1., 1.5), (1., 2.)]
    energy = {branch: sum(float((param.detach() - before).double().square().sum())
                          for name, param, before in snapshots if name == branch)
              for branch in ('S', 'P')}
    result = [(1., 1.)]
    for s, p in ((0., 1.), (.5, 1.5), (1.5, .5), (1., 0.)):
        denominator = s * s * energy['S'] + p * p * energy['P']
        if denominator == 0:
            result.append((1., 1.))
        else:
            normalizer = ((energy['S'] + energy['P']) / denominator) ** .5
            result.append((s * normalizer, p * normalizer))
    return result


def choose_candidates(rows):
    """Improve new all-seen CE without increasing the chosen risk vs raw SGD."""
    raw = rows[0]
    choices = {}
    for name, risk in (('oracle', 'old_loss'), ('logit', 'old_logit_shift'),
                       ('feature', 'feature_shift')):
        if risk not in raw:
            choices[name] = None
            continue
        tolerance = 1e-6 if risk == 'old_loss' else abs(raw[risk]) * .001 + 1e-14
        allowed = [i for i, row in enumerate(rows) if row[risk] <= raw[risk] + tolerance]
        best = min(allowed, key=lambda i: rows[i]['new_loss'])
        choices[name] = best if rows[best]['new_loss'] < raw['new_loss'] - 1e-7 else 0
    return choices


@torch.no_grad()
def apply_branch_choice(network, snapshots, inputs, targets, old_batch,
                        known_classes, scale, mode, scope):
    """Hold heads/momentum common; swap only current S/P B displacements.

    The SP comparison matches joint B-step norm, not gated QKV update norm.
    The P comparison intentionally varies P-step magnitude with S unchanged.
    """
    model = network.module if isinstance(network, torch.nn.DataParallel) else network
    params = [param for _, param, _ in snapshots]
    raw = [param.detach().clone() for param in params]
    coefficients = candidate_coefficients(snapshots, scope)
    steps = [end - before for end, (_, _, before) in zip(raw, snapshots)]
    target_norm = float(sum(step.double().square().sum() for step in steps).sqrt())
    selected_values = raw

    def values(pair):
        if pair == (1., 1.):
            return raw
        return [before + pair[0 if branch == 'S' else 1] * step
                for (branch, _, before), step in zip(snapshots, steps)]

    weights = torch.cat([head.weight for head in model.classifier_pool[:model.numtask]])
    weights = F.normalize(weights, dim=1)

    def encode(batch):
        features = F.normalize(model.extract_vector(batch), dim=1)
        return features, F.linear(features, weights)

    with oracle_eval(network):
        try:
            reference = [end if scope == 'p' and branch == 'S' else before
                         for end, (branch, _, before) in zip(raw, snapshots)]
            copy_p(params, reference)
            ref_features, ref_logits = encode(inputs)
            rows = []
            combined = inputs if old_batch is None else torch.cat((inputs, old_batch[0]))
            for index, pair in enumerate(coefficients):
                copy_p(params, raw if index == 0 else values(pair))
                features, logits = encode(combined)
                new_features, new_logits = features[:len(inputs)], logits[:len(inputs)]
                row = dict(s_multiplier=pair[0], p_multiplier=pair[1],
                           new_loss=float(F.cross_entropy((scale * new_logits).double(), targets)),
                           old_logit_shift=float((new_logits[:, :known_classes] -
                                                  ref_logits[:, :known_classes]).square().mean()),
                           feature_shift=float(.5 * (new_features.double() - ref_features.double())
                                               .square().sum(1).mean()),
                           joint_b_step_norm=float(sum((param.detach().double() - before.double())
                                                       .square().sum()
                                                       for _, param, before in snapshots).sqrt()))
                if old_batch is not None:
                    row['old_loss'] = float(F.cross_entropy((scale * logits[len(inputs):]).double(), old_batch[1]))
                rows.append(row)
            choices = choose_candidates(rows)
            selected = 0 if mode == 'audit' else choices[mode]
            selected_values = raw if selected == 0 else values(coefficients[selected])
            return dict(applied=selected != 0, selected=selected,
                        choices=choices, candidates=rows,
                        new_objective='all_seen_ce_selector_only',
                        risk_reference='raw_sgd_with_common_post_sgd_heads',
                        norm_matched=scope == 'sp', norm_space='joint_S_P_B_displacement',
                        target_joint_b_step_norm=target_norm,
                        optimizer_momentum='raw_sgd_unchanged')
        finally:
            copy_p(params, selected_values)


class BranchStepChoice:
    def __init__(self, data_manager, known_classes, total_classes, seed, batch_size, mode, scope):
        self.mode, self.scope, self.known_classes = mode, scope, known_classes
        self.loaders = None
        self.iterator = None
        if mode in ('audit', 'oracle'):
            self.loaders = build_oracle_loaders(data_manager, known_classes, total_classes, seed, batch_size)
        self.counts = dict(sampled=0, applied=0, choices=[0] * 5)
        data = dict(mode=mode, scope=scope, formal_exemplar_free=mode not in ('audit', 'oracle'),
                    old_data_access=self.loaders is not None,
                    source='old_train_privileged' if self.loaders else 'current_train_only',
                    checkpoints=False, dense_history_components=False)
        if self.loaders:
            data.update(samples=dict(zip(('old_selector', 'old_probe', 'new_probe'),
                                         (len(loader.dataset) for loader in self.loaders))),
                        sample_sha256=hashlib.sha256(json.dumps(
                            [loader.dataset.indices for loader in self.loaders]).encode()).hexdigest(),
                        selector_probe_disjoint=True, probe_used_for_selection=False)
        logging.info('BranchChoiceData %s', json.dumps(data))

    def step(self, network, snapshots, inputs, targets, scale, task, epoch, batch):
        old_batch = None
        if self.loaders:
            if self.iterator is None:
                self.iterator = iter(self.loaders[0])
            try:
                _, old_inputs, old_targets = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loaders[0])
                _, old_inputs, old_targets = next(self.iterator)
            old_batch = (old_inputs.to(inputs.device), old_targets.to(inputs.device))
        reference = [param.detach().clone() for _, param, _ in snapshots]
        row = apply_branch_choice(network, snapshots, inputs, targets + self.known_classes,
                                  old_batch, self.known_classes, scale, self.mode, self.scope)
        self.counts['sampled'] += 1
        self.counts['applied'] += int(row['applied'])
        self.counts['choices'][row['selected']] += 1
        logging.info('BranchChoiceStep %s', json.dumps(dict(
            task=task, epoch=epoch + 1, batch=batch + 1, mode=self.mode, scope=self.scope,
            source='train_selector', test_used_for_selection=False, **row), allow_nan=False))
        if self.loaders and batch == 0 and epoch in (0, 9, 19):
            # Audit probes the hypothetical Oracle, but always restores raw SGD.
            selected = [param.detach().clone() for _, param, _ in snapshots]
            index = row['choices']['oracle'] if self.mode == 'audit' else row['selected']
            pair = row['candidates'][index]
            hypothetical = reference if index == 0 else [
                before + pair['s_multiplier' if branch == 'S' else 'p_multiplier'] * (end - before)
                for (branch, _, before), end in zip(snapshots, reference)]
            try:
                probe = independent_probe(network, [(p, b) for _, p, b in snapshots], reference,
                                          hypothetical, self.loaders[1:], inputs.device, self.known_classes)
                logging.info('BranchChoiceProbe %s', json.dumps(dict(
                    task=task, epoch=epoch + 1, mode=self.mode, scope=self.scope,
                    hypothetical=self.mode == 'audit', source='disjoint_train_probe_not_unseen_holdout',
                    used_for_selection=False, metrics=probe), allow_nan=False))
            finally:
                copy_p([p for _, p, _ in snapshots], selected)

    def summary(self, task, epoch):
        logging.info('BranchChoiceSummary %s', json.dumps(dict(
            task=task, epoch=epoch + 1, mode=self.mode, scope=self.scope, **self.counts)))
        self.counts = dict(sampled=0, applied=0, choices=[0] * 5)
