"""Offline checks for head-only old/new competition; never run by training scripts."""
import ast
import copy
import json
import logging
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.head_balance import head_balance_loss, sample_old_feature_pool
from models.losses import AngularPenaltySMLoss


def learner_method(name):
    tree = ast.parse(Path('methods/dlora.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(torch=torch, logging=logging)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<learner>', 'exec'), ns)
    return ns[name]


class HeadBalanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.features = torch.randn(6, 4, requires_grad=True)
        self.old_features = torch.randn(3, 4, requires_grad=True)
        self.old_weights = nn.Parameter(torch.randn(3, 4))
        self.new_weights = nn.Parameter(torch.randn(2, 4))
        self.labels = torch.tensor([0, 0, 0, 0, 1, 1])

    def loss(self):
        return head_balance_loss(self.features, self.labels, self.old_features,
                                 self.old_weights, self.new_weights, 20.)

    def test_only_current_head_receives_gradient(self):
        loss, metrics = self.loss()
        loss.backward()
        self.assertIsNone(self.features.grad)
        self.assertIsNone(self.old_features.grad)
        self.assertIsNone(self.old_weights.grad)
        self.assertGreater(self.new_weights.grad.norm().item(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(not x.requires_grad for x in metrics.values()))

    def test_class_balanced_global_ce_and_global_labels(self):
        loss, _ = self.loss()
        w = F.normalize(torch.cat([self.old_weights, self.new_weights]), dim=1)
        new_ce = F.cross_entropy(20 * F.normalize(self.features, dim=1) @ w.T,
                                 self.labels + 3, reduction='none')
        new_ce = (new_ce[self.labels == 0].mean() + new_ce[self.labels == 1].mean()) / 2
        old_ce = F.cross_entropy(20 * F.normalize(self.old_features, dim=1) @ w.T,
                                 torch.arange(3))
        torch.testing.assert_close(loss, (2 * new_ce + 3 * old_ce) / 5)

    def test_repeating_a_new_class_does_not_change_its_weight(self):
        a, _ = head_balance_loss(self.features[[0, 4]], torch.tensor([0, 1]),
                                 self.old_features, self.old_weights, self.new_weights, 20.)
        b, _ = head_balance_loss(self.features[[0, 0, 0, 4]], torch.tensor([0, 0, 0, 1]),
                                 self.old_features, self.old_weights, self.new_weights, 20.)
        torch.testing.assert_close(a, b)

    def test_missing_new_class_keeps_configured_partition_weight(self):
        labels = torch.zeros(3, dtype=torch.long)
        features = self.features[:3]
        loss, _ = head_balance_loss(features, labels, self.old_features,
                                    self.old_weights, self.new_weights, 20.)
        weights = F.normalize(torch.cat([self.old_weights, self.new_weights]), dim=1)
        new_ce = F.cross_entropy(20 * F.normalize(features, dim=1) @ weights.T, labels + 3)
        old_ce = F.cross_entropy(20 * F.normalize(self.old_features, dim=1) @ weights.T, torch.arange(3))
        torch.testing.assert_close(loss, (2 * new_ce + 3 * old_ce) / 5)

    def test_original_backbone_gradient_unchanged_in_same_step(self):
        base = F.cross_entropy(self.features @ self.new_weights.T, self.labels)
        expected = torch.autograd.grad(base, self.features, retain_graph=True)[0]
        aux, _ = self.loss()
        actual = torch.autograd.grad(base + .1 * aux, self.features)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_old_negative_term_discourages_new_head_response(self):
        old = torch.tensor([[1., 0.]])
        current = nn.Parameter(torch.tensor([[1., 1.]]))
        _, metrics = head_balance_loss(torch.tensor([[0., 1.]]), torch.tensor([0]),
                                       old, old, current, 20.)
        # Inspect the old CE itself: its gradient rotates the new head away from old input.
        logits = 20 * F.normalize(old, dim=1) @ F.normalize(torch.cat([old, current]), dim=1).T
        grad = torch.autograd.grad(F.cross_entropy(logits, torch.tensor([0])), current)[0]
        before = F.cosine_similarity(old, current).item()
        after = F.cosine_similarity(old, current - .01 * grad).item()
        self.assertLess(after, before)
        self.assertIn('head_balance_old_ce', metrics)

    def test_pool_reuses_gaussian_statistics_and_local_rng(self):
        means = torch.tensor([[2., 0.], [0., 2.], [3., 3.]])
        covs = torch.eye(2).repeat(3, 1, 1) * .04
        original = (means.clone(), covs.clone())
        rng = torch.get_rng_state().clone()
        pool = sample_old_feature_pool(means, covs, [2, 1, 2], 2, 'cpu',
                                       torch.Generator().manual_seed(19))
        generator = torch.Generator().manual_seed(19)
        expected = []
        for c, t in enumerate([0, 0, 1]):
            noise = torch.randn(256, 2, generator=generator)
            expected.append(noise @ torch.linalg.cholesky(covs[c]).T + means[c] * (.9 + .1 * (t + 1) / 3))
        torch.testing.assert_close(pool, torch.stack(expected))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(pool.device.type, 'cpu')
        self.assertFalse(pool.requires_grad)
        torch.testing.assert_close(means, original[0])
        torch.testing.assert_close(covs, original[1])

    def test_task0_and_disabled_are_noops_without_old_statistics(self):
        prepare = learner_method('_prepare_head_balance')
        term = learner_method('_head_balance_term')
        for task, args in [(0, {'head_balance_weight': .1}), (1, {}), (1, {'head_balance_weight': 0.})]:
            obj = SimpleNamespace(args=args, _cur_task=task)
            rng = torch.get_rng_state().clone()
            prepare(obj)
            self.assertEqual(term(obj, {}, None), (None, {}))
            self.assertIsNone(obj._head_balance_pool)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_learner_term_samples_every_old_class_and_updates_only_new_head(self):
        prepare = learner_method('_prepare_head_balance')
        term = learner_method('_head_balance_term')
        net = nn.Module()
        net.classifier_pool = nn.ModuleList([nn.Linear(4, 3, bias=False), nn.Linear(4, 2, bias=False)])
        obj = SimpleNamespace(args={'head_balance_weight': .1, 'scale': 20}, _cur_task=1,
                              _known_classes=3, _network=net, _device=torch.device('cpu'),
                              task_sizes=[3, 2], _class_means=torch.randn(3, 4),
                              _class_covs=torch.eye(4).repeat(3, 1, 1))
        rng = torch.get_rng_state().clone()
        prepare(obj)
        before = copy.deepcopy(net.state_dict())
        loss, metrics = term(obj, {'features': self.features}, self.labels)
        optimizer = torch.optim.SGD(net.parameters(), lr=.01)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(before['classifier_pool.0.weight'], net.classifier_pool[0].weight))
        self.assertFalse(torch.equal(before['classifier_pool.1.weight'], net.classifier_pool[1].weight))
        self.assertIsNone(self.features.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(obj._head_balance_pool.shape, (3, 256, 4))
        self.assertIn('head_balance_weighted', metrics)

    def test_training_calls_prepare_term_and_releases_pool(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        train = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'train_function')
        calls = [n.func.attr for n in ast.walk(train) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual(calls.count('_prepare_head_balance'), 1)
        self.assertEqual(calls.count('_head_balance_term'), 1)
        source = ast.get_source_segment(Path('methods/dlora.py').read_text(), train)
        self.assertIn('self._head_balance_pool = None', source)

    def test_actual_training_loop_default_matches_previous_and_candidate_runs(self):
        class Progress:
            def __init__(self, items):
                self.items = items

            def __iter__(self):
                return iter(self.items)

            def set_description(self, description):
                pass

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(4, 4)
                self.classifier_pool = nn.ModuleList([nn.Linear(4, 3, bias=False), nn.Linear(4, 2, bias=False)])
                self.classifier_pool[0].requires_grad_(False)

            def forward(self, x):
                features = self.encoder(x)
                logits = F.linear(F.normalize(features, dim=1), F.normalize(self.classifier_pool[1].weight, dim=1))
                return {'features': features, 'logits': logits}

        def training_method(source):
            tree = ast.parse(source)
            method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'train_function')
            ns = dict(torch=torch, logging=logging, np=np, tqdm=Progress,
                      count_parameters=lambda net, *_: sum(p.numel() for p in net.parameters() if p.requires_grad),
                      tensor2numpy=lambda x: x.cpu().numpy(), AngularPenaltySMLoss=AngularPenaltySMLoss)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<train>', 'exec'), ns)
            return ns['train_function']

        current_source = Path('methods/dlora.py').read_text()
        previous_source = subprocess.check_output(['git', 'show', 'f304601:methods/dlora.py'], text=True)
        net = Network()
        initial = copy.deepcopy(net.state_dict())
        data = [(torch.arange(6), self.features.detach().clone(), self.labels + 3)] * 2
        runs = []
        for source, weight in [(previous_source, 0.), (current_source, 0.), (current_source, .1)]:
            obj = SimpleNamespace(args={'scale': 20, 'head_balance_weight': weight},
                                  _cur_task=1, _known_classes=3, _device=torch.device('cpu'),
                                  _network=net, task_sizes=[3, 2], run_epoch=2, scale=20, margin=.1,
                                  _class_means=torch.ones(3, 4), _class_covs=torch.eye(4).repeat(3, 1, 1),
                                  _training_margin=lambda: .1, _global_conflict_enabled=lambda: False,
                                  _extra_training_context=lambda *args: None,
                                  _extra_training_loss=lambda **kwargs: None,
                                  _old_competition_term=lambda *args: (None, {}),
                                  _old_model_distillation_term=lambda *args: (None, {}),
                                  _compute_accuracy=lambda *args: 0.)
            for name in ('_prepare_head_balance', '_head_balance_term', '_backward_and_step'):
                setattr(obj, name, learner_method(name).__get__(obj))
            net.load_state_dict(initial)
            optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=.02)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            torch.manual_seed(29)
            training_method(source)(obj, data, [], optimizer, scheduler)
            runs.append((copy.deepcopy(net.state_dict()), torch.get_rng_state().clone()))
            if source == current_source:
                self.assertIsNone(obj._head_balance_pool)
            self.assertTrue(torch.equal(net.classifier_pool[0].weight, initial['classifier_pool.0.weight']))
            self.assertIsNone(net.classifier_pool[0].weight.grad)
            if weight:
                self.assertGreater(obj._last_epoch_training_loss_metrics['head_balance_weighted'], 0)
        for key in initial:
            self.assertTrue(torch.equal(runs[0][0][key], runs[1][0][key]), key)
        self.assertTrue(torch.equal(runs[0][1], runs[1][1]))
        self.assertTrue(torch.equal(runs[1][1], runs[2][1]))
        self.assertFalse(torch.equal(runs[1][0]['classifier_pool.1.weight'], runs[2][0]['classifier_pool.1.weight']))

    def test_machine_specs_and_scripts_are_paired(self):
        for gpu, anchor in [('3090', 10), ('5090', 5)]:
            path = f'scripts/9_27_imgr10_head_balance_{gpu}.sh'
            spec = json.loads(Path(f'scripts/sweeps/imgr10_head_balance_{gpu}.json').read_text())
            common = spec['common_overrides']
            self.assertEqual(common['dual_mask_anchor_reg_weight'], anchor)
            self.assertEqual([v['overrides'] for v in spec['variants']],
                             [{'head_balance_weight': 0.}, {'head_balance_weight': .1}])
            self.assertEqual(spec['seeds'], [1993])
            self.assertEqual(common['ca_epochs'], 5)
            self.assertFalse(common['ca_real_new_features'])
            self.assertEqual(common['old_competition_weight'], 0)
            self.assertTrue(common['disable_fused_sdpa'])
            source = Path(path).read_text()
            for forbidden in ('unittest', 'pytest', 'preflight', 'check_ca_real_new'):
                self.assertNotIn(forbidden, source)
            subprocess.run(['bash', '-n', path], check=True)
            commands = subprocess.check_output(['bash', path, '--dry-run'], text=True).split('    python main.py')[1:]
            self.assertEqual(len(commands), 2)
            for command, variant in zip(commands, spec['variants']):
                tokens = shlex.split(command.replace('\\\n', ' '))
                settings = dict(tokens[i+1].split('=', 1) for i, token in enumerate(tokens) if token == '--set')
                for key, value in {**common, **variant['overrides']}.items():
                    self.assertEqual(settings[key], value if isinstance(value, str) else json.dumps(value, separators=(',', ':')))
                self.assertNotIn('data_path', settings)


if __name__ == '__main__':
    unittest.main()
