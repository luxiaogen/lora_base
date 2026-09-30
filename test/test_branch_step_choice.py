import ast
import copy
import json
import logging
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from utils.branch_step_choice import (
    BranchStepChoice, apply_branch_choice, candidate_coefficients, choose_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = torch.nn.Parameter(torch.tensor([.1, -.2]))
        self.p = torch.nn.Parameter(torch.tensor([-.1, .2]))
        self.dropout = torch.nn.Dropout(.2)
        self.classifier_pool = torch.nn.ModuleList([
            torch.nn.Linear(2, 2, bias=False) for _ in range(2)])
        self.numtask = 2

    def extract_vector(self, inputs):
        return self.dropout(inputs) + self.s + self.p

    def interface(self, inputs):
        weights = torch.cat([head.weight for head in self.classifier_pool])
        return F.linear(F.normalize(self.extract_vector(inputs), dim=1),
                        F.normalize(weights, dim=1))

    def forward(self, inputs):
        return {'logits': self.interface(inputs)[:, 2:]}


class BranchChoiceTests(unittest.TestCase):
    def snapshots(self, net):
        return [('S', net.s, net.s.detach().clone() - .03),
                ('P', net.p, net.p.detach().clone() + .07)]

    def test_p_coefficients_do_not_change_s(self):
        net = Network()
        pairs = candidate_coefficients(self.snapshots(net), 'p')
        self.assertEqual(pairs[0], (1., 1.))
        self.assertEqual(sorted(p for s, p in pairs), [0., .5, 1., 1.5, 2.])
        self.assertTrue(all(s == 1. for s, p in pairs))

    def test_sp_candidates_match_joint_b_displacement_not_effective_weights(self):
        snapshots = self.snapshots(Network())
        norms = {branch: float((param.detach() - before).square().sum())
                 for branch, param, before in snapshots}
        target = sum(norms.values())
        for s, p in candidate_coefficients(snapshots, 'sp'):
            self.assertAlmostEqual(s * s * norms['S'] + p * p * norms['P'], target, places=7)

    def test_sp_tiny_steps_and_zero_branch_preserve_relative_norm(self):
        for s_step, p_step in ((1e-7, 1e-7), (1e-3, 0.), (0., 1e-3), (0., 0.)):
            s, p = torch.nn.Parameter(torch.tensor([s_step])), torch.nn.Parameter(torch.tensor([p_step]))
            snapshots = [('S', s, torch.zeros_like(s)), ('P', p, torch.zeros_like(p))]
            target = s_step ** 2 + p_step ** 2
            pairs = candidate_coefficients(snapshots, 'sp')
            self.assertEqual(len(pairs), 5)
            for a, b in pairs:
                measured = a * a * float(s.detach().double().square().sum()) + \
                           b * b * float(p.detach().double().square().sum())
                if target:
                    self.assertAlmostEqual(measured / target, 1., places=6)
                else:
                    self.assertEqual(measured, 0.)

    def test_oracle_and_legal_choices_use_different_caps_and_ties_keep_raw(self):
        rows = [dict(new_loss=2., old_loss=1., old_logit_shift=.2, feature_shift=.3),
                dict(new_loss=1., old_loss=2., old_logit_shift=.1, feature_shift=.2),
                dict(new_loss=1.5, old_loss=.9, old_logit_shift=.3, feature_shift=.1)]
        self.assertEqual(choose_candidates(rows), dict(oracle=2, logit=1, feature=1))
        self.assertEqual(choose_candidates([rows[0], dict(rows[0])]),
                         dict(oracle=0, logit=0, feature=0))
        legal = [{k: v for k, v in row.items() if k != 'old_loss'} for row in rows]
        self.assertIsNone(choose_candidates(legal)['oracle'])

    def test_legal_modes_do_not_construct_an_old_loader(self):
        manager = SimpleNamespace(get_dataset=lambda *a, **kw: self.fail('legal mode reads old data'))
        with patch('utils.branch_step_choice.build_oracle_loaders') as builder:
            for mode in ('logit', 'feature'):
                chooser = BranchStepChoice(manager, 2, 4, 1993, 8, mode, 'p')
                self.assertIsNone(chooser.loaders)
            builder.assert_not_called()

    def test_proxy_cap_is_not_vacuous_for_small_per_step_changes(self):
        raw = dict(new_loss=1., old_logit_shift=1e-10, feature_shift=1e-9)
        unsafe = dict(new_loss=.5, old_logit_shift=2e-10, feature_shift=2e-9)
        safe = dict(new_loss=.8, old_logit_shift=.5e-10, feature_shift=.5e-9)
        self.assertEqual(choose_candidates([raw, unsafe, safe]), dict(oracle=None, logit=2, feature=2))

    def test_audit_is_bitwise_raw_and_preserves_rng_modes_grads_and_heads(self):
        torch.manual_seed(1993)
        net = Network().train()
        net.dropout.eval()
        net.s.grad = torch.ones_like(net.s)
        snapshots = self.snapshots(net)
        state = copy.deepcopy(net.state_dict())
        rng = torch.get_rng_state().clone()
        py, np_state = random.getstate(), np.random.get_state()
        inputs, targets = torch.ones(3, 2), torch.tensor([2, 3, 2])
        row = apply_branch_choice(net, snapshots, inputs, targets, None, 2, 20., 'audit', 'sp')
        self.assertFalse(row['applied'])
        self.assertEqual(row['selected'], 0)
        self.assertTrue(all(torch.equal(v, net.state_dict()[k]) for k, v in state.items()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(py, random.getstate())
        np.testing.assert_equal(np_state, np.random.get_state())
        self.assertTrue(net.training)
        self.assertFalse(net.dropout.training)
        self.assertTrue(torch.equal(net.s.grad, torch.ones_like(net.s)))

    def test_only_oracle_selected_b_can_change_and_reference_is_exact(self):
        net = Network()
        snapshots = self.snapshots(net)
        original = copy.deepcopy(net.state_dict())
        inputs, labels = torch.ones(3, 2), torch.tensor([2, 3, 2])
        with patch('utils.branch_step_choice.choose_candidates',
                   return_value=dict(oracle=1, logit=0, feature=0)):
            row = apply_branch_choice(net, snapshots, inputs, labels,
                                      (inputs, torch.tensor([0, 1, 0])), 2, 20., 'oracle', 'p')
        self.assertTrue(row['applied'])
        self.assertTrue(torch.equal(net.s, original['s']))
        self.assertTrue(torch.equal(net.p, snapshots[1][2]))
        self.assertTrue(all(torch.equal(v, net.state_dict()[k]) for k, v in original.items()
                            if k != 'p'))
        self.assertTrue(all('old_loss' in r for r in row['candidates']))

    def test_failure_restores_exact_raw_parameters(self):
        net = Network()
        original = copy.deepcopy(net.state_dict())
        with patch.object(net, 'extract_vector', side_effect=RuntimeError('probe failed')):
            with self.assertRaisesRegex(RuntimeError, 'probe failed'):
                apply_branch_choice(net, self.snapshots(net), torch.ones(2, 2),
                                    torch.tensor([2, 3]), None, 2, 20., 'feature', 'sp')
        self.assertTrue(all(torch.equal(v, net.state_dict()[k]) for k, v in original.items()))

    def test_default_training_hook_remains_bitwise_sgd(self):
        tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == '_backward_and_step')
        namespace = dict(torch=torch, logging=logging)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<hook>', 'exec'), namespace)
        torch.manual_seed(1993)
        raw, hooked = Network(), Network()
        hooked.load_state_dict(raw.state_dict())
        data, labels = torch.ones(3, 2), torch.tensor([0, 1, 0])
        rng = torch.get_rng_state().clone()
        optimizers = [torch.optim.SGD(n.parameters(), lr=.02, momentum=.9) for n in (raw, hooked)]
        for index, (net, opt) in enumerate(zip((raw, hooked), optimizers)):
            torch.set_rng_state(rng)
            output = net(data)
            loss = F.cross_entropy(output['logits'], labels)
            if index:
                learner = SimpleNamespace(args={}, _cur_task=1, _network=net)
                namespace[node.name](learner, loss, None, opt, output, labels)
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(raw.parameters(), hooked.parameters())))
        for a, b in zip(raw.parameters(), hooked.parameters()):
            self.assertTrue(torch.equal(optimizers[0].state[a]['momentum_buffer'],
                                        optimizers[1].state[b]['momentum_buffer']))

    def test_lifecycle_disables_task0_and_releases_before_ca(self):
        source = (ROOT / 'methods/dlora.py').read_text()
        self.assertIn("self._cur_task > 0 and self.args.get('branch_choice_mode', 'off') != 'off'", source)
        start = source.index('self._train(self.train_loader, self.test_loader)')
        self.assertLess(source.index('self._branch_choice = None', start),
                        source.index('self._compute_class_mean(', start))

    def test_real_attention_audit_hook_matches_raw_trajectory_and_buffers(self):
        from test.test_global_conflict_budget import GlobalBudgetSelectionTests
        from test.test_p_old_gradient_oracle import Samples

        class RealNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = GlobalBudgetSelectionTests._make_attention('layer')
                self.attention.before_task(1)
                self.attention.set_task_and_stage(1, 0)
                self.classifier_pool = torch.nn.ModuleList([
                    torch.nn.Linear(4, 2, bias=False) for _ in range(2)])
                self.classifier_pool[0].requires_grad_(False)
                self.numtask = 2

            def extract_vector(self, inputs):
                return self.attention(inputs[:, None], task=1)[:, 0]

            def interface(self, inputs):
                weights = torch.cat([head.weight for head in self.classifier_pool])
                return F.linear(F.normalize(self.extract_vector(inputs), dim=1),
                                F.normalize(weights, dim=1))

            def forward(self, inputs):
                return {'logits': self.interface(inputs)[:, 2:]}

        class Images(Samples):
            def __getitem__(self, index):
                return index, torch.tensor([float(index) / 48, 1., .2, .7]), int(self.labels[index])

        tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == '_backward_and_step')
        namespace = dict(torch=torch, logging=logging)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<real-hook>', 'exec'), namespace)
        for scope in ('p', 'sp'):
            torch.manual_seed(1993)
            raw, hooked = RealNetwork(), None
            hooked = copy.deepcopy(raw)
            chooser = BranchStepChoice(SimpleNamespace(get_dataset=lambda *a, **kw: Images()),
                                       2, 4, 1993, 8, 'audit', scope)
            inputs, labels = torch.randn(3, 4), torch.tensor([0, 1, 0])
            rng = torch.get_rng_state().clone()
            optimizers = [torch.optim.SGD([p for p in n.parameters() if p.requires_grad],
                                         lr=.02, momentum=.9) for n in (raw, hooked)]
            for index, (net, opt) in enumerate(zip((raw, hooked), optimizers)):
                torch.set_rng_state(rng)
                for batch in range(2):
                    output = net(inputs)
                    loss = F.cross_entropy(output['logits'], labels)
                    if index:
                        learner = SimpleNamespace(args={}, _cur_task=1, _network=net, scale=20.,
                                                  _branch_choice_context=(0, batch + 1, inputs),
                                                  _branch_choice=chooser,
                                                  _iter_lora_modules=lambda: [net.attention])
                        namespace[node.name](learner, loss, None, opt, output, labels)
                    else:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                end_rng = torch.get_rng_state().clone()
                if index == 0:
                    raw_rng = end_rng
                else:
                    self.assertTrue(torch.equal(raw_rng, end_rng))
            self.assertEqual(chooser.counts['sampled'], 2)
            for name, value in raw.state_dict().items():
                self.assertTrue(torch.equal(value, hooked.state_dict()[name]), name)
            for (name, a), (_, b) in zip(raw.named_buffers(), hooked.named_buffers()):
                self.assertTrue(torch.equal(a, b), name)
            for a, b in zip(raw.parameters(), hooked.parameters()):
                if a.requires_grad:
                    self.assertTrue(torch.equal(optimizers[0].state[a]['momentum_buffer'],
                                                optimizers[1].state[b]['momentum_buffer']))

    def test_legal_step_reports_no_privileged_metric_and_keeps_s_at_raw(self):
        manager = SimpleNamespace(get_dataset=lambda *a, **kw: self.fail('old-data read'))
        for mode in ('logit', 'feature'):
            net = Network()
            chooser = BranchStepChoice(manager, 2, 4, 1993, 8, mode, 'p')
            s = net.s.detach().clone()
            with self.assertLogs(level='INFO') as logs:
                chooser.step(net, self.snapshots(net), torch.ones(3, 2),
                             torch.tensor([0, 1, 0]), 20., 1, 0, 1)
            row = json.loads(next(line.split('BranchChoiceStep ', 1)[1] for line in logs.output
                                  if 'BranchChoiceStep ' in line))
            self.assertTrue(all('old_loss' not in candidate for candidate in row['candidates']))
            self.assertIsNone(row['choices']['oracle'])
            self.assertTrue(torch.equal(s, net.s))


if __name__ == '__main__':
    unittest.main()
