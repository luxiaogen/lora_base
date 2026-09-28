"""Loss semantics and real optimizer updates, without loading a ViT checkpoint."""
import ast
import copy
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F


def backward_hook():
    tree = ast.parse(Path('methods/dlora.py').read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_backward_and_step')
    namespace = {'torch': torch, 'logging': logging}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-training-hook>', 'exec'), namespace)
    return namespace['_backward_and_step']


class PairLossTests(unittest.TestCase):
    def loss(self, features, labels):
        self.assertIsNotNone(importlib.util.find_spec('utils.pair_separation'))
        from utils.pair_separation import pair_separation_loss
        return pair_separation_loss(features, labels, margin=.1)

    def test_mean_positive_hardest_negative_and_valid_denominator(self):
        # Class0 has two orthogonal members and class1 has one singleton.
        # Only the first two anchors are valid: violations 1.1 and .1.
        features = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], requires_grad=True)
        loss, metrics = self.loss(features, torch.tensor([0, 0, 1]))
        self.assertAlmostEqual(loss.item(), .6, places=6)
        self.assertAlmostEqual(metrics['pair_valid_ratio'].item(), 2 / 3, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_multiple_positives_are_averaged_not_maximized(self):
        features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [1., 0.]])
        loss, _ = self.loss(features, torch.tensor([0, 0, 0, 1]))
        # Valid anchor hinges: 1.6, .1, 0. Singleton remains a negative.
        self.assertAlmostEqual(loss.item(), 1.7 / 3, places=6)

    def test_separated_classes_need_no_extra_update(self):
        x = torch.tensor([[1., 0.], [2., 0.], [0., 1.], [0., 2.]], requires_grad=True)
        loss, metrics = self.loss(x, torch.tensor([0, 0, 1, 1]))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(metrics['pair_active_ratio'].item(), 0.)
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))

    def test_no_positive_or_no_negative_has_finite_zero_gradient(self):
        for labels in ([0, 1, 2], [0, 0, 0], [0]):
            x = torch.ones(len(labels), 3, requires_grad=True)
            loss, metrics = self.loss(x, torch.tensor(labels))
            loss.backward()
            self.assertEqual(loss.item(), 0.)
            self.assertEqual(metrics['pair_valid_ratio'].item(), 0.)
            self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))


class PairUpdateTests(unittest.TestCase):
    def run_step(self, scope='p', weight=.05, task=1, diagnostic=False, reference=False,
                 all_loss_reference=False):
        torch.manual_seed(29)
        class Unit(nn.Module):
            def __init__(self):
                super().__init__()
                self.A_weight = nn.Parameter(torch.randn(3, 4), requires_grad=False)
                self.B_weight = nn.Parameter(torch.randn(4, 3) * .2)

        s, p = Unit(), Unit()
        head = nn.Linear(4, 2, bias=False)
        parameters = list(s.parameters()) + list(p.parameters()) + list(head.parameters())
        optimizer = torch.optim.SGD(parameters, lr=.02, momentum=.9)
        module = SimpleNamespace(S_lora=[s, s], P_lora=[None, p])
        learner = SimpleNamespace(args={'pair_separation_weight': weight,
                                        'pair_separation_scope': scope,
                                        'pair_separation_margin': .1}, _cur_task=task,
                                  _iter_lora_modules=lambda: [module],
                                  _pair_separation_diagnostic=(0, 0) if diagnostic else None)
        x = torch.randn(6, 4)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        initial = [param.detach().clone() for param in parameters]
        features = x + x @ (s.B_weight @ s.A_weight).t() + x @ (p.B_weight @ p.A_weight).t()
        task_loss = F.cross_entropy(head(features), labels)
        extra = .01 * (s.B_weight.square().mean() + p.B_weight.square().mean())
        rng = torch.get_rng_state().clone()
        if reference:
            loss = task_loss + extra
            if all_loss_reference:
                from utils.pair_separation import pair_separation_loss
                loss = loss + weight * pair_separation_loss(features, labels, .1)[0]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            loss = backward_hook()(learner, task_loss, extra, optimizer,
                                   {'features': features}, labels)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        return ([v.detach().clone() for v in parameters],
                copy.deepcopy(optimizer.state_dict()), float(loss.detach()), initial, learner)

    def test_p_only_adds_gradient_only_to_p_b(self):
        base, _, _, initial, _ = self.run_step(weight=0)
        candidate, _, _, _, _ = self.run_step()
        # Order is S.A, S.B, P.A, P.B, head.weight.
        for index in (0, 1, 2, 4):
            self.assertTrue(torch.equal(base[index], candidate[index]), index)
        self.assertFalse(torch.equal(base[3], candidate[3]))
        self.assertFalse(torch.equal(candidate[1], initial[1]))  # S still learns CE.
        self.assertFalse(torch.equal(candidate[4], initial[4]))

    def test_sp_adds_gradient_to_both_b_not_a_or_head(self):
        base = self.run_step(weight=0)[0]
        candidate = self.run_step(scope='sp')[0]
        for index in (1, 3):
            self.assertFalse(torch.equal(base[index], candidate[index]), index)
        for index in (0, 2, 4):
            self.assertTrue(torch.equal(base[index], candidate[index]), index)

    def test_off_and_task0_match_original_sgd_and_momentum(self):
        reference = self.run_step(reference=True)
        for kwargs in ({'weight': 0}, {'weight': .05, 'task': 0}):
            result = self.run_step(**kwargs)
            for a, b in zip(reference[0], result[0]):
                self.assertTrue(torch.equal(a, b))
            for index, state in reference[1]['state'].items():
                self.assertTrue(torch.equal(state['momentum_buffer'],
                                            result[1]['state'][index]['momentum_buffer']))
            self.assertEqual(reference[2], result[2])

    def test_diagnostics_do_not_change_updates(self):
        plain = self.run_step()
        with self.assertLogs(level='INFO') as logs:
            measured = self.run_step(diagnostic=True)
        self.assertTrue(any('PairSeparationGrad' in line for line in logs.output))
        for a, b in zip(plain[0], measured[0]):
            self.assertTrue(torch.equal(a, b))

    def test_sp_matches_single_backward_of_weighted_total_loss(self):
        expected = self.run_step(reference=True, all_loss_reference=True)
        actual = self.run_step(scope='sp')
        for a, b in zip(expected[0], actual[0]):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(expected[2], actual[2], places=6)

    def test_real_attention_gates_receive_scoped_auxiliary_gradient(self):
        from test.test_global_conflict_budget import GlobalBudgetSelectionTests
        torch.manual_seed(72)
        attention = GlobalBudgetSelectionTests._make_attention('layer')
        attention.before_task(1)
        attention.set_task_and_stage(1, 0)
        for unit in (attention.S_lora[1], attention.P_lora[1]):
            unit.A_weight.requires_grad_(False)
            with torch.no_grad():
                unit.B_weight.normal_(std=.03)
        head = nn.Linear(12, 2)
        initial = copy.deepcopy(attention.state_dict())
        head_state = copy.deepcopy(head.state_dict())
        x, labels = torch.randn(6, 4), torch.tensor([0, 0, 0, 1, 1, 1])
        finals = []
        for weight in (0., .05):
            attention.load_state_dict(initial)
            head.load_state_dict(head_state)
            opt = torch.optim.SGD(list(attention.parameters()) + list(head.parameters()), lr=.02)
            features = x.repeat(1, 3) + attention._contrib_from_units(x, 1)
            learner = SimpleNamespace(args={'pair_separation_weight': weight, 'pair_separation_scope': 'p'},
                                      _cur_task=1, _iter_lora_modules=lambda: [attention])
            backward_hook()(learner, F.cross_entropy(head(features), labels), None,
                            opt, {'features': features}, labels)
            finals.append((attention.S_lora[1].B_weight.detach().clone(),
                           attention.P_lora[1].B_weight.detach().clone(),
                           head.weight.detach().clone()))
        self.assertTrue(torch.equal(finals[0][0], finals[1][0]))
        self.assertFalse(torch.equal(finals[0][1], finals[1][1]))
        self.assertTrue(torch.equal(finals[0][2], finals[1][2]))


if __name__ == '__main__':
    unittest.main()
