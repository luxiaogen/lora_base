import unittest
import ast
import copy
import logging
from pathlib import Path

import torch

from utils.ca_boundary import boundary_weights, temporary_classifier


class BoundaryTests(unittest.TestCase):
    def test_actual_ca_fork_leaves_baseline_trajectory_unchanged(self):
        # Execute the actual method without importing GPU/model dependencies.
        tree = ast.parse(Path('methods/dlora.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Learner')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_stage2_compact_classifier')
        namespace = dict(torch=torch, optim=torch.optim, F=torch.nn.functional, logging=logging,
                         MultivariateNormal=torch.distributions.MultivariateNormal)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-ca>', 'exec'), namespace)

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier_pool = torch.nn.ModuleList([torch.nn.Linear(3, 2, bias=False) for _ in range(2)])

            def forward(self, x, fc_only=False):
                return torch.cat([head(x) for head in self.classifier_pool], dim=1)

        class Learner:
            _stage2_compact_classifier = namespace['_stage2_compact_classifier']

        learner = Learner()
        learner.args = dict(ca_epochs=2, ca_lrate=.01, scale=20)
        learner._network = Net()
        learner._cur_task, learner._known_classes, learner._total_classes = 1, 2, 4
        learner._device, learner.logit_norm, learner.task_sizes = torch.device('cpu'), .1, [2, 2]
        learner._class_means = torch.randn(4, 3)
        learner._class_covs = torch.eye(3).repeat(4, 1, 1) * .1
        initial = copy.deepcopy(learner._network.state_dict())
        outcomes = []
        for fork in (False, True):
            learner._network.load_state_dict(initial)
            torch.manual_seed(1993)
            if fork:
                with temporary_classifier(learner._network):
                    learner._stage2_compact_classifier(2, boundary=True)
            learner._stage2_compact_classifier(2)
            outcomes.append((copy.deepcopy(learner._network.state_dict()), torch.get_rng_state()))
        for name in initial:
            self.assertTrue(torch.equal(outcomes[0][0][name], outcomes[1][0][name]), name)
        self.assertTrue(torch.equal(outcomes[0][1], outcomes[1][1]))

    def test_class_balance_near_ties_and_detach(self):
        logits = torch.tensor([[.5, .5], [1., 0.], [0., 1.], [.49, .5]], requires_grad=True)
        targets = torch.tensor([0, 0, 1, 1])
        w = boundary_weights(logits, targets)
        self.assertFalse(w.requires_grad)
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[3], w[2])
        for c in targets.unique():
            self.assertAlmostEqual(w[targets == c].mean().item(), 1., places=6)
        self.assertTrue(((w >= .5) & (w <= 2)).all())

    def test_extreme_wrong_not_mined(self):
        w = boundary_weights(torch.tensor([[0., 10.], [0., .01]]), torch.zeros(2, dtype=torch.long))
        self.assertLess(w[0], w[1])

    def test_shadow_restore_and_same_random_samples(self):
        net = torch.nn.Module()
        net.classifier_pool = torch.nn.Linear(3, 2)
        net.train()
        original = net.classifier_pool.weight.detach().clone()
        rng = torch.get_rng_state().clone()
        with temporary_classifier(net):
            draws = torch.randn(9)
            net.eval()
            net.classifier_pool.weight.data.add_(1)
            net.classifier_pool.weight.grad = torch.ones_like(original)
        self.assertTrue(net.training)
        self.assertTrue(torch.equal(original, net.classifier_pool.weight))
        self.assertIsNone(net.classifier_pool.weight.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(draws, torch.randn(9)))
