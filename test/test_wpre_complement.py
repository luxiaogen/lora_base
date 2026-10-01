"""Teacher/student complementarity, matched sampling and scoped real SGD steps."""
import copy
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from test.test_pair_separation import backward_hook


class ComplementUpdateTests(unittest.TestCase):
    def step(self, selection='complement', scope='s', weight=1., correct=True):
        torch.manual_seed(17)
        s = nn.Linear(2, 2, bias=False)
        p = nn.Linear(2, 2, bias=False)
        heads = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(2)])
        with torch.no_grad():
            s.weight.fill_(.02)
            p.weight.fill_(.01)
            heads[0].weight.zero_()
            heads[1].weight.copy_(torch.tensor([[1., 0.], [-1., 0.]]))
            if not correct:
                heads[0].weight[0].copy_(torch.tensor([1., 1.]))
        x = torch.tensor([[2., 1.], [3., 1.], [4., 2.]])
        features = x + s(x) + p(x)
        logits = F.linear(F.normalize(features, dim=1), F.normalize(heads[1].weight, dim=1))
        parameters = [s.weight, p.weight, heads[1].weight]
        optimizer = torch.optim.SGD(parameters, lr=.02, momentum=.9)
        module = SimpleNamespace(S_lora=[None, SimpleNamespace(B_weight=s.weight)],
                                 P_lora=[None, SimpleNamespace(B_weight=p.weight)])
        ridge = torch.tensor([[0., 0., 1., 0.], [0., 0., 1., 0.]])
        learner = SimpleNamespace(args=dict(seed=1993, wpre_distill_weight=weight,
            wpre_distill_selection=selection, wpre_distill_scope=scope,
            wpre_distill_normalization='batch'), _cur_task=1, _known_classes=2,
            _network=SimpleNamespace(classifier_pool=heads, numtask=2),
            _wpre_ridge_weight=ridge, _iter_lora_modules=lambda: [module])
        output = dict(features=features, wpre_teacher_features=torch.ones_like(features))
        before = torch.get_rng_state().clone()
        backward_hook()(learner, F.cross_entropy(logits, torch.zeros(3, dtype=torch.long)),
                        None, optimizer, output, torch.zeros(3, dtype=torch.long))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        return [v.detach().clone() for v in parameters], learner

    def test_student_already_correct_has_no_complement_update(self):
        baseline, _ = self.step(weight=0.)
        candidate, learner = self.step()
        for a, b in zip(baseline, candidate):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(learner._last_wpre_distill_metrics['wpre_selected_ratio'].item(), 0.)

    def test_all_seen_old_head_can_make_current_sample_wrong(self):
        # The current head predicts the correct new class; an old head beats it globally.
        baseline, _ = self.step(weight=0., correct=False)
        candidate, learner = self.step(correct=False)
        self.assertFalse(torch.equal(baseline[0], candidate[0]))
        self.assertTrue(torch.equal(baseline[1], candidate[1]))
        self.assertTrue(torch.equal(baseline[2], candidate[2]))
        self.assertAlmostEqual(learner._last_wpre_distill_metrics['wpre_rescue_ratio'].item(), 2 / 3, places=6)

    def test_only_requested_b_branches_get_auxiliary_update(self):
        baseline, _ = self.step(weight=0., correct=False)
        for scope, changed in [('s', (0,)), ('p', (1,)), ('sp', (0, 1))]:
            candidate, _ = self.step(scope=scope, correct=False)
            for index, (a, b) in enumerate(zip(baseline, candidate)):
                self.assertEqual(not torch.equal(a, b), index in changed)

    def test_random_hook_is_repeatable_and_uses_the_same_rescue_count(self):
        first, learner = self.step(selection='random_matched', correct=False)
        second, _ = self.step(selection='random_matched', correct=False)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(learner._last_wpre_distill_metrics['wpre_selected_ratio'],
                         learner._last_wpre_distill_metrics['wpre_rescue_ratio'])


class ComplementLossTests(unittest.TestCase):
    def fixture(self):
        student = torch.tensor([[0., 1.], [-1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
        teacher = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
        ridge = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.]], requires_grad=True)
        scores = torch.tensor([[2., 0., 1., 0.], [0., 0., 2., 0.],
                               [0., 0., 0., 2.], [0., 0., 2., 0.]], requires_grad=True)
        return student, teacher, ridge, scores

    def test_complement_mask_batch_denominator_and_teacher_detach(self):
        from utils.wpre_distill import selective_feature_loss
        student, teacher, ridge, scores = self.fixture()
        loss, metrics = selective_feature_loss(student, teacher, ridge, torch.tensor([2, 2, 2, 2]),
            selection='complement', student_logits=scores, normalization='batch')
        self.assertAlmostEqual(loss.item(), .5)
        self.assertEqual(metrics['wpre_selected_ratio'].item(), .5)
        self.assertEqual(metrics['wpre_teacher_correct_ratio'].item(), .75)
        loss.backward()
        self.assertTrue(torch.equal(student.grad[1], torch.zeros(2)))
        self.assertTrue(torch.equal(student.grad[3], torch.zeros(2)))
        for value in (teacher, ridge, scores):
            self.assertIsNone(value.grad)

    def test_random_control_matches_count_and_only_selects_teacher_correct(self):
        from utils.wpre_distill import selective_feature_loss
        random.seed(17)
        np.random.seed(17)
        state = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())
        gradients = []
        for _ in range(2):
            student, teacher, ridge, scores = self.fixture()
            loss, metrics = selective_feature_loss(student, teacher, ridge, torch.tensor([2, 2, 2, 2]),
                selection='random_matched', student_logits=scores, normalization='batch',
                generator=torch.Generator().manual_seed(1993))
            loss.backward()
            self.assertEqual(metrics['wpre_selected_ratio'].item(), .5)
            self.assertTrue(torch.equal(student.grad[3], torch.zeros(2)))
            gradients.append(student.grad)
        self.assertTrue(torch.equal(*gradients))
        self.assertEqual(state[0], random.getstate())
        np.testing.assert_array_equal(state[1][1], np.random.get_state()[1])
        self.assertTrue(torch.equal(state[2], torch.get_rng_state()))

    def test_zero_rescue_random_and_complement_are_differentiable_zero(self):
        from utils.wpre_distill import selective_feature_loss
        for selection in ('complement', 'random_matched'):
            student, teacher, ridge, _ = self.fixture()
            scores = torch.tensor([[0., 0., 2., 0.]]).expand(4, -1)
            loss, metrics = selective_feature_loss(student, teacher, ridge, torch.tensor([2, 2, 2, 2]),
                selection=selection, student_logits=scores, normalization='batch',
                generator=torch.Generator().manual_seed(1993))
            loss.backward()
            self.assertEqual(loss.item(), 0.)
            self.assertEqual(metrics['wpre_selected_ratio'].item(), 0.)
            self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

    def test_teacher_correct_batch_control_does_not_rescale_selected_samples(self):
        from utils.wpre_distill import selective_feature_loss
        student, teacher, ridge, scores = self.fixture()
        loss, _ = selective_feature_loss(student, teacher, ridge, torch.tensor([2, 2, 2, 2]),
            selection='teacher_correct', student_logits=scores, normalization='batch')
        self.assertAlmostEqual(loss.item(), 1.)


if __name__ == '__main__':
    unittest.main()
