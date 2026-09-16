import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.p_conflict_diagnostics import (
    conditional_blend_weights,
    conditional_onehot_weights,
    diagnostic_logits,
    evaluate_p_conflict,
    select_top2_counterfactual,
)


class ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.numtask, self.class_num = 2, 2
        self._p_conflict_weights = None

    def interface(self, images):
        if self._p_conflict_weights is None:
            return images
        return images + (self._p_conflict_weights - 1).repeat_interleave(self.class_num, dim=1)


class RepeatedForwardDriftNetwork(ToyNetwork):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def interface(self, images):
        self.calls += 1
        return super().interface(images) + self.calls * 2e-6


class PredictedOnehotTests(unittest.TestCase):
    def setUp(self):
        self.net = ToyNetwork()
        # Row 0: global top class belongs to T0, but summed task probability favors T1.
        self.images = torch.tensor([[4., 0., 3.9, 3.9], [0., 4., 0., 0.], [0., 0., 0., 4.]])

    def test_global_class_winner_not_summed_task_probability(self):
        outputs, weights = diagnostic_logits(self.net, self.images, 1., torch.tensor([0, 1, 1]))
        self.assertEqual(int(weights['soft'][0].argmax()), 1)
        expected = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        torch.testing.assert_close(weights['predicted_onehot'], expected)
        self.assertEqual(outputs['predicted_onehot'].shape, (3, 4))
        self.assertIsNone(self.net._p_conflict_weights)
        # Where the predicted task is correct, hard routing equals oracle exactly.
        torch.testing.assert_close(outputs['predicted_onehot'][[0, 2]], outputs['oracle'][[0, 2]])

    def test_prediction_does_not_read_true_tasks(self):
        a, wa = diagnostic_logits(self.net, self.images, 1., torch.tensor([0, 1, 1]))
        b, wb = diagnostic_logits(self.net, self.images, 1., torch.tensor([1, 0, 0]))
        torch.testing.assert_close(a['predicted_onehot'], b['predicted_onehot'])
        torch.testing.assert_close(wa['predicted_onehot'], wb['predicted_onehot'])
        self.assertFalse(torch.equal(a['oracle'], b['oracle']))

    def test_oracle_low_and_high_margin_partitions_are_complementary(self):
        outputs, weights = diagnostic_logits(
            self.net, self.images, 1., torch.tensor([0, 1, 1]), margin_threshold=0.5)
        torch.testing.assert_close(
            weights['conditional_oracle'],
            torch.tensor([[1., 0.], [1., 1.], [1., 1.]]),
        )
        torch.testing.assert_close(
            weights['high_confidence_oracle'],
            torch.tensor([[1., 1.], [0., 1.], [0., 1.]]),
        )
        self.assertEqual(outputs['conditional_oracle'].shape, self.images.shape)
        self.assertEqual(outputs['high_confidence_oracle'].shape, self.images.shape)

    def test_baseline_mode_reuses_first_forward(self):
        net = RepeatedForwardDriftNetwork()
        ambiguous = torch.zeros_like(self.images)
        outputs, _ = diagnostic_logits(net, ambiguous, 1., torch.tensor([0, 1, 1]))
        torch.testing.assert_close(outputs['ones'], ambiguous + 2e-6, rtol=0, atol=0)
        self.assertEqual(net.calls, 13)

    def test_top2_task_oracle_only_routes_covered_low_margin_samples(self):
        net = ToyNetwork()
        net.numtask, net.class_num = 3, 1
        images = torch.tensor([[2., 1.9, -5.], [2., 1.9, -5.], [8., 0., 0.]])
        outputs, weights = diagnostic_logits(
            net, images, 1., torch.tensor([1, 2, 0]), margin_threshold=0.1)
        torch.testing.assert_close(
            weights['top2_task_oracle'],
            torch.tensor([[0., 1., 0.], [1., 1., 1.], [1., 1., 1.]]),
        )
        self.assertFalse(torch.equal(outputs['top2_task_oracle'][0], outputs['ones'][0]))
        self.assertTrue(torch.equal(outputs['top2_task_oracle'][1:], outputs['ones'][1:]))

    def test_top2_counterfactual_accepts_only_clear_margin_improvement(self):
        baseline = torch.tensor([[0.1, 0.], [2., 0.]])
        candidates = torch.tensor([
            [[0.2, 0.], [0., 2.]],
            [[1., 0.], [0., 0.5]],
        ])
        top2_tasks = torch.tensor([[0, 1], [0, 1]])
        output, accepted, selected_tasks = select_top2_counterfactual(
            baseline, candidates, top2_tasks, scale=1., class_num=1)
        torch.testing.assert_close(output, torch.tensor([[0., 2.], [2., 0.]]))
        self.assertTrue(torch.equal(accepted, torch.tensor([True, False])))
        self.assertTrue(torch.equal(selected_tasks, torch.tensor([1, 0])))

    def test_conditional_onehot_only_changes_low_margin_samples(self):
        task_probs = torch.tensor([[0.55, 0.45], [0.9, 0.1]])
        weights = conditional_onehot_weights(task_probs, torch.tensor([0, 0]), 0.2)
        torch.testing.assert_close(weights, torch.tensor([[1., 0.], [1., 1.]]))
        torch.testing.assert_close(
            conditional_onehot_weights(torch.ones(2, 1), torch.zeros(2, dtype=torch.long), 0.2),
            torch.ones(2, 1),
        )

    def test_conditional_blend_changes_strength_continuously(self):
        task_probs = torch.tensor([[0.5, 0.5], [0.55, 0.45], [0.9, 0.1]])
        predicted_tasks = torch.tensor([0, 0, 0])
        weights = conditional_blend_weights(task_probs, predicted_tasks, 0.2)
        torch.testing.assert_close(weights, torch.tensor([[1., 0.], [1., 0.5], [1., 1.]]))
        torch.testing.assert_close(
            conditional_blend_weights(torch.ones(2, 1), torch.zeros(2, dtype=torch.long), 0.2),
            torch.ones(2, 1),
        )

    def test_conditional_blend_is_label_free_and_preserves_high_margin_logits(self):
        a, wa = diagnostic_logits(self.net, self.images, 1., torch.tensor([0, 1, 1]), 0.5)
        b, wb = diagnostic_logits(self.net, self.images, 1., torch.tensor([1, 0, 0]), 0.5)
        torch.testing.assert_close(a['conditional_blend'], b['conditional_blend'])
        torch.testing.assert_close(wa['conditional_blend'], wb['conditional_blend'])
        high_margin = torch.tensor([[8., 0., 0., 0.]])
        outputs, weights = diagnostic_logits(self.net, high_margin, 1., torch.tensor([0]), 0.1)
        torch.testing.assert_close(weights['conditional_blend'], torch.ones(1, 2))
        self.assertTrue(torch.equal(outputs['conditional_blend'], outputs['ones']))

    def test_conditional_prediction_does_not_read_true_tasks(self):
        a, _ = diagnostic_logits(self.net, self.images, 1., torch.tensor([0, 1, 1]), 0.5)
        b, _ = diagnostic_logits(self.net, self.images, 1., torch.tensor([1, 0, 0]), 0.5)
        torch.testing.assert_close(a['conditional_onehot'], b['conditional_onehot'])
        torch.testing.assert_close(a['top2_counterfactual'], b['top2_counterfactual'])

    def test_top2_counterfactual_preserves_high_margin_samples(self):
        high_margin = torch.tensor([[8., 0., 0., 0.]])
        outputs, weights = diagnostic_logits(self.net, high_margin, 1., torch.tensor([0]), 0.1)
        self.assertTrue(torch.equal(outputs['top2_counterfactual'], outputs['ones']))
        torch.testing.assert_close(weights['top2_counterfactual'], torch.ones(1, 2))

    def test_group_counts_and_correction_accounting(self):
        loader = DataLoader(TensorDataset(torch.arange(3), self.images, torch.tensor([0, 2, 3])), batch_size=2)
        report = evaluate_p_conflict(
            self.net, loader, torch.device('cpu'), 1., margin_threshold=1.0)
        hard = report['predicted_onehot']
        conditional = report['conditional_onehot']
        blend = report['conditional_blend']
        counterfactual = report['top2_counterfactual']
        top2_oracle = report['top2_task_oracle']
        low_oracle = report['conditional_oracle']
        high_oracle = report['high_confidence_oracle']
        groups = hard['by_first_pass_task']
        self.assertEqual(groups['correct']['samples'], 2)
        self.assertEqual(groups['wrong']['samples'], 1)
        self.assertAlmostEqual(hard['first_pass_task_accuracy'], 200 / 3)
        for key in ('corrected', 'broken'):
            self.assertEqual(sum(group[key] for group in groups.values()), hard[key])
        self.assertAlmostEqual(hard['total'] - report['ones']['total'],
                               (hard['corrected'] - hard['broken']) * 100 / 3)
        self.assertEqual(conditional['margin_threshold'], 1.0)
        self.assertEqual(conditional['selected_corrected'], conditional['corrected'])
        self.assertEqual(conditional['selected_broken'], conditional['broken'])
        self.assertEqual(blend['margin_threshold'], 1.0)
        self.assertGreaterEqual(blend['mean_strength'], 0)
        self.assertLessEqual(blend['mean_strength'], 1)
        self.assertEqual(counterfactual['margin_threshold'], 1.0)
        self.assertGreaterEqual(counterfactual['evaluated_samples'], counterfactual['accepted_samples'])
        self.assertEqual(counterfactual['accepted_corrected'], counterfactual['corrected'])
        self.assertEqual(counterfactual['accepted_broken'], counterfactual['broken'])
        self.assertTrue(top2_oracle['oracle_only'])
        self.assertLessEqual(top2_oracle['true_task_in_top2_samples'],
                             top2_oracle['evaluated_samples'])
        self.assertGreaterEqual(top2_oracle['true_task_in_top2_rate'], 0.)
        self.assertLessEqual(top2_oracle['true_task_in_top2_rate'], 100.)
        self.assertTrue(low_oracle['oracle_only'])
        self.assertTrue(high_oracle['oracle_only'])
        self.assertEqual(low_oracle['selected_samples'] + high_oracle['selected_samples'], 3)
        self.assertAlmostEqual(low_oracle['selected_rate'] + high_oracle['selected_rate'], 100.)
        self.assertEqual(set(hard['first_pass_evidence']), {'corrected', 'broken'})
        self.assertEqual(hard['first_pass_evidence']['corrected']['task_margin']['count'], hard['corrected'])
        self.assertEqual(hard['first_pass_evidence']['broken']['task_margin']['count'], hard['broken'])


if __name__ == '__main__':
    unittest.main()
