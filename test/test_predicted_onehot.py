import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.p_conflict_diagnostics import diagnostic_logits, evaluate_p_conflict


class ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.numtask, self.class_num = 2, 2
        self._p_conflict_weights = None

    def interface(self, images):
        if self._p_conflict_weights is None:
            return images
        return images + (self._p_conflict_weights - 1).repeat_interleave(2, dim=1)


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

    def test_group_counts_and_correction_accounting(self):
        loader = DataLoader(TensorDataset(torch.arange(3), self.images, torch.tensor([0, 2, 3])), batch_size=2)
        report = evaluate_p_conflict(self.net, loader, torch.device('cpu'), 1.)
        hard = report['predicted_onehot']
        groups = hard['by_first_pass_task']
        self.assertEqual(groups['correct']['samples'], 2)
        self.assertEqual(groups['wrong']['samples'], 1)
        self.assertAlmostEqual(hard['first_pass_task_accuracy'], 200 / 3)
        for key in ('corrected', 'broken'):
            self.assertEqual(sum(group[key] for group in groups.values()), hard[key])
        self.assertAlmostEqual(hard['total'] - report['ones']['total'],
                               (hard['corrected'] - hard['broken']) * 100 / 3)


if __name__ == '__main__':
    unittest.main()
