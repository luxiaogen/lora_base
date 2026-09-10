import copy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from methods.base import BaseLearner
from methods.dlora import Learner


class DiagnosticDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        # Deliberately consume RNG to check that diagnostics cannot change CA sampling.
        random.random()
        np.random.rand()
        torch.rand(1)
        return index, torch.eye(4)[index], index


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier_pool = nn.ModuleList([nn.Linear(4, 2, bias=False) for _ in range(2)])
        with torch.no_grad():
            self.classifier_pool[0].weight.copy_(torch.eye(4)[:2])
            self.classifier_pool[1].weight.copy_(torch.eye(4)[2:])

    def forward(self, inputs, fc_only=False):
        return torch.cat([head(inputs) for head in self.classifier_pool], dim=1)

    def interface(self, inputs):
        return self(inputs)


class CADiagnosticsTests(unittest.TestCase):
    def make_learner(self):
        learner = Learner.__new__(Learner)
        learner.args = dict(ca_epochs=1, ca_lrate=0.01, scale=1.0)
        learner._network = TinyClassifier()
        learner._network.classifier_pool[0].eval()
        learner._device = torch.device('cpu')
        learner._multiple_gpus = [learner._device]
        learner._cur_task = 1
        learner._known_classes = 2
        learner._total_classes = 4
        learner.class_num = 2
        learner.topk = 1
        learner.task_sizes = [2, 2]
        learner.logit_norm = None
        learner.acc_matrix = np.zeros((2, 2))
        learner._class_means = torch.eye(4)
        learner._class_covs = torch.eye(4).repeat(4, 1, 1) * 0.1
        learner.test_loader = DataLoader(DiagnosticDataset(), batch_size=2, num_workers=0)
        return learner

    def test_accuracy_is_read_only_and_restores_rng_and_modes(self):
        learner = self.make_learner()
        modes = [m.training for m in learner._network.modules()]
        params = copy.deepcopy(learner._network.state_dict())
        random.seed(19); np.random.seed(19); torch.manual_seed(19)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        random.seed(19); np.random.seed(19); torch.manual_seed(19)
        metrics = learner._measure_ca_accuracy()
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertEqual(metrics['total'], 100.0)
        self.assertEqual(metrics['old'], 100.0)
        self.assertEqual(metrics['new'], 100.0)
        self.assertEqual(metrics['task_prediction'], 100.0)
        self.assertEqual(modes, [m.training for m in learner._network.modules()])
        self.assertFalse(learner.acc_matrix.any())
        for name, value in params.items():
            self.assertTrue(torch.equal(value, learner._network.state_dict()[name]))

    def test_diagnostics_do_not_change_actual_ca_training(self):
        results = []
        for enabled in (False, True):
            random.seed(7); np.random.seed(7); torch.manual_seed(7)
            learner = self.make_learner()
            learner.args['dual_mask_ca_diagnostics'] = enabled
            learner._stage2_compact_classifier(task_size=2)
            results.append((copy.deepcopy(learner._network.state_dict()), torch.get_rng_state()))
            self.assertEqual(getattr(learner, '_ca_before_metrics', None) is not None, enabled)
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        for name, value in results[0][0].items():
            self.assertTrue(torch.equal(value, results[1][0][name]), name)

    def test_post_ca_reuses_regular_evaluation_and_logs_delta(self):
        learner = self.make_learner()
        learner._ca_before_metrics = {'total': 70., 'old': 60., 'new': 80., 'task_prediction': 75.}
        result = ({'grouped': {'total': 75., 'old': 62., 'new': 88.}}, None, None, 0.82)
        with patch.object(BaseLearner, 'eval_task', return_value=result) as evaluate:
            with self.assertLogs(level='INFO') as logs:
                self.assertIs(learner.eval_task(), result)
        evaluate.assert_called_once()
        self.assertIn('delta_total=+5.00', '\n'.join(logs.output))
        self.assertIn('delta_old=+2.00', '\n'.join(logs.output))
        self.assertIn('delta_new=+8.00', '\n'.join(logs.output))
        self.assertIn('before=75.00, after=82.00, delta=+7.00', '\n'.join(logs.output))
        self.assertIsNone(learner._ca_before_metrics)


if __name__ == '__main__':
    unittest.main()
