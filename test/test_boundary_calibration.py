import unittest
import copy

import numpy as np
import torch
from torch.utils.data import TensorDataset

from methods.dlora import fit_boundary_task_logit_bias, task_score_margins
from methods import dlora
from test.test_task_bias_calibration import TinyBiasClassifier


class BoundaryCalibrationTests(unittest.TestCase):
    def test_selects_bottom_ratio_per_task_and_preserves_old_margin(self):
        logits = torch.tensor([
            [0.90, 0.10, 0.20, 0.10],
            [0.51, 0.10, 0.50, 0.10],
            [0.10, 0.20, 0.90, 0.10],
            [0.50, 0.10, 0.51, 0.10],
        ])
        targets = torch.tensor([0, 0, 2, 2])
        before_margin, _, target_tasks = task_score_margins(logits, targets, [2, 2])

        bias, metrics = fit_boundary_task_logit_bias(
            logits, targets, [2, 2], scale=1.0, conflict_ratio=0.5
        )
        class_tasks = torch.tensor([0, 0, 1, 1])
        after_margin, _, _ = task_score_margins(
            logits + bias[class_tasks], targets, [2, 2]
        )
        old = target_tasks == 0

        self.assertEqual(metrics["selected_count"], 2)
        self.assertAlmostEqual(float(metrics["selected_ratio"]), 0.5)
        self.assertGreaterEqual(float(after_margin[old].mean()), float(before_margin[old].mean()) - 1e-5)
        self.assertAlmostEqual(float(bias.mean()), 0.0, places=6)

    def test_unequal_task_sizes_are_supported(self):
        logits = torch.tensor([[0.8, 0.1, 0.7], [0.1, 0.8, 0.9]])
        margins, scores, target_tasks = task_score_margins(
            logits, torch.tensor([0, 2]), [2, 1]
        )
        self.assertEqual(scores.shape, (2, 2))
        self.assertEqual(target_tasks.tolist(), [0, 1])
        self.assertTrue(torch.allclose(margins, torch.tensor([0.1, 0.1]), atol=1e-6))

    def test_symmetric_safety_covers_old_and_current_tasks(self):
        logits = torch.tensor([
            [0.90, 0.10, 0.20, 0.10],
            [0.51, 0.10, 0.50, 0.10],
            [0.10, 0.20, 0.90, 0.10],
            [0.50, 0.10, 0.51, 0.10],
        ])
        targets = torch.tensor([0, 0, 2, 2])

        _, metrics = fit_boundary_task_logit_bias(
            logits,
            targets,
            [2, 2],
            scale=1.0,
            conflict_ratio=0.5,
            symmetric_safe=True,
        )

        self.assertEqual(metrics["safe_sample_count"], 4)
        self.assertEqual(metrics["safe_task_count"], 2)

    def test_stage_changes_only_bias_and_restores_rng(self):
        learner = dlora.Learner.__new__(dlora.Learner)
        learner.args = {'seed': 1993, 'scale': 1.0, 'dual_mask_conflict_ratio': 0.5}
        learner._network = TinyBiasClassifier()
        learner._device = torch.device('cpu')
        learner._multiple_gpus = [learner._device]
        learner._cur_task = 1
        learner._total_classes = 4
        learner.task_sizes = [2, 2]
        learner._class_means = torch.eye(4)
        learner._class_covs = torch.eye(4).repeat(4, 1, 1) * 1e-6
        parameters = copy.deepcopy(learner._network.state_dict())
        torch.manual_seed(13)
        expected_rng = torch.get_rng_state().clone()

        learner._stage3_boundary_calibration(task_size=2)

        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertFalse(torch.equal(learner._network.task_logit_bias, torch.zeros(2)))
        self.assertAlmostEqual(learner._boundary_calibration_metrics['selected_ratio'], 0.5)
        for name, value in parameters.items():
            if name != 'task_logit_bias':
                self.assertTrue(torch.equal(value, learner._network.state_dict()[name]), name)

    def test_real_current_stage_replaces_current_pseudo_features(self):
        learner = dlora.Learner.__new__(dlora.Learner)
        learner.args = {
            'seed': 1993,
            'scale': 1.0,
            'dual_mask_conflict_ratio': 0.5,
            'dual_mask_boundary_real_current': True,
        }
        learner._network = TinyBiasClassifier()
        learner._device = torch.device('cpu')
        learner._multiple_gpus = [learner._device]
        learner._cur_task = 1
        learner._known_classes = 2
        learner._total_classes = 4
        learner.batch_size = 2
        learner.num_workers = 0
        learner.task_sizes = [2, 2]
        learner._class_means = torch.eye(4)
        learner._class_covs = torch.eye(4).repeat(4, 1, 1) * 1e-6
        learner._extract_vectors = lambda loader: (
            np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            np.array([2, 3]),
        )
        parameters = copy.deepcopy(learner._network.state_dict())
        calls = []

        class DataManager:
            def get_dataset(self, classes, source, mode):
                calls.append((classes.tolist(), source, mode))
                return TensorDataset(torch.zeros(2))

        learner._stage3_boundary_calibration(task_size=2, data_manager=DataManager())

        self.assertEqual(calls, [([2, 3], 'train', 'test')])
        self.assertEqual(learner._boundary_calibration_metrics['old_pseudo_count'], 512)
        self.assertEqual(learner._boundary_calibration_metrics['current_real_count'], 2)
        self.assertEqual(learner._boundary_calibration_metrics['calibration_count'], 514)
        self.assertTrue(learner._boundary_calibration_metrics['symmetric_safe'])
        for name, value in parameters.items():
            if name != 'task_logit_bias':
                self.assertTrue(torch.equal(value, learner._network.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
