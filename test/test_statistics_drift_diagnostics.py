import copy
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from methods.dlora import Learner


class DriftDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        vector, target = self.samples[index]
        return index, vector, target


class DriftDataManager:
    def __init__(self, samples):
        self.samples = samples

    def get_dataset(self, class_ids, source, mode):
        selected = set(int(class_id) for class_id in class_ids)
        return DriftDataset([
            (vector, target) for vector, target in self.samples if target in selected
        ])


class TinyFeatureNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(3))

    def interface(self, inputs):
        return inputs @ self.weight.t()

    def extract_vector(self, inputs):
        return inputs


class StatisticsDriftDiagnosticsTests(unittest.TestCase):
    def make_learner(self):
        learner = Learner.__new__(Learner)
        learner._cur_task = 2
        learner._known_classes = 2
        learner._total_classes = 3
        learner.task_sizes = [1, 1, 1]
        learner._class_means = torch.tensor(
            [[1., 0.], [0., 1.], [1., 1.]],
        )
        learner._class_covs = torch.eye(2).repeat(3, 1, 1) * 1e-3
        learner._statistics_margin_baselines = {0: 0.25, 1: 0.50}
        return learner

    def test_summary_reports_drift_margin_drop_and_misrouting(self):
        learner = self.make_learner()
        old_vectors = np.array(
            [[0., 1.], [0., 1.], [0., 1.], [0., 1.]],
        )
        old_targets = np.array([0, 0, 1, 1])
        test_logits = np.array(
            [
                [0.2, 0.6, 0.1],
                [0.4, 0.5, 0.0],
                [0.1, 0.8, 0.2],
                [0.1, 0.5, 0.3],
            ],
        )
        test_targets = np.array([0, 0, 1, 1])

        summary = learner._summarize_statistics_drift(
            old_vectors,
            old_targets,
            test_logits,
            test_targets,
        )

        task0, task1 = summary['tasks']
        self.assertAlmostEqual(task0['mean_cosine_drift'], 1.0)
        self.assertAlmostEqual(task1['mean_cosine_drift'], 0.0)
        self.assertAlmostEqual(task0['covariance_relative_drift'], 0.0)
        self.assertAlmostEqual(task1['covariance_relative_drift'], 0.0)
        self.assertAlmostEqual(task0['current_margin'], -0.25)
        self.assertAlmostEqual(task1['current_margin'], 0.40)
        self.assertAlmostEqual(task0['margin_drop'], 0.50)
        self.assertAlmostEqual(task1['margin_drop'], 0.10)
        self.assertAlmostEqual(task0['misroute_rate'], 100.0)
        self.assertAlmostEqual(task1['misroute_rate'], 0.0)
        self.assertAlmostEqual(summary['correlations']['mean_vs_margin_drop'], 1.0)
        self.assertAlmostEqual(summary['correlations']['mean_vs_misroute'], 1.0)

    def test_margin_baseline_is_recorded_once(self):
        learner = self.make_learner()
        learner._statistics_margin_baselines = {0: 0.25}
        task_metrics = [
            {'task_id': 0, 'current_margin': -0.30, 'misroute_rate': 80.0},
            {'task_id': 1, 'current_margin': 0.50, 'misroute_rate': 10.0},
            {'task_id': 2, 'current_margin': 0.60, 'misroute_rate': 5.0},
        ]

        learner._update_statistics_margin_baselines(task_metrics)
        learner._update_statistics_margin_baselines([
            {'task_id': 0, 'current_margin': -0.10, 'misroute_rate': 70.0},
            {'task_id': 1, 'current_margin': 0.20, 'misroute_rate': 20.0},
            {'task_id': 2, 'current_margin': 0.30, 'misroute_rate': 15.0},
        ])

        self.assertEqual(
            learner._statistics_margin_baselines,
            {0: 0.25, 1: 0.50, 2: 0.60},
        )

    def test_final_diagnostic_is_read_only_and_uses_old_training_samples(self):
        learner = Learner.__new__(Learner)
        learner._cur_task = 2
        learner._known_classes = 2
        learner._total_classes = 3
        learner.total_sessions = 3
        learner.task_sizes = [1, 1, 1]
        learner.batch_size = 2
        learner.num_workers = 0
        learner._device = torch.device('cpu')
        learner._multiple_gpus = [learner._device]
        learner._network = TinyFeatureNetwork()
        learner._network.train()
        learner._class_means = torch.eye(3)
        learner._class_covs = torch.eye(3).repeat(3, 1, 1) * 1e-3
        learner._statistics_margin_baselines = {0: 1.0, 1: 1.0}
        test_samples = [
            (torch.tensor([1., 0., 0.]), 0),
            (torch.tensor([1., 0., 0.]), 0),
            (torch.tensor([0., 1., 0.]), 1),
            (torch.tensor([0., 1., 0.]), 1),
            (torch.tensor([0., 0., 1.]), 2),
            (torch.tensor([0., 0., 1.]), 2),
        ]
        learner.test_loader = DataLoader(DriftDataset(test_samples), batch_size=2)
        data_manager = DriftDataManager(test_samples[:4])
        modes = [module.training for module in learner._network.modules()]
        parameters = copy.deepcopy(learner._network.state_dict())
        random.seed(31); np.random.seed(31); torch.manual_seed(31)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        random.seed(31); np.random.seed(31); torch.manual_seed(31)

        learner._run_statistics_drift_diagnostics(data_manager)

        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertEqual(modes, [module.training for module in learner._network.modules()])
        for name, value in parameters.items():
            self.assertTrue(torch.equal(value, learner._network.state_dict()[name]))
        summary = learner._last_statistics_drift_diagnostics
        self.assertEqual([task['task_id'] for task in summary['tasks']], [0, 1])
        self.assertAlmostEqual(summary['tasks'][0]['mean_cosine_drift'], 0.0)
        self.assertAlmostEqual(summary['tasks'][1]['mean_cosine_drift'], 0.0)


if __name__ == '__main__':
    unittest.main()
