"""直接执行真实任务入口的AST；只替换昂贵的模型训练和图片读取。"""
import ast
import json
import logging
from pathlib import Path
import time
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.ca_drift_augmentation import augment_old_covariances, collect_current_features


def task_entry():
    path = Path(__file__).resolve().parents[1] / 'methods/dlora.py'
    tree = ast.parse(path.read_text())
    learner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Learner')
    method = next(node for node in learner.body if isinstance(node, ast.FunctionDef) and node.name == 'incremental_train')
    namespace = dict(np=np, torch=torch, DataLoader=DataLoader, time=time, json=json, logging=logging,
                     augment_old_covariances=augment_old_covariances,
                     collect_current_features=collect_current_features)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['incremental_train']


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))
        self.numtask = 0

    def advance_task(self):
        self.numtask += 1

    def extract_vector(self, inputs):
        return inputs @ self.weight.T


class Data:
    def get_task_size(self, task):
        return 2

    def get_dataset(self, classes, source, mode):
        return TensorDataset(torch.arange(3), torch.tensor([[-1., -2.], [0., 0.], [1., 2.]]),
                             torch.tensor([classes[0], classes[0], classes[-1]]))


class Harness:
    incremental_train = task_entry()

    def __init__(self, mode, task=1):
        self.args = dict(ca_drift_augmentation=mode, seed=1993)
        self._cur_task, self._known_classes = task - 1, 2 * task
        self._total_classes = self._known_classes
        self._network, self._device = Encoder(), torch.device('cpu')
        self.batch_size, self.num_workers, self.task_sizes = 2, 0, []
        self._class_means = torch.ones(self._known_classes, 2) if task else None
        self._class_covs = torch.eye(2).repeat(self._known_classes, 1, 1) if task else None
        self.events = []

    def _prepare_w0_prototypes(self, loader):
        self.events.append('wpre')

    def _train(self, train_loader, test_loader):
        self.events.append('train-and-merge')
        with torch.no_grad():
            self._network.weight.mul_(2.)
        self.training_draw = torch.rand(3)

    def _measure_pretrained_drift(self, loader):
        self.events.append('legacy-diagnostic')

    def _compute_class_mean(self, manager):
        self.events.append('new-statistics')
        self.old_cov_at_new_stats = self._class_covs.clone() if self._class_covs is not None else None
        if self._class_means is None:
            self._class_means, self._class_covs = torch.ones(2, 2), torch.eye(2).repeat(2, 1, 1)
        else:
            self._class_means = torch.cat([self._class_means, torch.ones(2, 2)])
            self._class_covs = torch.cat([self._class_covs, torch.eye(2).repeat(2, 1, 1)])

    def _stage2_compact_classifier(self, size):
        self.events.append('ca')
        self.ca_draw = torch.rand(3)


class LifecycleTest(unittest.TestCase):
    def test_full_updates_old_cov_before_new_stats_and_ca(self):
        learner = Harness('full')
        learner.incremental_train(Data())
        expected = torch.tensor([[2., 2.], [2., 5.]])
        torch.testing.assert_close(learner.old_cov_at_new_stats[0], expected)
        torch.testing.assert_close(learner._class_covs[-1], torch.eye(2))
        self.assertTrue(torch.equal(learner._class_means, torch.ones(4, 2)))
        self.assertEqual(learner._ca_drift_audit['mode'], 'full')
        self.assertEqual(learner.events[-2:], ['new-statistics', 'ca'])

    def test_baseline_and_off_leave_same_next_training_randomness(self):
        runs = []
        for mode in ('off', 'baseline'):
            torch.manual_seed(55)
            learner = Harness(mode)
            learner.incremental_train(Data())
            runs.append((learner, torch.get_rng_state()))
        off, baseline = runs[0][0], runs[1][0]
        self.assertTrue(torch.equal(off._network.weight, baseline._network.weight))
        self.assertTrue(torch.equal(off._class_covs, baseline._class_covs))
        self.assertTrue(torch.equal(off.training_draw, baseline.training_draw))
        self.assertTrue(torch.equal(off.ca_draw, baseline.ca_draw))
        self.assertTrue(torch.equal(runs[0][1], runs[1][1]))
        self.assertTrue(baseline._ca_drift_audit)

    def test_task0_bypasses_all_new_modes(self):
        for mode in ('off', 'baseline', 'full', 'isotropic', 'permuted'):
            learner = Harness(mode, task=0)
            learner.incremental_train(Data())
            self.assertEqual(getattr(learner, '_ca_drift_audit', {}), {})
            self.assertTrue(torch.equal(learner._class_covs, torch.eye(2).repeat(2, 1, 1)))

    def test_old_increments_accumulate_once_but_newborn_classes_do_not(self):
        learner = Harness('full')
        learner.incremental_train(Data())
        learner._known_classes = learner._total_classes
        learner.incremental_train(Data())
        covariance = torch.tensor([[1., 2.], [2., 4.]])
        torch.testing.assert_close(learner._class_covs[0], torch.eye(2) + 5 * covariance)
        torch.testing.assert_close(learner._class_covs[2], torch.eye(2) + 4 * covariance)
        torch.testing.assert_close(learner._class_covs[-1], torch.eye(2))


if __name__ == '__main__':
    unittest.main()
