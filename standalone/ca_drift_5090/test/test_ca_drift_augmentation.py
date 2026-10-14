"""漂移增广的数量、方向与只读探针回归；使用手算小矩阵。"""
import importlib
import importlib.util
import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class DriftAugmentationTest(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('utils.ca_drift_augmentation'),
                             '漂移增广模块尚未实现')
        return importlib.import_module('utils.ca_drift_augmentation')

    def pairs(self):
        before = torch.zeros(3, 2)
        after = torch.tensor([[-1., -2.], [0., 0.], [1., 2.]])
        return before, after

    def test_covariance_is_centered_not_uncentered_second_moment(self):
        api = self.api()
        before, after = self.pairs()
        expected = torch.tensor([[1., 2.], [2., 4.]], dtype=torch.float64)
        torch.testing.assert_close(api.drift_covariance(before, after + 7.), expected)

    def test_constant_translation_has_zero_covariance(self):
        api = self.api()
        before = torch.tensor([[1., 2.], [3., 4.]])
        torch.testing.assert_close(api.drift_covariance(before, before + 9.),
                                   torch.zeros(2, 2, dtype=torch.float64))

    def test_contraction_is_inflation_candidate_not_true_transport(self):
        api = self.api()
        before = torch.tensor([[-1.], [0.], [1.]])
        torch.testing.assert_close(api.drift_covariance(before, .5 * before),
                                   torch.tensor([[.25]], dtype=torch.float64))

    def test_single_sample_has_zero_centered_covariance(self):
        api = self.api()
        torch.testing.assert_close(api.drift_covariance(torch.zeros(1, 2), torch.ones(1, 2)),
                                   torch.zeros(2, 2, dtype=torch.float64))

    def test_full_adds_shared_increment_without_mutating_input(self):
        api = self.api()
        before, after = self.pairs()
        old = torch.stack([torch.eye(2), 2 * torch.eye(2)])
        original = old.clone()
        updated, audit = api.augment_old_covariances(old, before, after, 'full', 1993)
        torch.testing.assert_close(updated, torch.tensor([[[2., 2.], [2., 5.]],
                                                         [[3., 2.], [2., 6.]]]))
        self.assertTrue(torch.equal(old, original))
        self.assertEqual(updated.dtype, old.dtype)
        self.assertEqual(audit['samples'], 3)
        self.assertEqual(audit['increment_trace'], 5.)

    def test_baseline_probes_but_keeps_old_covariance_bitwise(self):
        api = self.api()
        before, after = self.pairs()
        old = torch.eye(2).unsqueeze(0)
        updated, audit = api.augment_old_covariances(old, before, after, 'baseline', 1993)
        self.assertTrue(torch.equal(updated, old))
        self.assertEqual(audit['drift_cov_trace'], 5.)
        self.assertEqual(audit['increment_trace'], 0.)

    def test_isotropic_matches_trace_not_direction(self):
        api = self.api()
        before, after = self.pairs()
        old = torch.eye(2).unsqueeze(0)
        updated, audit = api.augment_old_covariances(old, before, after, 'isotropic', 1993)
        torch.testing.assert_close(updated, torch.tensor([[[3.5, 0.], [0., 3.5]]]))
        self.assertEqual(audit['increment_trace'], 5.)

    def test_permutation_preserves_spectrum_and_rng(self):
        api = self.api()
        cov = torch.tensor([[1., 2., 0.], [2., 4., 0.], [0., 0., 9.]], dtype=torch.float64)
        torch.manual_seed(22)
        state = torch.get_rng_state().clone()
        permuted = api.covariance_increment(cov, 'permuted', 1993)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(permuted, api.covariance_increment(cov, 'permuted', 1993)))
        torch.testing.assert_close(torch.linalg.eigvalsh(permuted), torch.linalg.eigvalsh(cov))
        self.assertEqual(float(permuted.trace()), 14.)

    def test_collector_restores_all_rng_modes_and_gradients(self):
        api = self.api()

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                self.dropout = torch.nn.Dropout(.5)

            def extract_vector(self, x):
                random.random()
                np.random.random()
                torch.rand(1)
                return self.dropout(self.linear(x))

        network = Encoder().train()
        network.dropout.eval()
        network.linear.weight.grad = torch.ones_like(network.linear.weight)
        inputs = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        loader = DataLoader(TensorDataset(torch.arange(3), inputs, torch.zeros(3)),
                            batch_size=2, shuffle=False)
        random.seed(4)
        np.random.seed(5)
        torch.manual_seed(6)
        py_state, np_state = random.getstate(), np.random.get_state()
        state = torch.get_rng_state().clone()
        weights = network.linear.weight.detach().clone()
        features = api.collect_current_features(network, loader, torch.device('cpu'))
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertTrue(network.training)
        self.assertFalse(network.dropout.training)
        self.assertFalse(features.requires_grad)
        torch.testing.assert_close(features, inputs @ weights.T)
        self.assertTrue(torch.equal(network.linear.weight.grad, torch.ones_like(weights)))

    def test_collector_restores_state_when_feature_extraction_fails(self):
        api = self.api()

        class FailingEncoder(torch.nn.Module):
            def extract_vector(self, inputs):
                random.random()
                np.random.random()
                torch.rand(1)
                raise RuntimeError('deliberate probe failure')

        network = FailingEncoder().train()
        generator = torch.Generator().manual_seed(91)
        loader = DataLoader(TensorDataset(torch.arange(2), torch.zeros(2, 2), torch.zeros(2)),
                            generator=generator)
        py_state, np_state = random.getstate(), np.random.get_state()
        cpu_state, loader_state = torch.get_rng_state().clone(), generator.get_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'deliberate probe failure'):
            api.collect_current_features(network, loader, torch.device('cpu'))
        self.assertTrue(network.training)
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
        self.assertTrue(torch.equal(generator.get_state(), loader_state))

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_collector_restores_cuda_rng(self):
        api = self.api()

        class Encoder(torch.nn.Linear):
            def extract_vector(self, inputs):
                torch.rand(1, device=inputs.device)
                return self(inputs)

        network = Encoder(2, 2).cuda().train()
        loader = DataLoader(TensorDataset(torch.arange(2), torch.ones(2, 2), torch.zeros(2)))
        states = torch.cuda.get_rng_state_all()
        features = api.collect_current_features(network, loader, torch.device('cuda:0'))
        self.assertEqual(features.device.type, 'cpu')
        self.assertTrue(network.training)
        for before, after in zip(states, torch.cuda.get_rng_state_all()):
            self.assertTrue(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
