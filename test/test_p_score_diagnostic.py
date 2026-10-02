import copy
import json
import random
import unittest

import numpy as np
import torch

from test import test_p_functional_score as functional_tests
from utils.p_score_diagnostic import diagnostic_indices, report_score_counterfactuals
from utils.p_score_diagnostic import MODES


class ToyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = functional_tests.FunctionalIntegrationTests().make_layer('coordinate')
        self.head = torch.nn.Linear(12, 4)

    def interface(self, inputs):
        return self.head((self.layer.qkv(inputs) + self.layer._contrib_from_units(inputs, 1)).mean(1))


class ToyDataset:
    labels = np.repeat(np.arange(4), 4)

    def __getitem__(self, index):
        # Deliberately consume all RNGs to test diagnostic isolation.
        random.random()
        np.random.random()
        return index, torch.randn(3, 4), int(self.labels[index])


class ScoreDiagnosticTests(unittest.TestCase):
    def test_sampling_is_fixed_balanced_and_has_both_partitions(self):
        labels = np.repeat(np.arange(4), 4)
        indices = diagnostic_indices(labels, 2, 4)
        self.assertEqual(len(indices), 8)
        self.assertEqual(int((labels[indices] < 2).sum()), 4)
        self.assertEqual(len(set(indices)), len(indices))
        self.assertEqual(indices, diagnostic_indices(labels, 2, 4))

    def test_report_restores_weights_modes_args_rng_and_actual_gate(self):
        network = ToyNetwork().train()
        network.head.eval()
        args = copy.deepcopy(network.layer.args)
        weights = {key: value.clone() for key, value in network.state_dict().items()}
        torch_state, py_state, np_state = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        with self.assertLogs(level='INFO') as records:
            rows = report_score_counterfactuals(network, ToyDataset(), 'cpu', 1, 2, 4)
        self.assertEqual([row['applied_mode'] for row in rows], list(MODES))
        self.assertEqual(network.layer.args, args)
        self.assertIsNone(network.layer._p_score_diagnostic_gate)
        self.assertTrue(network.training)
        self.assertFalse(network.head.training)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(py_state, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        for key, value in network.state_dict().items():
            self.assertTrue(torch.equal(weights[key], value))
        for row in rows:
            self.assertEqual(row['source'], 'test_report_only_fixed_weights')
            self.assertEqual(row['reference_mode'], 'coordinate')
            for metrics in row['metrics'].values():
                self.assertEqual(set(metrics), {'n', 'accuracy', 'margin', 'corrected', 'broken'})
            self.assertEqual(row['metrics']['old']['n'], 4)
            self.assertEqual(row['metrics']['new']['n'], 4)
            self.assertLess(row['max_relative_removed_norm_error'], 1e-5)
            json.dumps(row, allow_nan=False)
        self.assertTrue(any('PScoreCounterfactual ' in row for row in records.output))

    def test_non_coordinate_reference_is_original_actual_gate(self):
        network = ToyNetwork()
        network.layer.args['p_direction_score'] = 'task_qk'
        with self.assertLogs(level='INFO'):
            rows = report_score_counterfactuals(network, ToyDataset(), 'cpu', 1, 2, 4)
        self.assertTrue(all(row['reference_mode'] == 'task_qk' for row in rows))
        own = next(row for row in rows if row['applied_mode'] == 'task_qk')
        for metrics in own['metrics'].values():
            self.assertEqual(metrics['corrected'], 0)
            self.assertEqual(metrics['broken'], 0)
        self.assertEqual(network.layer.args['p_direction_score'], 'task_qk')

    def test_exception_restores_temporary_gate(self):
        network = ToyNetwork()
        original = network.interface
        def failing(inputs):
            if network.layer.args['p_direction_score'] == 'wpre_input':
                raise RuntimeError('test failure')
            return original(inputs)
        network.interface = failing
        with self.assertRaises(RuntimeError):
            report_score_counterfactuals(network, ToyDataset(), 'cpu', 1, 2, 4)
        self.assertEqual(network.layer.args['p_direction_score'], 'coordinate')
        self.assertIsNone(network.layer._p_score_diagnostic_gate)


if __name__ == '__main__':
    unittest.main()
