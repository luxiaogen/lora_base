import copy
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from test import test_dualmask_core
from utils.dualmask_core_audit import diagnostic_state, position_diagnostic, round_robin_indices


class Images(torch.utils.data.Dataset):
    labels = np.tile(np.arange(4), 4)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, torch.ones(4) * (index + 1), int(self.labels[index])


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = test_dualmask_core.CorePolicyTests().make(dual_mask_mechanism_audit=True)
        self.classifier = nn.Linear(12, 4)

    def interface(self, images):
        features = self.attention.qkv(images) + self.attention._contrib_from_units(images, 1)
        return self.classifier(features)


class AuditTests(unittest.TestCase):
    def learner(self, network):
        return SimpleNamespace(_network=network, _device=torch.device('cpu'),
            args={'dual_mask_mechanism_audit': True}, _cur_task=1, _known_classes=2,
            _total_classes=4, batch_size=3, _iter_lora_modules=lambda: [network.attention])

    def test_fixed_order_and_sample_caps(self):
        labels = [0, 0, 1, 2, 1, 2, 2]
        self.assertEqual(round_robin_indices(labels, range(3), 5), [0, 2, 3, 1, 4])
        self.assertEqual(round_robin_indices(labels, range(3), 99), [0, 2, 3, 1, 4, 5, 6])

    def test_diagnostic_restores_rng_modes_and_override_on_error(self):
        network = TinyNetwork()
        network.classifier.eval()
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with self.assertRaises(RuntimeError):
            with diagnostic_state(network, [network.attention], torch.device('cpu')):
                random.random()
                np.random.rand()
                torch.randn(2)
                network.attention._core_audit_position = 'permuted'
                raise RuntimeError('test diagnostic failure')
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertTrue(network.training)
        self.assertFalse(network.classifier.training)
        self.assertIsNone(network.attention._core_audit_position)

    def test_audit_does_not_change_the_next_training_step(self):
        first = TinyNetwork()
        with torch.no_grad():
            first.attention.S_lora[1].B_weight.normal_()
            first.attention.P_lora[1].B_weight.normal_()
        second = copy.deepcopy(first)
        before = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO') as log:
            position_diagnostic(self.learner(first), Images(), 1)
        self.assertTrue(any('CorePositionDiagnostic' in line for line in log.output))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for model in (first, second):
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            optimizer.zero_grad()
            model.interface(torch.arange(12).reshape(3, 4).float()).square().mean().backward()
            optimizer.step()
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
