import copy
import random
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from test.test_p_region_diagnostic import TinyData
from utils.p_conflict_groups import balanced_indices, compare_groups, remove_projection


class GroupNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12, bias=False)
        self._p_conflict_component = torch.ones(12, 4) * 0.1

    def interface(self, x):
        return self.qkv(x).reshape(-1, 3, 4).sum(1)


class GroupTests(unittest.TestCase):
    def test_runner_preserves_protocol_and_json_path(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(['bash', 'scripts/9_20_p_conflict_groups_5090.sh'], cwd=root,
                                    env=dict(os.environ, DRY_RUN='1', LOG_DIR=directory), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('DRYRUN python')]
        self.assertEqual(len(commands), 1)
        tokens = commands[0]
        settings = dict(tokens[i+1].split('=', 1) for i, value in enumerate(tokens) if value == '--set')
        self.assertNotIn('data_path', settings)
        self.assertEqual(settings['seed'], '[1993]')
        self.assertEqual(settings['dual_mask_p_region_train_mode'], 'none')
        self.assertEqual(settings['dual_mask_p_conflict_group_diagnostic'], 'true')
        self.assertEqual(settings['total_sessions'], '10')
        self.assertEqual(settings['ca_epochs'], '5')

    def test_balanced_subset_is_fixed(self):
        labels = np.array([1, 0, 1, 0, 1, 0, 2])
        ids = balanced_indices(labels, 2)
        self.assertEqual(ids, balanced_indices(labels, 2))
        self.assertEqual(sorted(labels[ids].tolist()), [0, 0, 1, 1, 2])

    def test_projection_support_and_exception_restore(self):
        net = GroupNet()
        original = net.qkv.weight.detach().clone()
        for projection in range(3):
            with self.assertRaises(RuntimeError):
                with remove_projection(net, projection):
                    expected = original.clone()
                    expected[projection*4:(projection+1)*4] -= 0.1
                    torch.testing.assert_close(net.qkv.weight, expected)
                    raise RuntimeError('test')
            self.assertTrue(torch.equal(original, net.qkv.weight))

    def test_repeatable_and_read_only(self):
        net = GroupNet()
        net.qkv.eval()
        original = copy.deepcopy(net.state_dict())
        modes = [m.training for m in net.modules()]
        rng = torch.get_rng_state().clone()
        py_rng, np_rng = random.getstate(), np.random.get_state()
        loader = DataLoader(TinyData(), batch_size=2)
        result = compare_groups(net, [net], loader, 'cpu', [2, 2])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(py_rng, random.getstate())
        self.assertTrue(np.array_equal(np_rng[1], np.random.get_state()[1]))
        self.assertEqual(modes, [m.training for m in net.modules()])
        self.assertEqual(result, compare_groups(net, [net], loader, 'cpu', [2, 2]))
        self.assertEqual(len(result['groups']), 3)
        self.assertEqual(result['restored_max_abs_logit_diff'], 0)
        for key, value in original.items():
            self.assertTrue(torch.equal(value, net.state_dict()[key]))

    def test_no_private_component(self):
        net = GroupNet()
        net._p_conflict_component = None
        report = compare_groups(net, [net], DataLoader(TinyData(), batch_size=2), 'cpu', [2, 2])
        self.assertEqual(report['groups'], [])


if __name__ == '__main__':
    unittest.main()
