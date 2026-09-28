import ast
import copy
import logging
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.ca_stats_transport import fit_diagonal_transport, transport_statistics


class StatsTransportTests(unittest.TestCase):
    def test_5090_is_one_candidate_with_original_anchor5_reference(self):
        spec = json.loads(Path('scripts/sweeps/imgr10_ca_stats_transport_5090.json').read_text())
        self.assertEqual(len(spec['variants']), 1)
        settings = {**spec['common_overrides'], **spec['variants'][0]['overrides']}
        self.assertEqual(settings['dual_mask_anchor_reg_weight'], 5)
        self.assertTrue(settings['ca_stats_transport'])
        self.assertEqual(settings['ca_cross_task_margin_weight'], 0)
        self.assertTrue(settings['save_task_weights'])
        self.assertNotIn('data_path', settings)
        # Compare all fields the archived 5090 anchor5 full-T10 reference specified.
        ref = json.loads(Path('scripts/sweeps/imgr10_head_balance_5090.json').read_text())
        reference = {**ref['common_overrides'], **ref['variants'][0]['overrides']}
        for key, value in reference.items():
            if key != 'wandb_group':
                self.assertEqual(settings[key], value, key)
        for mode, tasks in (([], '10'), (['--smoke'], '2')):
            output = subprocess.check_output(['bash', 'scripts/9_28_imgr10_ca_stats_transport_5090.sh',
                                               '--dry-run', *mode], text=True)
            self.assertEqual(output.count('main.py --config'), 1)
            self.assertIn('--set max_tasks=' + tasks, output)
            self.assertIn('--set ca_stats_transport=true', output)

    def test_actual_statistics_hook_changes_only_existing_arrays(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_transport_ca_statistics')
        ns = dict(logging=logging)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<transport-hook>', 'exec'), ns)
        before = torch.randn(10, 3)
        means, covs = torch.randn(4, 3), torch.eye(3).repeat(4, 1, 1)
        learner = SimpleNamespace(_cur_task=2, _known_classes=4, _class_means=means,
                                  _class_covs=covs, _ca_transport_features=lambda manager, task: before + .2)
        rng = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO') as messages:
            ns[method.name](learner, None, before)
        self.assertEqual(learner._class_means.shape[0], 4)
        torch.testing.assert_close(learner._class_means, means + .2)
        torch.testing.assert_close(learner._class_covs, covs)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIn('CATransport', messages.output[0])

    def test_identity_and_translation_are_exact(self):
        torch.manual_seed(3)
        before = torch.randn(20, 4)
        rng = torch.get_rng_state().clone()
        scale, offset = fit_diagonal_transport(before, before)
        torch.testing.assert_close(scale, torch.ones_like(scale))
        torch.testing.assert_close(offset, torch.zeros_like(offset), atol=1e-12, rtol=0)
        shift = torch.tensor([.1, -.2, .3, .4])
        scale, offset = fit_diagonal_transport(before, before + shift)
        torch.testing.assert_close(scale, torch.ones_like(scale))
        torch.testing.assert_close(offset, shift.double())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_shrunk_scale_is_bounded_and_covariance_stays_positive(self):
        x = torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
        scale, offset = fit_diagonal_transport(x, x * torch.tensor([4., -.2]))
        self.assertTrue(((scale >= .5) & (scale <= 1.5)).all())
        means, covs = torch.randn(3, 2), torch.eye(2).repeat(3, 1, 1)
        original = means.clone(), covs.clone()
        mapped_means, mapped_covs = transport_statistics(means, covs, scale, offset)
        torch.testing.assert_close(mapped_means, means * scale.float() + offset.float())
        torch.testing.assert_close(mapped_covs, torch.diag(scale.float().square()).repeat(3, 1, 1))
        self.assertTrue((torch.linalg.eigvalsh(mapped_covs) > 0).all())
        self.assertTrue(torch.equal(means, original[0]))
        self.assertTrue(torch.equal(covs, original[1]))
        self.assertEqual(mapped_means.dtype, means.dtype)

    def test_constant_features_produce_identity_scale_and_finite_offset(self):
        x = torch.ones(5, 3)
        scale, offset = fit_diagonal_transport(x, x + 2.)
        torch.testing.assert_close(scale, torch.ones_like(scale))
        torch.testing.assert_close(offset, torch.full_like(offset, 2.))

    def test_actual_capture_uses_current_train_and_preserves_network_and_rng(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Learner')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_ca_transport_features')
        ns = dict(torch=torch, np=np, DataLoader=DataLoader)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<capture>', 'exec'), ns)

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.tasks = []

            def extract_vector(self, x, task_id=None):
                self.tasks.append(task_id)
                return self.linear(x)

        net = Net().train()
        dataset = TensorDataset(torch.arange(6), torch.randn(6, 2), torch.tensor([2, 2, 2, 3, 3, 3]))
        calls = []

        def get_dataset(classes, source, mode):
            calls.append((list(classes), source, mode))
            return dataset

        learner = SimpleNamespace(_network=net, _device=torch.device('cpu'), _known_classes=2,
                                  _total_classes=4, _cur_task=1, batch_size=2)
        rng = torch.get_rng_state().clone()
        state = copy.deepcopy(net.state_dict())
        first = ns[method.name](learner, SimpleNamespace(get_dataset=get_dataset), 0)
        second = ns[method.name](learner, SimpleNamespace(get_dataset=get_dataset), 0)
        self.assertEqual(calls, [([2, 3], 'train', 'test')] * 2)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(net.training)
        self.assertEqual(net.tasks, [0] * 6)
        self.assertFalse(first.requires_grad)
        for key in state:
            self.assertTrue(torch.equal(state[key], net.state_dict()[key]))

    def test_lifecycle_is_opt_in_and_only_maps_old_statistics(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'incremental_train')
        source = ast.unparse(method)
        start = source.index('ca_transport_before = self._ca_transport_features')
        self.assertLess(start, source.index('self._train('))
        self.assertLess(source.index('self._train('), source.index('self._transport_ca_statistics('))
        self.assertLess(source.index('self._transport_ca_statistics('), source.index('self._compute_class_mean('))
        self.assertIn("self.args.get('ca_stats_transport', False)", source)
        self.assertIn('self._cur_task > 0', source)


if __name__ == '__main__':
    unittest.main()
