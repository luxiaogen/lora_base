import ast
import json
import copy
import random
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from utils.plora_a_init import select_a_basis, initialize_plora_a


class ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Linear(4, 4, bias=False)
        unit = nn.Module()
        unit.A = nn.Linear(4, 2, bias=False)
        unit.B = nn.Linear(2, 4, bias=False)
        unit.A.requires_grad_(False)
        nn.init.zeros_(unit.B.weight)
        self.attention.P_lora = nn.ModuleList([None, unit])

    def extract_vector(self, x, task_id=None):
        return self.attention(x)


class AInitTests(unittest.TestCase):
    def test_principal_direction_energy_and_row_scale(self):
        moment = torch.diag(torch.tensor([1., 9., 3., 20.]))
        original = torch.randn(2, 4)
        before = original.clone()
        basis = select_a_basis(moment, original, 'activation', 1993)
        directions = F.normalize(basis, dim=1)
        torch.testing.assert_close(directions, torch.eye(4)[[3, 1]])
        torch.testing.assert_close(basis.norm(dim=1), original.norm(dim=1))
        self.assertTrue(torch.equal(original, before))

    def test_random_control_is_orthogonal_scaled_reproducible_and_rng_isolated(self):
        original = torch.randn(2, 4)
        rng = torch.get_rng_state().clone()
        moment = torch.eye(4)
        a = select_a_basis(moment, original, 'random_orthogonal', 1993)
        self.assertTrue(torch.equal(a, select_a_basis(moment, original, 'random_orthogonal', 1993)))
        self.assertFalse(torch.equal(a, select_a_basis(moment, original, 'random_orthogonal', 1994)))
        torch.testing.assert_close(F.normalize(a, dim=1) @ F.normalize(a, dim=1).T,
                                   torch.eye(2), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a.norm(dim=1), original.norm(dim=1))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_collection_inputs_only_and_changes_only_private_a(self):
        net = ToyNetwork().train()
        net.attention.eval()
        x = torch.randn(16, 3, 4)
        data = TensorDataset(torch.arange(16), x, torch.arange(16) % 2)
        old = {k: v.clone() for k, v in net.state_dict().items()}
        rng = torch.get_rng_state().clone()
        records = initialize_plora_a(net, [net.attention], data, 'cpu', 1,
                                    'activation', batch_size=4, batches=2, seed=1993)
        self.assertEqual(records[0]['images'], 8)
        self.assertEqual(records[0]['tokens'], 24)
        self.assertEqual(records[0]['rank'], 2)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(net.training)
        self.assertFalse(net.attention.training)
        self.assertFalse(net.attention.P_lora[1].A.weight.requires_grad)
        self.assertEqual(len(net.attention._forward_pre_hooks), 0)
        for name, param in net.state_dict().items():
            if name != 'attention.P_lora.1.A.weight':
                self.assertTrue(torch.equal(param, old[name]), name)
        generator = torch.Generator().manual_seed(1993)
        indices = torch.randperm(16, generator=generator)[:8]
        inputs = x[indices].reshape(-1, 4)
        expected = select_a_basis(inputs.T @ inputs / len(inputs),
                                  old['attention.P_lora.1.A.weight'], 'activation', 1993)
        torch.testing.assert_close(net.attention.P_lora[1].A.weight, expected)

    def test_failure_removes_hooks_and_restores_rng_modes(self):
        net = ToyNetwork().train()
        def fail(*args, **kwargs):
            torch.rand(7)
            raise RuntimeError('test failure')
        net.extract_vector = fail
        data = TensorDataset(torch.arange(2), torch.randn(2, 4), torch.arange(2))
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            initialize_plora_a(net, [net.attention], data, 'cpu', 1, 'activation')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(net.training)
        self.assertEqual(len(net.attention._forward_pre_hooks), 0)

    def test_collection_restores_python_numpy_and_torch_randomness(self):
        net = ToyNetwork()
        data = TensorDataset(torch.arange(2), torch.randn(2, 4), torch.arange(2))
        forward = net.extract_vector
        def random_forward(x, task_id=None):
            random.random()
            np.random.rand()
            torch.rand(2)
            return forward(x, task_id)
        net.extract_vector = random_forward
        py, numpy, rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        initialize_plora_a(net, [net.attention], data, 'cpu', 1, 'random_orthogonal')
        self.assertEqual(random.getstate(), py)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy[1]))
        self.assertEqual(np.random.get_state()[2:], numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_real_attention_zero_b_frozen_a_training_and_merge(self):
        from models.attention import Attention_LoRA
        from test.test_dual_mask_core import make_args
        for mode in ('activation', 'random_orthogonal'):
            with self.subTest(mode=mode):
                net = ToyNetwork()
                net.attention = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
                net.attention._init_params(make_args())
                net.attention.before_task(1)
                net.attention.set_task_and_stage(1, 0)
                net.extract_vector = lambda x, task_id=None: net.attention(x, task_id)
                x = torch.randn(8, 3, 4)
                data = TensorDataset(torch.arange(8), x, torch.arange(8))
                before = copy.deepcopy(net.state_dict())
                initial_output = net.extract_vector(x, task_id=1).detach()
                initialize_plora_a(net, [net.attention], data, 'cpu', 1, mode, batches=1)
                for name, param in net.state_dict().items():
                    if name != 'attention.P_lora.1.A.weight':
                        self.assertTrue(torch.equal(param, before[name]), name)
                torch.testing.assert_close(net.extract_vector(x, task_id=1), initial_output, atol=0, rtol=0)
                unit = net.attention.P_lora[1]
                a = unit.A.weight.detach().clone()
                optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=.02)
                net.extract_vector(x, task_id=1).square().mean().backward()
                optimizer.step()
                self.assertTrue(torch.equal(a, unit.A.weight))
                self.assertGreater(unit.B.weight.norm().item(), 0)
                net.eval()
                output = net.extract_vector(x, task_id=1).detach()
                net.attention.after_task(1)
                torch.testing.assert_close(net.extract_vector(x, task_id=1), output, atol=1e-6, rtol=1e-5)

    def test_default_and_task0_do_not_access_data_or_network(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_initialize_plora_a')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<hook>', 'exec'), namespace)
        hook = namespace[method.name]
        hook(SimpleNamespace(_cur_task=1, args={}))
        hook(SimpleNamespace(_cur_task=0, args={'plora_a_init_mode': 'activation'}))

    def test_hook_runs_after_allocation_before_optimizer(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_train')
        source = ast.unparse(method)
        self.assertLess(source.index('module.set_task_and_stage'), source.index('self._initialize_plora_a()'))
        self.assertLess(source.index('self._initialize_plora_a()'), source.index('self._lora_optimizer_groups'))

    def test_queue_has_only_two_matched_t3_candidates(self):
        spec = json.loads(Path('scripts/sweeps/imgr10_plora_a_init_t3_3090.json').read_text())
        common = spec['common_overrides']
        self.assertEqual([v['overrides'] for v in spec['variants']], [
            {'plora_a_init_mode': 'activation'}, {'plora_a_init_mode': 'random_orthogonal'}])
        for key, value in {'max_tasks': 3, 'dual_mask_anchor_reg_weight': 2.5,
                           'ca_epochs': 5, 'init_epoch': 20, 'epochs': 20,
                           'plora_train_a': False, 'disable_fused_sdpa': True,
                           'ca_stats_transport': False, 'save_task_weights': True}.items():
            self.assertEqual(common[key], value, key)
        self.assertNotIn('data_path', common)
        self.assertNotIn('device', common)
        baseline = json.loads(Path('scripts/sweeps/imgr10_anchor2p5_save_t10_3090.json').read_text())
        expected = {**baseline['common_overrides'], **baseline['variants'][0]['overrides'],
                    'max_tasks': 3, 'wandb_group': spec['name'], 'plora_a_init_batches': 4,
                    'ca_cross_task_margin_weight': 0, 'ca_cov_shrinkage': 0,
                    'ca_stats_transport': False, 'ca_stats_transport_mean_only': False}
        self.assertEqual(common, expected)

    def test_cli_dry_run_normal_and_smoke(self):
        script = Path('scripts/9_28_imgr10_plora_a_init_t3_3090.sh').resolve()
        subprocess.run(['bash', '-n', str(script)], check=True)
        for options, tasks, epochs in (([], '3', '20'), (['--smoke'], '2', '1')):
            output = subprocess.check_output(['bash', str(script), '--dry-run', *options],
                                             cwd='/tmp', text=True)
            commands = [line for line in output.splitlines() if 'main.py --config' in line]
            self.assertEqual(len(commands), 2)
            for command in commands:
                tokens = shlex.split(command)
                settings = dict(tokens[i + 1].split('=', 1) for i, t in enumerate(tokens) if t == '--set')
                self.assertEqual(settings['max_tasks'], tasks)
                self.assertEqual(settings['epochs'], epochs)
                self.assertEqual(settings['seed'], '[1993]')


if __name__ == '__main__':
    unittest.main()
