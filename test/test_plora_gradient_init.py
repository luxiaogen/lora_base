import ast
import copy
import json
from pathlib import Path
import random
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from utils.plora_gradient_init import gradient_a_basis, initialize_gradient_a, basis_update_metrics


class ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.qkv = nn.Linear(4, 4, bias=False)
        self.attention.qkv.requires_grad_(False)
        self.attention.register_buffer('pretrained_weight', torch.eye(4))
        self.attention.register_buffer('general_mask', torch.zeros(4, 4))
        unit = nn.Module()
        unit.A = nn.Linear(4, 2, bias=False)
        unit.A.requires_grad_(False)
        unit.B = nn.Linear(2, 4, bias=False)
        nn.init.zeros_(unit.B.weight)
        self.attention.P_lora = nn.ModuleList([None, unit])
        self.head = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        unit = self.attention.P_lora[1]
        features = self.attention.qkv(x) + unit.B(unit.A(x))
        return {'logits': self.head(features)}


class GradientBasisTests(unittest.TestCase):
    def test_gradient_selects_useful_space_and_preserves_full_a_gram(self):
        gradient = torch.diag(torch.tensor([4., 3., 2., 1.]))
        original = torch.tensor([[1., 1., 0., 0.], [0., 2., 1., 0.]])
        a, record = gradient_a_basis(gradient, torch.eye(4), torch.zeros(4, 4), original, 'gradient')
        torch.testing.assert_close(a @ a.T, original @ original.T, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(torch.linalg.svdvals(a), torch.linalg.svdvals(original))
        self.assertTrue(torch.equal(a[:, 2:], torch.zeros(2, 2)))
        self.assertAlmostEqual(record['gradient_energy_fraction'], 25 / 30, places=6)

    def test_weight_prior_selects_lower_wpre_and_history_energy(self):
        gradient = torch.diag(torch.tensor([4., 3., 2., 1.]))
        original = torch.ones(1, 4)
        wpre = torch.diag(torch.tensor([10., 0., 0., 0.]))
        residual = torch.diag(torch.tensor([0., 10., 0., 0.]))
        a, record = gradient_a_basis(gradient, wpre, residual, original, 'weight_prior')
        self.assertEqual(a.abs().argmax(dim=1).item(), 2)
        self.assertEqual(record['wpre_energy_fraction'], 0)
        self.assertEqual(record['history_energy_fraction'], 0)
        without_history, _ = gradient_a_basis(gradient, wpre, torch.zeros_like(residual), original, 'weight_prior')
        self.assertEqual(without_history.abs().argmax(dim=1).item(), 1)

    def test_prior_scale_invariance_zero_history_and_no_input_mutation(self):
        gradient = torch.diag(torch.tensor([4., 3., 2., 1.]))
        wpre, history = torch.randn(4, 4), torch.randn(4, 4)
        original = torch.randn(2, 4)
        inputs = [gradient, wpre, history, original]
        copies = [value.clone() for value in inputs]
        a, _ = gradient_a_basis(*inputs, 'weight_prior')
        b, _ = gradient_a_basis(gradient * 3, wpre * 7, history * .3, original, 'weight_prior')
        torch.testing.assert_close(a @ a.T, b @ b.T)
        torch.testing.assert_close(a.T @ a, b.T @ b, atol=2e-5, rtol=2e-5)
        for value, before in zip(inputs, copies):
            self.assertTrue(torch.equal(value, before))
        zero, record = gradient_a_basis(torch.zeros(4, 4), torch.zeros(4, 4), torch.zeros(4, 4), original, 'weight_prior')
        self.assertTrue(torch.isfinite(zero).all())
        self.assertEqual(record['gradient_energy_fraction'], 0)

    def test_random_control_is_exact_original_not_new_qr(self):
        original = torch.randn(2, 4)
        a, _ = gradient_a_basis(torch.randn(4, 4), torch.eye(4), torch.eye(4), original, 'random')
        self.assertTrue(torch.equal(a, original))
        # Basis selection itself does not sample randomness.
        before = torch.get_rng_state().clone()
        gradient_a_basis(torch.eye(4), torch.eye(4), torch.eye(4), original, 'weight_prior')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_probe_restores_rng_modes_gradients_flags_and_changes_only_p_a(self):
        for mode in ('random', 'gradient', 'weight_prior'):
            with self.subTest(mode=mode):
                net = ToyNetwork().train()
                net.attention.eval()
                for param in net.parameters():
                    param.grad = torch.ones_like(param)
                data = TensorDataset(torch.arange(8), torch.randn(8, 4), torch.arange(8) % 2 + 20)
                before = copy.deepcopy(net.state_dict())
                flags = [p.requires_grad for p in net.parameters()]
                grads = [p.grad.clone() for p in net.parameters()]
                py, numpy, rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
                forward = net.forward
                def random_forward(x):
                    random.random()
                    np.random.rand()
                    torch.rand(2)
                    return forward(x)
                net.forward = random_forward
                rows = initialize_gradient_a(net, [net.attention], data, 'cpu', 1, mode,
                    known_classes=20, loss_fn=nn.CrossEntropyLoss(), batch_size=2, batches=2)
                self.assertEqual(rows[0]['images'], 4)
                self.assertFalse(rows[0]['a_trainable'])
                self.assertEqual(rows[0]['b_norm'], 0)
                self.assertEqual(rows[0]['source'], 'current_train_classification_gradient')
                self.assertTrue(net.training)
                self.assertFalse(net.attention.training)
                self.assertEqual(flags, [p.requires_grad for p in net.parameters()])
                for param, grad in zip(net.parameters(), grads):
                    self.assertTrue(torch.equal(param.grad, grad))
                self.assertEqual(random.getstate(), py)
                self.assertTrue(np.array_equal(np.random.get_state()[1], numpy[1]))
                self.assertEqual(np.random.get_state()[2:], numpy[2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                for name, value in net.state_dict().items():
                    if name != 'attention.P_lora.1.A.weight' or mode == 'random':
                        self.assertTrue(torch.equal(value, before[name]), name)

    def test_probe_gradient_is_classification_only_and_plastic_restricted(self):
        net = ToyNetwork()
        unit = net.attention.P_lora[1]
        unit.A, unit.B = nn.Linear(4, 1, bias=False), nn.Linear(1, 4, bias=False)
        unit.A.requires_grad_(False)
        nn.init.zeros_(unit.B.weight)
        net.attention.general_mask[:, 0] = 1
        data = TensorDataset(torch.arange(2), torch.randn(2, 4), torch.tensor([20, 21]))
        rows = initialize_gradient_a(net, [net.attention], data, 'cpu', 1, 'gradient',
            known_classes=20, loss_fn=nn.CrossEntropyLoss(), batch_size=2, batches=1)
        self.assertGreater(rows[0]['masked_gradient_energy_fraction'], 0)
        self.assertLess(rows[0]['masked_gradient_energy_fraction'], 1)
        torch.testing.assert_close(net.attention.P_lora[1].A.weight[:, 0], torch.zeros(1), atol=1e-6, rtol=0)

    def test_probe_failure_restores_state_without_partial_basis_replacement(self):
        net = ToyNetwork().train()
        data = TensorDataset(torch.arange(2), torch.randn(2, 4), torch.arange(2))
        before = copy.deepcopy(net.state_dict())
        rng = torch.get_rng_state().clone()
        def fail(x):
            torch.rand(4)
            raise RuntimeError('probe failed')
        net.forward = fail
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            initialize_gradient_a(net, [net.attention], data, 'cpu', 1, 'gradient',
                known_classes=0, loss_fn=nn.CrossEntropyLoss())
        self.assertTrue(net.training)
        self.assertFalse(net.attention.qkv.weight.requires_grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, value in net.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA server check')
    def test_cuda_probe_preserves_cpu_cuda_rng_and_frozen_state(self):
        for mode in ('random', 'gradient', 'weight_prior'):
            with self.subTest(mode=mode):
                net = ToyNetwork().cuda().train()
                data = TensorDataset(torch.arange(4), torch.randn(4, 4), torch.arange(4) % 2)
                forward = net.forward
                def random_forward(x):
                    torch.rand(3, device='cuda')
                    torch.rand(2)
                    return forward(x)
                net.forward = random_forward
                cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                before = copy.deepcopy(net.state_dict())
                rows = initialize_gradient_a(net, [net.attention], data, 'cuda', 1, mode,
                    known_classes=0, loss_fn=nn.CrossEntropyLoss(), batch_size=2, batches=1)
                self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state()))
                self.assertLess(rows[0]['a_gram_relative_error'], 3e-6)
                self.assertFalse(net.attention.qkv.weight.requires_grad)
                self.assertFalse(net.attention.P_lora[1].A.weight.requires_grad)
                self.assertTrue(net.training)
                for name, value in net.state_dict().items():
                    if name != 'attention.P_lora.1.A.weight' or mode == 'random':
                        self.assertTrue(torch.equal(value, before[name]), name)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA server check')
    def test_cuda_real_dimension_frame_preserves_gram_with_tf32_enabled(self):
        original = torch.randn(39, 768, device='cuda') * .02
        gradient = torch.randn(2304, 768, device='cuda')
        wpre, history = torch.randn_like(gradient), torch.randn_like(gradient)
        before = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            for mode in ('gradient', 'weight_prior'):
                a, record = gradient_a_basis(gradient, wpre, history, original, mode)
                reference = original.double() @ original.double().T
                actual = a.double() @ a.double().T
                relative_error = ((actual - reference).norm() / reference.norm()).item()
                self.assertLess(relative_error, 1e-6)
                self.assertAlmostEqual(record['a_gram_relative_error'], relative_error, places=12)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = before

    def test_effective_update_can_escape_raw_a_space_after_elementwise_gate(self):
        a = torch.tensor([[1., 1.]])
        raw = torch.tensor([[1., 1.], [2., 2.]])
        safe = raw * torch.tensor([[1., 0.], [0., 1.]])
        rows = basis_update_metrics(a, raw, safe, torch.eye(2), torch.zeros(2, 2))
        self.assertLess(rows['raw_space_escape'], 1e-6)
        self.assertGreater(rows['effective_space_escape'], .5)
        self.assertEqual(rows['history_overlap_effective'], 0)
        self.assertGreater(rows['effective_norm'], 0)

    def test_default_and_task0_still_skip_initialization(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == '_initialize_plora_a')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<hook>', 'exec'), namespace)
        hook = namespace[method.name]
        hook(SimpleNamespace(_cur_task=1, args={}))
        hook(SimpleNamespace(_cur_task=0, args={'plora_a_init_mode': 'weight_prior'}))

    def test_real_attention_zero_b_frozen_a_and_merge_consistency(self):
        from models.attention import Attention_LoRA
        from test.test_dual_mask_core import make_args
        for mode in ('random', 'gradient', 'weight_prior'):
            with self.subTest(mode=mode):
                class AttentionNet(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.layer = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
                        self.layer._init_params(make_args(plora_a_init_mode=mode))
                        self.layer.before_task(1)
                        self.layer.set_task_and_stage(1, 0)
                        self.head = nn.Linear(4, 2)
                    def forward(self, x):
                        return {'logits': self.head(self.layer(x, 1)[:, 0])}
                net = AttentionNet().eval()
                x = torch.randn(4, 3, 4)
                data = TensorDataset(torch.arange(4), x, torch.arange(4) % 2)
                output = net(x)['logits'].detach()
                initialize_gradient_a(net, [net.layer], data, 'cpu', 1, mode,
                    known_classes=0, loss_fn=nn.CrossEntropyLoss(), batch_size=2, batches=1)
                torch.testing.assert_close(net(x)['logits'], output, atol=0, rtol=0)
                unit = net.layer.P_lora[1]
                original_a = unit.A.weight.detach().clone()
                optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=.02)
                nn.CrossEntropyLoss()(net(x)['logits'], torch.arange(4) % 2).backward()
                optimizer.step()
                self.assertTrue(torch.equal(unit.A.weight, original_a))
                self.assertGreater(unit.B.weight.norm().item(), 0)
                output = net(x)['logits'].detach()
                net.layer.after_task(1)
                torch.testing.assert_close(net(x)['logits'], output, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
