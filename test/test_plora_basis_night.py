import json
from pathlib import Path
import random
import shlex
import subprocess
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from test.test_plora_gradient_init import ToyNetwork
from utils.plora_gradient_init import gradient_a_basis, initialize_gradient_a


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'scripts/sweeps/imgr10_plora_basis_night_3090.json'
NEW_MODES = ('wpre_prior', 'history_prior', 'wpre_reuse', 'weight_whiten')


class ProbeNetwork(ToyNetwork):
    @property
    def classifier_pool(self):
        return [None, self.head]

    def forward(self, x):
        unit = self.attention.P_lora[1]
        features = self.attention.qkv(x) + unit.B(unit.A(x))
        return {'features': features, 'logits': self.head(features)}


class NightBasisTests(unittest.TestCase):
    def test_priors_have_separate_and_opposite_effects(self):
        g = torch.diag(torch.tensor([4., 3., 2., 1.]))
        a = torch.ones(1, 4)
        pre = torch.diag(torch.tensor([10., 0., 0., 0.]))
        history = torch.diag(torch.tensor([0., 10., 0., 0.]))
        for mode, index in (('wpre_prior', 1), ('history_prior', 0),
                            ('weight_prior', 2), ('weight_whiten', 2)):
            with self.subTest(mode=mode):
                selected, _ = gradient_a_basis(g, pre, history, a, mode)
                self.assertEqual(selected.abs().argmax().item(), index)
        pre = torch.diag(torch.tensor([0., 0., 10., 0.]))
        selected, _ = gradient_a_basis(g, pre, history, a, 'wpre_reuse')
        self.assertEqual(selected.abs().argmax().item(), 2)

    def test_all_new_modes_preserve_gram_rng_and_prior_scale_invariance(self):
        g, pre, history = [torch.randn(5, 4) for _ in range(3)]
        a = torch.randn(2, 4)
        copies = [x.clone() for x in (g, pre, history, a)]
        rng = torch.get_rng_state().clone()
        for mode in NEW_MODES:
            with self.subTest(mode=mode):
                selected, _ = gradient_a_basis(g, pre, history, a, mode)
                scaled, _ = gradient_a_basis(g * 3, pre * 7, history * .3, a, mode)
                torch.testing.assert_close(selected @ selected.T, a @ a.T, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(selected.T @ selected, scaled.T @ scaled, atol=2e-5, rtol=2e-5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for current, before in zip((g, pre, history, a), copies):
            self.assertTrue(torch.equal(current, before))

    def test_whiten_rotates_directions_and_solves_generalized_energy_problem(self):
        g = torch.diag(torch.tensor([2., 1.]))
        pre = torch.tensor([[1., 1.], [0., 0.]])
        a, _ = gradient_a_basis(g, pre, torch.zeros(2, 2), torch.ones(1, 2), 'weight_whiten')
        v = torch.nn.functional.normalize(a[0].double(), dim=0)
        metric = torch.eye(2).double() + 2 * (pre.double().T @ pre.double()) / pre.square().sum()
        h = g.double().T @ g.double()
        value = (v @ h @ v) / (v @ metric @ v)
        torch.testing.assert_close(h @ v, value * (metric @ v), atol=2e-6, rtol=2e-6)
        self.assertGreater(v[1].abs().item(), .1)
        self.assertGreater(value.item(), 2.)

    def test_zero_priors_whiten_reduce_to_gradient_span(self):
        g = torch.diag(torch.tensor([4., 3., 2., 1.]))
        a = torch.randn(2, 4)
        first, _ = gradient_a_basis(g, torch.zeros_like(g), torch.zeros_like(g), a, 'gradient')
        second, _ = gradient_a_basis(g, torch.zeros_like(g), torch.zeros_like(g), a, 'weight_whiten')
        torch.testing.assert_close(first.T @ first, second.T @ second)

    def test_temporary_prototype_head_restores_weights_flags_gradients_and_rng(self):
        for mode in ('gradient', 'weight_prior'):
            with self.subTest(mode=mode):
                net = ProbeNetwork().train()
                net.attention.eval()
                data = TensorDataset(torch.arange(8), torch.randn(8, 4), torch.arange(8) % 2 + 20)
                before = {k: v.clone() for k, v in net.state_dict().items()}
                for p in net.parameters():
                    p.grad = torch.ones_like(p)
                gradients = [p.grad.clone() for p in net.parameters()]
                flags = [p.requires_grad for p in net.parameters()]
                py, numpy, rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
                rows = initialize_gradient_a(net, [net.attention], data, 'cpu', 1, mode,
                    known_classes=20, loss_fn=nn.CrossEntropyLoss(), batch_size=4, batches=2,
                    probe_head='prototype')
                self.assertEqual(rows[0]['probe_head'], 'prototype')
                self.assertEqual(rows[0]['probe_classes_present'], 2)
                self.assertEqual(rows[0]['images'], 8)
                self.assertTrue(0 <= rows[0]['probe_accuracy'] <= 100)
                self.assertFalse(net.attention.P_lora[1].A.weight.requires_grad)
                self.assertEqual(net.attention.P_lora[1].B.weight.norm().item(), 0)
                self.assertEqual(flags, [p.requires_grad for p in net.parameters()])
                self.assertTrue(net.training)
                self.assertFalse(net.attention.training)
                for p, grad in zip(net.parameters(), gradients):
                    self.assertTrue(torch.equal(p.grad, grad))
                for key, value in net.state_dict().items():
                    if key != 'attention.P_lora.1.A.weight':
                        self.assertTrue(torch.equal(value, before[key]), key)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(py, random.getstate())
                self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))

    def test_prototype_probe_failure_restores_original_head(self):
        net = ProbeNetwork().train()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        data = TensorDataset(torch.arange(4), torch.randn(4, 4), torch.arange(4) % 2)
        forward = net.forward
        calls = []
        def fail_after_prototype(x):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError('gradient failed after prototype')
            return forward(x)
        net.forward = fail_after_prototype
        with self.assertRaisesRegex(RuntimeError, 'gradient failed after prototype'):
            initialize_gradient_a(net, [net.attention], data, 'cpu', 1, 'gradient',
                known_classes=0, loss_fn=nn.CrossEntropyLoss(), batch_size=2, batches=2,
                probe_head='prototype')
        for key, value in net.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertTrue(net.training)
        self.assertFalse(net.attention.qkv.weight.requires_grad)

    def test_missing_probe_class_preserves_its_original_head_row(self):
        net = ProbeNetwork()
        data = TensorDataset(torch.arange(2), torch.randn(2, 4), torch.zeros(2, dtype=torch.long))
        row = net.head.weight[1].detach().clone()
        rows = initialize_gradient_a(net, [net.attention], data, 'cpu', 1, 'gradient',
            known_classes=0, loss_fn=nn.CrossEntropyLoss(), batch_size=2, batches=1,
            probe_head='prototype')
        self.assertEqual(rows[0]['probe_classes_present'], 1)
        self.assertTrue(torch.isfinite(net.attention.P_lora[1].A.weight).all())
        self.assertTrue(torch.equal(row, net.head.weight[1]))

    def test_real_attention_new_bases_and_prototype_probe_keep_merge_consistent(self):
        from models.attention import Attention_LoRA
        from test.test_dual_mask_core import make_args
        class Network(nn.Module):
            def __init__(self, mode):
                super().__init__()
                self.layer = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
                self.layer._init_params(make_args(plora_a_init_mode=mode))
                self.layer.before_task(1)
                self.layer.set_task_and_stage(1, 0)
                self.head = nn.Linear(4, 2)
            @property
            def classifier_pool(self):
                return [None, self.head]
            def forward(self, x):
                features = self.layer(x, 1)[:, 0]
                return {'features': features, 'logits': self.head(features)}
        cases = [(mode, 'random') for mode in NEW_MODES] + [
            ('gradient', 'prototype'), ('weight_prior', 'prototype')]
        for mode, head in cases:
            with self.subTest(mode=mode, probe_head=head):
                net = Network(mode).eval()
                x = torch.randn(4, 3, 4)
                data = TensorDataset(torch.arange(4), x, torch.arange(4) % 2)
                before = net(x)['logits'].detach()
                initialize_gradient_a(net, [net.layer], data, 'cpu', 1, mode,
                    known_classes=0, loss_fn=nn.CrossEntropyLoss(), batch_size=2, batches=2,
                    probe_head=head)
                torch.testing.assert_close(net(x)['logits'], before, atol=0, rtol=0)
                a = net.layer.P_lora[1].A.weight.detach().clone()
                optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=.02)
                nn.CrossEntropyLoss()(net(x)['logits'], torch.arange(4) % 2).backward()
                optimizer.step()
                self.assertTrue(torch.equal(a, net.layer.P_lora[1].A.weight))
                before = net(x)['logits'].detach()
                net.layer.after_task(1)
                torch.testing.assert_close(net(x)['logits'], before, atol=1e-6, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA server check')
    def test_cuda_real_size_new_bases_keep_full_gram_with_tf32(self):
        a = torch.randn(39, 768, device='cuda') * .02
        g, pre, history = [torch.randn(2304, 768, device='cuda') for _ in range(3)]
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            for mode in NEW_MODES:
                with self.subTest(mode=mode):
                    selected, row = gradient_a_basis(g, pre, history, a, mode)
                    self.assertLess(row['a_gram_relative_error'], 1e-6)
                    self.assertTrue(torch.isfinite(selected).all())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


class NightQueueTests(unittest.TestCase):
    def test_six_jobs_are_new_and_hold_training_recipe_fixed(self):
        spec = json.loads(SPEC.read_text())
        old = json.loads((ROOT / 'scripts/sweeps/imgr10_plora_weight_basis_t3_3090.json').read_text())
        common = {**old['common_overrides'], 'max_tasks': 10,
                  'wandb_group': 'imgr10_plora_basis_night_t10_3090'}
        self.assertEqual(spec['common_overrides'], common)
        expected = [('wpre_prior', {'plora_a_init_mode': 'wpre_prior'}),
                    ('history_prior', {'plora_a_init_mode': 'history_prior'}),
                    ('wpre_reuse', {'plora_a_init_mode': 'wpre_reuse'}),
                    ('weight_whiten', {'plora_a_init_mode': 'weight_whiten'}),
                    ('prototype_gradient', {'plora_a_init_mode': 'gradient', 'plora_a_probe_head': 'prototype'}),
                    ('prototype_prior', {'plora_a_init_mode': 'weight_prior', 'plora_a_probe_head': 'prototype'})]
        self.assertEqual([(v['name'], v['overrides']) for v in spec['variants']], expected)
        self.assertFalse(common['save_task_weights'])
        self.assertNotIn('data_path', common)
        self.assertFalse(common['plora_train_a'])

    def test_night_dry_run_prints_six_smokes_and_six_full_t10_jobs(self):
        script = ROOT / 'scripts/10_03_imgr10_plora_basis_night_3090.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        output = subprocess.check_output(['bash', str(script), '--mode', 'dry-run'], cwd='/tmp', text=True)
        commands = [shlex.split(line.split('Command:', 1)[1]) for line in output.splitlines()
                    if line.startswith('Command:')]
        self.assertEqual(len(commands), 12)
        for index, command in enumerate(commands):
            settings = dict(token.split('=', 1) for token in command if '=' in token and not token.startswith('--'))
            self.assertEqual(settings['max_tasks'], '2' if index < 6 else '10')
            self.assertEqual(settings['epochs'], '1' if index < 6 else '20')
            self.assertEqual(settings['save_task_weights'], 'false')
            self.assertEqual(settings['seed'], '[1993]')
        output_path = output.split('Outputs:', 1)[1].splitlines()[0].strip()
        self.assertFalse(Path(output_path).exists())

    def test_waiter_updates_only_after_full_current_queue_and_checks_it_completed(self):
        script = ROOT / 'scripts/after_plora_basis_night_3090.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        source = script.read_text()
        self.assertLess(source.index('while kill -0'), source.index('git fetch'))
        self.assertIn("['gradient', 'weight_prior']", source)
        self.assertIn("full_t10_completed", source)
        self.assertNotIn('git reset', source)
        self.assertNotIn('kill -9', source)


if __name__ == '__main__':
    unittest.main()
