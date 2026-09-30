import ast
import copy
import json
import logging
from pathlib import Path
import random
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from test import test_global_conflict_budget as global_budget_tests
from utils.p_old_gradient_oracle import (
    OldGradientOracle, apply_oracle_step, build_oracle_loaders, project_old_gradient,
)


ROOT = Path(__file__).resolve().parents[1]


class Samples(Dataset):
    def __init__(self):
        self.labels = np.repeat(np.arange(4), 12)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, torch.tensor([float(index), 1.]), int(self.labels[index])


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(2))
        self.s = torch.nn.Parameter(torch.zeros(2))
        self.head = torch.nn.Parameter(torch.zeros(2))
        self.dropout = torch.nn.Dropout(.2)

    def interface(self, inputs):
        positive = self.dropout(inputs) @ (self.p + self.s + self.head)
        return torch.stack((positive, positive * 0), dim=1)

    def forward(self, inputs):
        return {'logits': self.interface(inputs)}


class OldGradientOracleTests(unittest.TestCase):
    def test_projection_preserves_safe_steps_and_removes_harmful_component(self):
        step = [torch.tensor([.1, .2]), torch.tensor([.3])]
        gradients = [torch.tensor([1., 0.]), torch.tensor([1.])]
        candidate = project_old_gradient(step, gradients)
        self.assertAlmostEqual(sum(float((a * b).sum()) for a, b in zip(candidate, gradients)), 0., places=6)
        self.assertLessEqual(sum(float(x.square().sum()) for x in candidate),
                             sum(float(x.square().sum()) for x in step) + 1e-8)
        for old in ([torch.zeros_like(x) for x in gradients], [-x for x in gradients]):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(step, project_old_gradient(step, old))))

    def test_loaders_read_only_training_data_with_disjoint_old_selector_and_probe(self):
        requests = []
        def get_dataset(classes, source, mode):
            requests.append((list(classes), source, mode))
            return Samples()
        manager = SimpleNamespace(get_dataset=get_dataset)
        rng = torch.get_rng_state().clone()
        loaders = build_oracle_loaders(manager, 2, 4, 1993, 8)
        other = build_oracle_loaders(manager, 2, 4, 1993, 8)
        old, old_probe, new_probe = loaders
        a, b, c = (set(loader.dataset.indices) for loader in loaders)
        self.assertFalse(a & b)
        self.assertFalse((a | b) & c)
        self.assertEqual([len(loader.dataset) for loader in loaders], [8, 8, 8])
        self.assertTrue(all(source == 'train' and mode == 'test' for _, source, mode in requests))
        self.assertTrue(torch.equal(next(iter(old))[0], next(iter(other[0]))[0]))
        self.assertTrue(all(labels < 2 for _, _, batch in old for labels in batch.tolist()))
        self.assertTrue(all(labels >= 2 for _, _, batch in new_probe for labels in batch.tolist()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def setup_step(self):
        net = Network().train()
        net.dropout.eval()
        before = net.p.detach().clone()
        with torch.no_grad():
            net.p.copy_(torch.tensor([.1, .1]))
            net.s.copy_(torch.tensor([.05, -.05]))
        old = (torch.tensor([[-1., 0.]]), torch.tensor([0]))
        new = (torch.tensor([[1., 1.]]), torch.tensor([0]))
        return net, [(net.p, before)], old, new

    def test_actual_loss_check_accepts_jointly_feasible_projection_only_changes_p(self):
        net, snapshots, old, new = self.setup_step()
        s, head = net.s.detach().clone(), net.head.detach().clone()
        reference = net.p.detach().clone()
        opt = torch.optim.SGD(net.parameters(), lr=.1, momentum=.9)
        opt.state[net.p]['momentum_buffer'] = torch.tensor([2., 3.])
        momentum = opt.state[net.p]['momentum_buffer'].clone()
        row = apply_oracle_step(net, snapshots, old, new, F.cross_entropy, 1.)
        self.assertTrue(row['applied'])
        self.assertTrue(row['gradient_conflict'])
        self.assertLessEqual(row['selected']['old_loss'], row['before_p']['old_loss'] + 1e-6)
        self.assertLess(row['selected']['new_loss'], row['before_p']['new_loss'])
        self.assertFalse(torch.equal(net.p, reference))
        self.assertTrue(torch.equal(s, net.s))
        self.assertTrue(torch.equal(head, net.head))
        self.assertTrue(torch.equal(momentum, opt.state[net.p]['momentum_buffer']))
        self.assertTrue(net.training)
        self.assertFalse(net.dropout.training)
        self.assertTrue(all(p.grad is None for p in net.parameters()))

    def test_no_new_descent_falls_back_to_exact_raw_sgd(self):
        net, snapshots, old, _ = self.setup_step()
        original = net.p.detach().clone()
        row = apply_oracle_step(net, snapshots, old, old, F.cross_entropy, 1.)
        self.assertFalse(row['applied'])
        self.assertTrue(torch.equal(original, net.p))

    def test_rng_gradients_modes_and_parameters_restored_on_probe_failure(self):
        net, snapshots, old, new = self.setup_step()
        weights = copy.deepcopy(net.state_dict())
        net.p.grad = torch.ones_like(net.p)
        torch_rng = torch.get_rng_state().clone()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        def failure(*args):
            torch.rand(1)
            random.random()
            np.random.rand()
            raise RuntimeError('oracle probe failed')
        with self.assertRaisesRegex(RuntimeError, 'oracle probe failed'):
            apply_oracle_step(net, snapshots, old, new, failure, 1.)
        self.assertTrue(all(torch.equal(v, net.state_dict()[k]) for k, v in weights.items()))
        self.assertTrue(torch.equal(torch.ones_like(net.p), net.p.grad))
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        np.testing.assert_equal(numpy_rng, np.random.get_state())
        self.assertTrue(net.training)
        self.assertFalse(net.dropout.training)

    def test_training_hook_disabled_is_bitwise_original_sgd(self):
        tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == '_backward_and_step')
        namespace = {'torch': torch, 'logging': logging}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<oracle-hook>', 'exec'), namespace)
        torch.manual_seed(1993)
        base, hooked = Network(), Network()
        hooked.load_state_dict(base.state_dict())
        inputs, targets = torch.randn(7, 2), torch.zeros(7, dtype=torch.long)
        optimizers = [torch.optim.SGD(net.parameters(), lr=.02, momentum=.9) for net in (base, hooked)]
        state = torch.get_rng_state().clone()
        for index, (net, optimizer) in enumerate(zip((base, hooked), optimizers)):
            torch.set_rng_state(state)
            for _ in range(3):
                output = net(inputs)
                loss = F.cross_entropy(output['logits'], targets)
                if index:
                    learner = SimpleNamespace(_network=net, _cur_task=1, args={},
                                              _p_old_gradient_context=None)
                    namespace[method.name](learner, loss, None, optimizer, output, targets)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(base.parameters(), hooked.parameters())))
        for a, b in zip(base.parameters(), hooked.parameters()):
            self.assertTrue(torch.equal(optimizers[0].state[a]['momentum_buffer'],
                                        optimizers[1].state[b]['momentum_buffer']))

    def test_real_dynamic_attention_hook_leaves_s_heads_a_and_momentum_at_raw_sgd(self):
        class ActualNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = global_budget_tests.GlobalBudgetSelectionTests._make_attention('layer')
                self.attention.before_task(1)
                self.attention.set_task_and_stage(1, 0)
                self.heads = torch.nn.ModuleList([torch.nn.Linear(4, 2, bias=False) for _ in range(2)])
                self.heads[0].requires_grad_(False)

            def scores(self, inputs):
                features = F.normalize(self.attention(inputs[:, None], task=1)[:, 0], dim=1)
                return [features @ F.normalize(head.weight, dim=1).T for head in self.heads]

            def interface(self, inputs):
                return torch.cat(self.scores(inputs), dim=1)

            def forward(self, inputs):
                return {'logits': self.scores(inputs)[1]}

        class ImageSamples(Samples):
            def __getitem__(self, index):
                return index, torch.tensor([float(index) / 48, 1., .2, .7]), int(self.labels[index])

        tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == '_backward_and_step')
        namespace = {'torch': torch, 'logging': logging}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<real-oracle-hook>', 'exec'), namespace)
        torch.manual_seed(1993)
        base = ActualNetwork()
        hooked = copy.deepcopy(base)
        inputs, targets = torch.randn(4, 4), torch.tensor([0, 1, 0, 1])
        oracle = OldGradientOracle(SimpleNamespace(get_dataset=lambda *a, **kw: ImageSamples()),
                                   2, 4, 1993, 8)
        before_rng = torch.get_rng_state().clone()
        optimizers = []
        for net in (base, hooked):
            optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=.02, momentum=.9)
            optimizers.append(optimizer)
            output = net(inputs)
            loss = F.cross_entropy(output['logits'], targets)
            if net is hooked:
                learner = SimpleNamespace(_network=net, _cur_task=1, args={}, scale=2.,
                                          _p_old_gradient_context=(0, 1, inputs, F.cross_entropy),
                                          _p_old_gradient_oracle=oracle,
                                          _iter_lora_modules=lambda: [net.attention])
                namespace[method.name](learner, loss, None, optimizer, output, targets)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.assertEqual(oracle.counts['sampled'], 1)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        for (name, raw), (_, selected) in zip(base.named_parameters(), hooked.named_parameters()):
            if name != 'attention.P_lora.1.B.weight':
                self.assertTrue(torch.equal(raw, selected), name)
            if raw.requires_grad:
                self.assertTrue(torch.equal(raw.grad, selected.grad), name)
                self.assertTrue(torch.equal(optimizers[0].state[raw]['momentum_buffer'],
                                            optimizers[1].state[selected]['momentum_buffer']), name)
        self.assertFalse(hooked.attention.P_lora[1].A_weight.requires_grad)

    def test_oracle_pool_lifecycle_skips_task0_and_releases_before_ca(self):
        tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
        incremental = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                           and node.name == 'incremental_train')
        source = ast.unparse(incremental)
        self.assertIn("self._cur_task > 0 and self.args.get('p_old_gradient_oracle', False)", source)
        release = source.index('self._p_old_gradient_oracle = None', source.index('self._train('))
        self.assertLess(release, source.index('self._compute_class_mean'))

    def test_launcher_one_oracle_t3_fixed_recipe_no_checkpoint_or_path(self):
        spec = json.loads((ROOT / 'scripts/sweeps/imgr10_p_old_gradient_oracle_5090.json').read_text())
        common = spec['common_overrides']
        self.assertEqual(spec['seeds'], [1993])
        self.assertEqual(len(spec['variants']), 1)
        self.assertEqual(spec['variants'][0]['overrides'], {'p_old_gradient_oracle': True})
        self.assertEqual(common['max_tasks'], 3)
        self.assertEqual(common['dual_mask_anchor_reg_weight'], 2.5)
        self.assertEqual(common['p_old_gradient_interval'], 5)
        self.assertEqual(common['ca_epochs'], 5)
        self.assertFalse(common['save_task_weights'])
        self.assertEqual(common['p_step_direction'], 'off')
        self.assertTrue(common['disable_fused_sdpa'])
        self.assertEqual(common['p_hard_zero_mode'], 'off')
        self.assertNotIn('data_path', common)
        output = subprocess.check_output(['bash', 'scripts/9_30_imgr10_p_old_gradient_oracle_5090.sh',
                                         '--dry-run'], cwd=ROOT, text=True)
        commands = [shlex.split(line[9:]) for line in output.splitlines() if line.startswith('Command: ')]
        self.assertEqual(len(commands), 1)
        settings = dict(commands[0][i+1].split('=', 1) for i, arg in enumerate(commands[0]) if arg == '--set')
        self.assertEqual(settings['p_old_gradient_oracle'], 'true')
        self.assertEqual(settings['max_tasks'], '3')


if __name__ == '__main__':
    unittest.main()
