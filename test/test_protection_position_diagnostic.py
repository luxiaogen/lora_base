"""位置诊断必须保持训练状态，且只干预 S，不重新训练 P。"""
import copy
import importlib
import importlib.util
import json
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from test import test_dualmask_core_audit as fixtures
from utils.protect_position import permute_protect_mask


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.qkv = nn.Linear(2, 6, bias=False, dtype=torch.float64)
        nn.init.zeros_(self.block.qkv.weight)
        self.register_buffer('calls', torch.zeros(()))

    def interface(self, x):
        self.calls.add_(1)
        return self.block.qkv(x)[:, :2]


class TwoImages(torch.utils.data.Dataset):
    labels = np.array([0, 1])

    def __getitem__(self, index):
        return index, torch.tensor([[1., 0.], [0., 3.]][index], dtype=torch.float64), index

    def __len__(self):
        return 2


class PositionSignalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('utils.protection_position_diagnostic'),
                             '缺少默认关闭的位置诊断实现')
        self.api = importlib.import_module('utils.protection_position_diagnostic')

    def test_split_is_disjoint_class_balanced_and_deterministic(self):
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 3]
        state = torch.get_rng_state().clone()
        actual = self.api.split_indices(labels, 2, 4)
        self.assertEqual(actual, {'score': [0, 1, 2, 3], 'old': [4, 5, 6, 7], 'new': [8, 11, 9]})
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for a, b in [('score', 'old'), ('score', 'new'), ('old', 'new')]:
            self.assertFalse(set(actual[a]) & set(actual[b]))

    def test_ranked_masks_match_counts_and_zero_residual_keeps_spectrum(self):
        importance = torch.arange(1., 25.).reshape(6, 4)
        reference = torch.zeros_like(importance, dtype=torch.bool)
        reference[1, 3] = reference[3, 3] = reference[5, 3] = True
        fisher = importance.flip(1)
        masks = self.api.position_masks(importance, torch.zeros_like(importance), fisher,
                                        reference, 1993, 0)
        self.assertEqual(tuple(masks), ('W', 'WR', 'WRperm', 'MC', 'Random'))
        self.assertTrue(torch.equal(masks['WR'], reference))
        self.assertTrue(torch.equal(masks['WRperm'], reference))
        self.assertTrue(torch.equal(masks['Random'], permute_protect_mask(reference, 1993, 0)))
        for mask in masks.values():
            self.assertEqual([int(x.sum()) for x in mask.chunk(3)], [1, 1, 1])
        tied = self.api.position_masks(torch.zeros_like(importance), importance * 0,
                                      importance * 0, reference, 1993, 0)
        self.assertEqual(tied['MC'].nonzero().tolist(), [[0, 0], [2, 0], [4, 0]])

    def test_sampled_fisher_averages_per_example_squared_gradients(self):
        model = LinearModel()
        weight = model.block.qkv.weight
        weight.requires_grad_(False)
        weight.grad = torch.ones_like(weight)
        original_grad = weight.grad
        rng = torch.get_rng_state().clone()
        scores = self.api.sampled_fisher(model, [model.block], TwoImages(), [0, 1],
                                        known_classes=2, scale=1., seed=1993)
        expected = torch.tensor([[.125, 1.125], [.125, 1.125], [0, 0], [0, 0], [0, 0], [0, 0]],
                                dtype=torch.float64)
        for draw in scores:
            torch.testing.assert_close(draw[0], expected, rtol=0, atol=0)
        self.assertFalse(weight.requires_grad)
        self.assertIs(weight.grad, original_grad)
        self.assertTrue(torch.equal(weight.grad, torch.ones_like(weight)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(float(model.calls), 0.)

    def test_context_restores_flags_buffers_modes_rng_on_error(self):
        model = LinearModel()
        model.block.eval()
        model.block._position_signal_s_override = None
        random_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            with self.api.preserve_state(model, [model.block]):
                model.calls.add_(7)
                model.block.qkv.weight.requires_grad_(False)
                model.block._position_signal_s_override = torch.ones(2)
                random.random()
                np.random.rand()
                torch.randn(2)
                raise RuntimeError('probe failed')
        self.assertEqual(float(model.calls), 0.)
        self.assertTrue(model.block.qkv.weight.requires_grad)
        self.assertTrue(model.training)
        self.assertFalse(model.block.training)
        self.assertIsNone(model.block._position_signal_s_override)
        self.assertEqual(random.getstate(), random_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_common_min_only_shrinks_and_preserves_zero(self):
        v = torch.tensor([[1., -2.], [3., 4.]])
        masks = {'one': torch.tensor([[1, 0], [0, 0]], dtype=torch.bool),
                 'two': torch.tensor([[0, 0], [0, 1]], dtype=torch.bool)}
        safe, scales = self.api.matched_s_updates(v, masks, .5)
        torch.testing.assert_close(safe['one'].norm(), safe['two'].norm())
        self.assertTrue(all(0 <= float(s) <= 1 for s in scales.values()))
        safe, scales = self.api.matched_s_updates(v * 0, masks, .5)
        self.assertTrue(all(float(s) == 1 for s in scales.values()))
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in safe.values()))

    def network(self):
        model = fixtures.TinyNetwork()
        module = model.attention
        module.args.update(dual_mask_protection_rule='static', dual_mask_permission_mode='asymmetric',
                           dual_mask_position_norm_match='off')
        module.effective_protect_strength = .5
        with torch.no_grad():
            module.S_lora[1].B_weight.normal_(std=.01)
            module.P_lora[1].B_weight.normal_(std=.01)
        return model

    def test_forward_override_is_s_only_and_does_not_change_merge(self):
        model = self.network()
        module = model.attention
        x = torch.randn(3, 4)
        p = module._masked_unit_forward(x, module.P_lora[1], True).detach().clone()
        s = module._masked_unit_forward(x, module.S_lora[1], False).detach().clone()
        with self.api.preserve_state(model, [module]):
            module._position_signal_s_override = torch.zeros_like(module.qkv.weight)
            self.assertEqual(float(module._masked_unit_forward(x, module.S_lora[1], False).abs().sum()), 0.)
            torch.testing.assert_close(module._masked_unit_forward(x, module.P_lora[1], True), p, rtol=0, atol=0)
        torch.testing.assert_close(module._masked_unit_forward(x, module.S_lora[1], False), s, rtol=0, atol=0)
        before = module.qkv(x) + module._contrib_from_units(x, 1)
        module.after_task(1)
        torch.testing.assert_close(before, module.qkv(x), rtol=1e-5, atol=1e-6)
        once = module.qkv.weight.detach().clone()
        module.after_task(1)
        torch.testing.assert_close(once, module.qkv.weight, rtol=0, atol=0)

    def test_complete_diagnostic_preserves_next_step_and_p_state(self):
        model = self.network()
        control = copy.deepcopy(model)
        dataset = fixtures.Images()
        module = model.attention
        masks = self.api.position_masks(module.w0_importance, module.qkv.weight - module.pretrained_weight,
            module.w0_importance.flip(1), module.general_mask.bool(), 1993, 0)
        learner = SimpleNamespace(_network=model, _device=torch.device('cpu'), _cur_task=1,
            _known_classes=2, _total_classes=4, batch_size=3, scale=20.,
            args={'dual_mask_position_signal_audit': True}, _iter_lora_modules=lambda: [module],
            _position_signal_context={'dataset': dataset, 'indices': self.api.split_indices(dataset.labels, 2, 4),
                'layers': [{'masks': masks, 'fisher': module.w0_importance.cpu()}]})
        rng = torch.get_rng_state().clone()
        mask_before = module.general_mask.clone(), module.isolated_mask.clone()
        with self.assertLogs(level='INFO') as logs:
            self.api.report_positions(learner, 2)
        reports = [json.loads(line.split('PositionSignalIntervention ', 1)[1])
                   for line in logs.output if 'PositionSignalIntervention ' in line]
        self.assertEqual(len(reports), 12)
        self.assertEqual({r['position'] for r in reports}, {'W', 'WR', 'WRperm', 'MC', 'Random', 'ActualF'})
        self.assertTrue(all(r['source'] == 'independent_train_diagnostic' for r in reports))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(module.general_mask, mask_before[0], rtol=0, atol=0)
        torch.testing.assert_close(module.isolated_mask, mask_before[1], rtol=0, atol=0)
        for network in (model, control):
            optimizer = torch.optim.SGD(network.parameters(), lr=.01, momentum=.9)
            optimizer.zero_grad()
            network.interface(torch.ones(3, 4)).square().mean().backward()
            optimizer.step()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)

    def test_prepare_uses_task_start_state_and_restores_all_trainable_state(self):
        model = self.network()
        module = model.attention
        with torch.no_grad():
            module.S_lora[1].B_weight.zero_()
            module.P_lora[1].B_weight.zero_()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        flags = [p.requires_grad for p in model.parameters()]
        learner = SimpleNamespace(_network=model, _device=torch.device('cpu'), _cur_task=1,
            _known_classes=2, _total_classes=4, scale=20., args={'seed': 1993},
            _iter_lora_modules=lambda: [module])
        state = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO'):
            self.api.prepare_positions(learner, fixtures.Images())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        context = learner._position_signal_context
        self.assertEqual(len(context['layers']), 1)
        self.assertTrue(all(t.device.type == 'cpu' for t in context['layers'][0]['masks'].values()))

    def test_sampled_fisher_ignores_new_classifier_and_is_seed_repeatable(self):
        model = LinearModel()
        with torch.no_grad():
            model.block.qkv.weight[:2].normal_(std=.1)
        first = self.api.sampled_fisher(model, [model.block], TwoImages(), [0, 1], 2, 1., 1993)
        second = self.api.sampled_fisher(model, [model.block], TwoImages(), [0, 1], 2, 1., 1993)
        for a, b in zip(first, second):
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)

    def test_default_and_task0_match_previous_forward_gradient_step_merge(self):
        import subprocess
        from pathlib import Path
        from torch.nn import functional as F
        code = subprocess.check_output(['git', 'show',
            'f4d12ec7a03a6ea7092e51c5225a2b50decf4181:models/attention.py'], text=True,
            cwd=Path(__file__).resolve().parents[1])
        namespace = {'__name__': 'old_attention'}
        exec(compile(code, '<old_attention>', 'exec'), namespace)
        for task in (0, 1):
            current = self.network().attention
            if task == 0:
                current.before_task(0)
                current._init_lora_weight(0, 0)
                current.set_task_and_stage(0, 0)
            current.cur_task = task
            previous = copy.deepcopy(current)
            previous.__class__ = namespace[current.__class__.__name__]
            x = torch.randn(3, 4)
            state = torch.get_rng_state().clone()
            values = [m._contrib_from_units(x, task) for m in (previous, current)]
            torch.testing.assert_close(*values, rtol=0, atol=0)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            for module, value in zip((previous, current), values):
                value.square().mean().backward()
                optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.02)
                optimizer.step()
                module.after_task(task)
            torch.testing.assert_close(previous.qkv.weight, current.qkv.weight, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
