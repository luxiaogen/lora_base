import copy
import subprocess
import json
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from test.test_protect_position import ProtectPositionTests
from utils.p_permission_release import choose_release, permission_release_gate, probe_state, refresh_release


class ReleaseTests(unittest.TestCase):
    def test_benefit_sign_count_and_private_random_state(self):
        protect = torch.ones(30, 4)
        utility = torch.arange(120).reshape_as(protect).float() - 50
        extra = utility.abs()
        state = torch.get_rng_state().clone()
        masks = [choose_release(utility, extra, protect, mode, 123)
                 for mode in ('benefit', 'random', 'magnitude')]
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for mask in masks:
            for part, scores in zip(mask.chunk(3), utility.chunk(3)):
                self.assertEqual(int(part.sum()), min(4, int((scores > 0).sum())))
            self.assertTrue(torch.all(mask <= protect.bool()))
        self.assertTrue(torch.all(utility[masks[0]] > 0))
        self.assertTrue(torch.equal(masks[1], choose_release(utility, extra, protect, 'random', 123)))
        self.assertFalse(choose_release(-torch.ones_like(utility), extra, protect, 'benefit', 1).any())

    def test_matching_only_shrinks_each_projection_and_zero_keeps_gradients(self):
        delta = torch.arange(48).reshape(12, 4).float() + 1
        protect = (delta.remainder(3) == 0).float()
        release = protect.bool()
        conflict = torch.ones_like(delta) * .7
        gate, scales = permission_release_gate(delta, protect, release, .5, conflict, True)
        target = (delta * (1 - protect) * conflict).reshape(3, -1).norm(dim=1)
        torch.testing.assert_close((delta * gate).reshape(3, -1).norm(dim=1), target)
        self.assertTrue(torch.all(scales <= 1))
        self.assertFalse(scales.requires_grad)
        empty = torch.zeros_like(delta, requires_grad=True)
        gate, scales = permission_release_gate(empty, protect, release, .5, conflict, True)
        self.assertTrue(torch.equal(scales, torch.ones_like(scales)))
        (empty * gate).sum().backward()
        torch.testing.assert_close(empty.grad, ((1 - protect) + .5 * release) * conflict)

    def make(self, **settings):
        module = ProtectPositionTests().make()
        module.args.update(settings)
        module.before_task(1)
        module.set_task_and_stage(1, 2)
        return module

    def test_p_position_changes_only_p_and_fixed_controls_leave_task0_unchanged(self):
        first = self.make()
        second = copy.deepcopy(first)
        second.args['p_permission_position'] = 'permuted'
        second.rebuild_dual_masks()
        value = torch.arange(48).reshape(12, 4).float() + 1
        torch.testing.assert_close(first._safe_delta(value, False), second._safe_delta(value, False), atol=0, rtol=0)
        self.assertFalse(torch.equal(first._p_protect_mask(), second._p_protect_mask()))
        self.assertEqual(int(first.general_mask.sum()), int(second._p_protect_mask().sum()))
        template = ProtectPositionTests().make(task=0)
        candidate = copy.deepcopy(template)
        candidate.args.update(dual_mask_fixed_coverage=.825, dual_mask_fixed_protect_strength=.5,
                              dual_mask_fixed_conflict_strength=.5, p_permission_release='benefit',
                              p_permission_position='permuted')
        state = torch.get_rng_state().clone()
        template.before_task(0)
        torch.set_rng_state(state)
        candidate.before_task(0)
        self.assertEqual(template.effective_protect_strength, candidate.effective_protect_strength)
        self.assertTrue(torch.equal(template.general_mask, candidate.general_mask))
        torch.testing.assert_close(template._contrib_from_units(value[:3, :], 0), candidate._contrib_from_units(value[:3, :], 0))

    def test_regularization_and_merge_use_released_effective_update(self):
        for match in (False, True):
            module = self.make(p_permission_release='benefit', p_permission_norm_match=match)
            module.p_permission_release_mask = module.general_mask.bool().clone()
            with torch.no_grad():
                for unit in (module.S_lora[1], module.P_lora[1]):
                    unit.B_weight.normal_(std=.1)
            unit = module.P_lora[1]
            raw = unit.B_weight @ unit.A_weight
            safe, gate, _ = module._safe_delta(raw, True, return_details=True)
            self.assertGreater(safe[module.general_mask.bool()].abs().sum(), 0)
            loss = module._joint_conflict_regularization(unit, True)
            score, _ = module._joint_conflict(raw)
            expected = (module.w0_importance * safe.square()).mean()
            if module.dual_mask_conflict_reg_enabled:
                expected += (score * safe.square()).mean()
            torch.testing.assert_close(loss, expected)
            x = torch.randn(3, 4)
            before = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
            module.after_task(1)
            torch.testing.assert_close(before, F.linear(x, module.qkv.weight, module.qkv.bias), atol=2e-6, rtol=2e-5)
            self.assertIsNone(module.p_permission_release_mask)
            self.assertEqual(module._contrib_from_units(x, 1).count_nonzero(), 0)

    def test_probe_restores_flags_modes_rng_and_training_step(self):
        module = self.make(p_permission_release='benefit')
        reference = copy.deepcopy(module)
        network = torch.nn.Sequential(module.qkv)
        py_flags = [p.requires_grad for p in module.qkv.parameters()]
        state = torch.get_rng_state().clone()
        with probe_state(network, [module], torch.device('cpu')):
            torch.rand(5)
            module.qkv.weight.requires_grad_(True)
            module._p_release_disabled = True
            network.eval()
            torch.autograd.grad(network(torch.ones(2, 4)).sum(), module.qkv.weight)
        self.assertEqual(py_flags, [p.requires_grad for p in module.qkv.parameters()])
        self.assertTrue(network.training)
        self.assertFalse(module._p_release_disabled)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        x = torch.randn(3, 4)
        for candidate in (module, reference):
            candidate._contrib_from_units(x, 1).sum().backward()
        torch.testing.assert_close(module.P_lora[1].B_weight.grad, reference.P_lora[1].B_weight.grad, atol=0, rtol=0)

    def test_taylor_score_predicts_small_loss_change(self):
        weight = torch.tensor([[.3, -.4]], requires_grad=True)
        inputs = torch.tensor([[1., 2.], [2., 1.]])
        labels = torch.tensor([[.5], [1.]])
        loss = (F.linear(inputs, weight) - labels).square().mean()
        gradient, = torch.autograd.grad(loss, weight)
        extra = -gradient.detach() * .2
        predicted = -(gradient * extra).sum()
        epsilon = 1e-3
        actual = loss.detach() - (F.linear(inputs, weight.detach() + epsilon * extra) - labels).square().mean()
        torch.testing.assert_close(actual / epsilon, predicted, rtol=.002, atol=1e-5)

    def test_off_matches_previous_training_source(self):
        source = subprocess.check_output(['git', 'show',
            '922c189e4fa66bcbc888e626dac90c84fdc51ae4:models/attention.py'], text=True)
        namespace = {'__name__': 'previous_attention'}
        exec(compile(source, 'previous_attention.py', 'exec'), namespace)
        current = ProtectPositionTests().make()
        reference = namespace['Attention_LoRA'](dim=4, num_heads=1, r=2, n_tasks=2)
        reference._init_params(copy.deepcopy(current.args))
        reference.load_state_dict(current.state_dict())
        reference.layer_idx = current.layer_idx
        reference._combined_importance = current._combined_importance
        for task in (0, 1):
            state = torch.get_rng_state().clone()
            reference.before_task(task)
            final_state = torch.get_rng_state().clone()
            torch.set_rng_state(state)
            current.before_task(task)
            self.assertTrue(torch.equal(final_state, torch.get_rng_state()))
            for module in (reference, current):
                module.set_task_and_stage(task, 2)
                with torch.no_grad():
                    module.S_lora[task].B_weight.fill_(.04)
                    module.P_lora[task].B_weight.fill_(.07)
            x = torch.randn(3, 4)
            first, second = reference._contrib_from_units(x, task), current._contrib_from_units(x, task)
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            first.sum().backward()
            second.sum().backward()
            torch.testing.assert_close(reference.S_lora[task].B_weight.grad,
                                       current.S_lora[task].B_weight.grad, atol=0, rtol=0)
            with torch.no_grad():
                reference.after_task(task)
                current.after_task(task)
            torch.testing.assert_close(reference.qkv.weight, current.qkv.weight, atol=0, rtol=0)

    def test_fixed_controls_override_incremental_controller_outputs(self):
        module = ProtectPositionTests().make()
        module.args.update(dual_mask_competence_adaptive=True,
            dual_mask_fixed_coverage=.825, dual_mask_fixed_protect_strength=.5,
            dual_mask_fixed_conflict_strength=.5, dual_mask_private_rank=1)
        module._init_params(module.args)
        module.set_pretrained_competence(.9, .3)
        module.before_task(1)
        self.assertEqual(module.effective_energy_coverage, .825)
        self.assertEqual(module.effective_protect_strength, .5)
        self.assertEqual(module.P_lora[1].A_weight.shape[0], 1)
        self.assertEqual(module._conflict_parameters()[1], .5)

    def test_full_current_train_probe_keeps_parameters_and_optimizer_gradients(self):
        module = self.make(p_permission_release='benefit', seed=1993)
        with torch.no_grad():
            module.P_lora[1].B_weight.normal_(std=.1)
        class Network(torch.nn.Module):
            def __init__(self, attention):
                super().__init__()
                self.attention = attention
                self.head = torch.nn.Linear(12, 20)
            def forward(self, inputs):
                features = self.attention.qkv(inputs) + self.attention._contrib_from_units(inputs, 1)
                return {'logits': self.head(features)}
        class Dataset:
            labels = np.repeat(np.arange(20, 40), 16)
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, index):
                return index, torch.tensor([index % 5, index % 7, 1., index % 3]), self.labels[index]
        network = Network(module)
        learner = SimpleNamespace(_network=network, _device=torch.device('cpu'), _cur_task=1,
            _known_classes=20, _total_classes=40, batch_size=48,
            _iter_lora_modules=lambda: iter([module]),
            args=dict(seed=1993, p_permission_release='benefit', p_permission_norm_match=True))
        before = {name: value.detach().clone() for name, value in network.named_parameters()}
        for parameter in network.parameters():
            parameter.grad = torch.ones_like(parameter)
        grads = [p.grad.clone() for p in network.parameters()]
        rng = torch.get_rng_state().clone()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        with self.assertLogs(level='INFO') as logs:
            refresh_release(learner, Dataset(), 1, F.cross_entropy)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
        for (name, parameter), grad in zip(network.named_parameters(), grads):
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
            torch.testing.assert_close(parameter.grad, grad, atol=0, rtol=0)
        records = [json.loads(line.split('PPermissionRelease ', 1)[1]) for line in logs.output if 'PPermissionRelease ' in line]
        self.assertEqual(records[0]['samples'], 160)
        self.assertIn('PPermissionTrainDiagnostic', '\n'.join(logs.output))
        self.assertFalse(module.qkv.weight.requires_grad)
        self.assertFalse(module.P_lora[1].A_weight.requires_grad)
        self.assertTrue(module.P_lora[1].B_weight.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_real_qkv_energy_gating_forward_merge_and_reconstruction(self):
        from models.attention import Attention_LoRA
        for settings in ({}, dict(p_permission_release='benefit', p_permission_norm_match=True),
                         dict(p_permission_position='permuted')):
            module = Attention_LoRA(dim=768, num_heads=12, qkv_bias=True, r=64, n_tasks=2).cuda()
            module._init_params(dict(use_slora=True, use_plora=True, dual_mask_general_ratio=.4,
                dual_mask_competence_adaptive=False, dual_mask_conflict_energy_adaptive=True,
                dual_mask_conflict_ratio=.1, dual_mask_conflict_strength=.5,
                dual_mask_conflict_granularity='layer', dual_mask_position_audit=True,
                dual_mask_applied_budget_log=True, dual_mask_private_conflict_mode='global',
                dual_mask_task0_gate_mode='unmasked', **settings))
            module.w0_importance.uniform_()
            module.before_task(1)
            module.set_task_and_stage(1, 2)
            if settings.get('p_permission_release'):
                module.p_permission_release_mask = module.general_mask.bool().clone()
            x = torch.randn(2, 5, 768, device='cuda')
            module._contrib_from_units(x, 1).sum().backward()
            self.assertGreater(module.P_lora[1].B_weight.grad.norm(), 0)
            with torch.no_grad():
                for unit in (module.S_lora[1], module.P_lora[1]):
                    unit.B_weight.normal_(std=.01)
                before = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
                module.after_task(1)
                after = F.linear(x, module.qkv.weight, module.qkv.bias)
            torch.testing.assert_close(before, after, atol=5e-5, rtol=5e-5)


if __name__ == '__main__':
    unittest.main()
