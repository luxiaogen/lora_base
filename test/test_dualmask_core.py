import copy
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from utils.dualmask_core import paired_min_gates, permission_gate
import test.test_protect_position as position_tests


class CorePolicyTests(unittest.TestCase):
    def make(self, **args):
        module = position_tests.ProtectPositionTests().make()
        module.args.update(args)
        module.before_task(1)
        module.set_task_and_stage(1, 2)
        return module

    def test_permissions_and_block_norms(self):
        delta = torch.arange(48).reshape(12, 4).float() + 1
        mask = (delta.remainder(3) == 0).float()
        self.assertTrue(torch.equal(permission_gate(mask, .6, True, 'asymmetric'), 1 - mask))
        self.assertTrue(torch.equal(permission_gate(mask, .6, False, 'symmetric_hard'), 1 - mask))
        self.assertTrue(torch.equal(permission_gate(mask, .6, True, 'symmetric_soft'), 1 - .6 * mask))
        gates, norms, target, scales = paired_min_gates(delta, 1 - mask, 1 - mask.flip(1))
        for gate in gates:
            actual = (delta * gate).reshape(3, -1).norm(dim=1)
            torch.testing.assert_close(actual, target)
        self.assertTrue(torch.all(scales <= 1))
        self.assertTrue(torch.equal(gates[0][mask.bool()], torch.zeros_like(gates[0][mask.bool()])))
        self.assertFalse(scales.requires_grad)

    def test_zero_initialization_keeps_gradient(self):
        delta = torch.zeros(12, 4, requires_grad=True)
        first = torch.ones_like(delta)
        second = torch.ones_like(delta) * .5
        gates, _, _, scales = paired_min_gates(delta, first, second)
        self.assertTrue(torch.equal(scales, torch.ones_like(scales)))
        (delta * gates[0]).sum().backward()
        self.assertTrue(torch.equal(delta.grad, first))

    def test_default_and_task0_do_not_change(self):
        original = self.make()
        explicit = copy.deepcopy(original)
        explicit.args.update(dual_mask_position_norm_match='off', dual_mask_permission_mode='asymmetric')
        value = torch.randn_like(original.qkv.weight)
        for isolated in (False, True):
            torch.testing.assert_close(original._safe_delta(value, isolated), explicit._safe_delta(value, isolated), atol=0, rtol=0)
        for module in (original, explicit):
            module.cur_task = 0
            module.dual_mask_task0_gate_mode = 'unmasked'
        explicit.args.update(dual_mask_position_norm_match='paired_min', dual_mask_permission_mode='symmetric_hard')
        torch.testing.assert_close(original._safe_delta(value, False), explicit._safe_delta(value, False), atol=0, rtol=0)

    def test_real_zero_b_gradient_and_freezing(self):
        module = self.make(dual_mask_position_norm_match='paired_min', dual_mask_mechanism_audit=True)
        x = torch.randn(3, 4)
        module._contrib_from_units(x, 1).square().sum().backward()
        # Squared zero output has zero gradient; a linear objective tests the startup path.
        module.zero_grad()
        module._contrib_from_units(x, 1).sum().backward()
        for unit in (module.S_lora[1], module.P_lora[1]):
            self.assertFalse(unit.A_weight.requires_grad)
            self.assertIsNone(unit.A_weight.grad)
            self.assertGreater(unit.B_weight.grad.norm().item(), 0)

    def test_permissions_really_change_effective_update_and_reg_uses_same_gate(self):
        for mode in ('asymmetric', 'symmetric_soft', 'symmetric_hard'):
            module = self.make(dual_mask_permission_mode=mode, dual_mask_position_norm_match='paired_min',
                dual_mask_mechanism_audit=True)
            module.dual_mask_s_conflict_enabled = False
            module.dual_mask_p_conflict_enabled = False
            module.dual_mask_conflict_reg_enabled = False
            strength = module.effective_protect_strength
            for isolated in (False, True):
                delta = (torch.arange(48).reshape(12, 4).float() + 1).requires_grad_()
                gates = [permission_gate(mask, strength, isolated, mode) for mask in
                         (module.core_reference_protect, module.core_permuted_protect)]
                gates, _, _, _ = paired_min_gates(delta, *gates)
                expected = delta * gates[0]
                actual = module._safe_delta(delta, isolated)
                torch.testing.assert_close(actual, expected)
                regularizer = module._joint_conflict_regularization(SimpleNamespace(A_weight=torch.eye(4), B_weight=delta), isolated)
                torch.testing.assert_close(regularizer, (module.w0_importance * expected.square()).mean())
                if mode == 'symmetric_hard' or (mode == 'asymmetric' and isolated):
                    self.assertEqual(actual[module.general_mask.bool()].count_nonzero().item(), 0)
                else:
                    self.assertGreater(actual[module.general_mask.bool()].count_nonzero().item(), 0)

    def test_permissions_have_identical_initialization_and_parameter_counts(self):
        template = position_tests.ProtectPositionTests().make()
        rng = torch.get_rng_state().clone()
        states = []
        for mode in ('asymmetric', 'symmetric_soft', 'symmetric_hard'):
            torch.set_rng_state(rng)
            module = copy.deepcopy(template)
            module.args.update(dual_mask_permission_mode=mode, dual_mask_mechanism_audit=True)
            module.before_task(1)
            module.set_task_and_stage(1, 2)
            states.append((torch.get_rng_state().clone(), {n: (p.detach().clone(), p.requires_grad) for n, p in module.named_parameters()}))
        for state, params in states[1:]:
            self.assertTrue(torch.equal(state, states[0][0]))
            self.assertEqual(params.keys(), states[0][1].keys())
            for name, (value, trainable) in params.items():
                self.assertTrue(torch.equal(value, states[0][1][name][0]), name)
                self.assertEqual(trainable, states[0][1][name][1], name)

    def test_norm_control_removal_not_called_conflict_removal(self):
        from utils.protect_position import update_rows
        raw = torch.ones(12, 4)
        rows = update_rows(raw, raw, raw * .5, torch.zeros_like(raw), torch.zeros_like(raw),
                           branch='P', position_norm_match='paired_min')
        for row in rows:
            self.assertEqual(row['selected_coordinates'], 0)
            self.assertIsNone(row['conflict_removed_norm'])
            self.assertGreater(row['post_permission_removed_norm'], 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_vit_shape_cuda_startup_and_merge_all_new_paths(self):
        from models.attention import Attention_LoRA
        configurations = [dict(dual_mask_position_norm_match='paired_min', dual_mask_protect_position=position,
                              dual_mask_conflict_score_mode=score)
                          for position in ('wpre', 'permuted') for score in ('conflict', 'magnitude')]
        configurations += [dict(dual_mask_permission_mode=mode) for mode in ('symmetric_soft', 'symmetric_hard')]
        for configuration in configurations:
            module = Attention_LoRA(dim=768, num_heads=12, qkv_bias=True, r=64, n_tasks=2).cuda()
            module._init_params(dict(use_slora=True, use_plora=True, dual_mask_mechanism_audit=True,
                dual_mask_competence_adaptive=False, dual_mask_conflict_energy_adaptive=False,
                dual_mask_conflict_ratio=.1, dual_mask_task0_gate_mode='unmasked', seed=1993, **configuration))
            module.layer_idx = 2
            module.w0_importance.uniform_()
            module.before_task(1)
            module.set_task_and_stage(1, 2)
            x = torch.randn(2, 5, 768, device='cuda')
            module._contrib_from_units(x, 1).sum().backward()
            for unit in (module.S_lora[1], module.P_lora[1]):
                self.assertGreater(unit.B_weight.grad.norm().item(), 0)
                self.assertIsNone(unit.A_weight.grad)
            with torch.no_grad():
                for unit in (module.S_lora[1], module.P_lora[1]):
                    unit.B_weight.normal_(std=.01)
                before = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
                module.after_task(1)
                torch.testing.assert_close(before, F.linear(x, module.qkv.weight, module.qkv.bias), atol=5e-5, rtol=5e-5)
                self.assertIsNone(module.S_lora[1])
                self.assertIsNone(module.P_lora[1])

    def test_same_state_positions_and_forward_merge(self):
        for mode in ('asymmetric', 'symmetric_soft', 'symmetric_hard'):
            for match in ('off', 'paired_min'):
                for position in ('wpre', 'permuted'):
                    module = self.make(dual_mask_permission_mode=mode,
                        dual_mask_position_norm_match=match, dual_mask_protect_position=position,
                        dual_mask_mechanism_audit=True)
                    self.assertEqual(module._core_policy_active(), match == 'paired_min' or mode != 'asymmetric')
                    with torch.no_grad():
                        for unit in (module.S_lora[1], module.P_lora[1]):
                            unit.B_weight.normal_()
                    if match == 'paired_min':
                        for isolated, unit in ((False, module.S_lora[1]), (True, module.P_lora[1])):
                            raw = unit.B_weight @ unit.A_weight
                            values = []
                            for candidate in ('wpre', 'permuted'):
                                module._core_audit_position = candidate
                                values.append(module._safe_delta(raw, isolated).reshape(3, -1).norm(dim=1))
                            module._core_audit_position = None
                            torch.testing.assert_close(values[0], values[1])
                            self.assertFalse(torch.equal(module.core_reference_protect, module.core_permuted_protect))
                    x = torch.randn(5, 4)
                    before = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
                    rng = torch.get_rng_state().clone()
                    module.after_task(1)
                    torch.testing.assert_close(before, F.linear(x, module.qkv.weight, module.qkv.bias), atol=3e-6, rtol=1e-5)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertEqual(module._contrib_from_units(x, 1).count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
