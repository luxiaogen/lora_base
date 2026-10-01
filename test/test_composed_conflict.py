"""Common conflict gates preserve S/P cancellation after their base masks."""
import copy
import unittest

import torch
from torch.nn import functional as F

from test import test_global_conflict_budget as fixtures


class ComposedConflictTests(unittest.TestCase):
    def make_layer(self, mode='net', device='cpu'):
        torch.manual_seed(57)
        layer = fixtures.GlobalBudgetSelectionTests._make_attention('layer',
            dual_mask_composed_conflict=mode, dual_mask_task0_gate_mode='unmasked')
        layer.before_task(1)
        layer.set_task_and_stage(1, 0)
        layer.to(device)
        with torch.no_grad():
            layer.general_mask.zero_()
            layer.general_mask[0, 0] = 1
            layer.w0_importance.copy_(torch.linspace(.1, 1., 48, device=device).reshape(12, 4))
            for unit in (layer.S_lora[1], layer.P_lora[1]):
                unit.B_weight.normal_(std=.1)
        return layer

    def test_shared_gate_cannot_amplify_or_reverse_net_coordinates(self):
        for mode in ('net', 'gross', 'magnitude', 'w_pre', 'random', 'uniform'):
            layer = self.make_layer(mode)
            state = layer._composed_conflict_state()
            net, gate = state['net'], state['gate']
            safe = state['s_base'] * gate + state['p_base'] * gate
            torch.testing.assert_close(safe, net * gate)
            self.assertTrue(bool((gate >= 0).all() & (gate <= 1).all()))
            self.assertTrue(bool((safe.abs() <= net.abs() + 1e-7).all()))
            self.assertTrue(bool((safe * net >= -1e-8).all()))

    def test_independent_gates_can_break_cancellation(self):
        s, p = torch.tensor([10.]), torch.tensor([-9.])
        self.assertEqual((s + p).item(), 1.)
        self.assertEqual((s * .5 + p).item(), -4.)
        self.assertEqual(((s + p) * .5).item(), .5)

    def test_composition_includes_gamma_and_protection_before_ranking(self):
        layer = self.make_layer()
        layer.slora_gamma, layer.plora_gamma = .2, 1.3
        layer.effective_protect_strength = .6
        state = layer._composed_conflict_state()
        s, p = layer.S_lora[1], layer.P_lora[1]
        expected_s = .2 * (s.B_weight @ s.A_weight) * (1 - .6 * layer.general_mask)
        expected_p = 1.3 * (p.B_weight @ p.A_weight) * (1 - layer.general_mask)
        torch.testing.assert_close(state['s_base'], expected_s)
        torch.testing.assert_close(state['p_base'], expected_p)
        self.assertEqual(state['p_base'][0, 0].item(), 0.)
        expected_score = layer.w0_importance * (expected_s + expected_p).detach().abs()
        k = state['reference_k']
        expected_indices = expected_score.flatten().topk(k).indices.sort().values
        actual_indices = state['selected'].flatten().nonzero().flatten().sort().values
        self.assertTrue(torch.equal(actual_indices, expected_indices))

    def test_selection_controls_have_identical_reference_budget(self):
        counts = []
        for mode in ('net', 'gross', 'magnitude', 'w_pre', 'random', 'uniform'):
            layer = self.make_layer(mode)
            state = layer._composed_conflict_state()
            counts.append(state['reference_k'])
            self.assertEqual(int(state['selected'].sum()), state['reference_k'])
            self.assertEqual(state['reference_k'],
                (int(state['s_reference'].sum()) + int(state['p_reference'].sum()) + 1) // 2)
        self.assertEqual(len(set(counts)), 1)

    def test_adaptive_reference_counts_are_from_original_branch_updates(self):
        layer = self.make_layer()
        layer.dual_mask_conflict_energy_adaptive = True
        state = layer._composed_conflict_state()
        for key, unit, gamma in (('s_reference', layer.S_lora[1], layer.slora_gamma),
                                 ('p_reference', layer.P_lora[1], layer.plora_gamma)):
            _, expected = layer._joint_conflict(gamma * (unit.B_weight @ unit.A_weight))
            self.assertTrue(torch.equal(expected, state[key]))

    def test_uniform_matches_net_selection_removed_norm_at_same_weights(self):
        layer = self.make_layer()
        net_state = layer._composed_conflict_state()
        layer.dual_mask_composed_conflict = 'uniform'
        uniform = layer._composed_conflict_state()
        a = net_state['net'] * (1 - net_state['gate'])
        b = uniform['net'] * (1 - uniform['gate'])
        torch.testing.assert_close(a.norm(), b.norm())

    def test_forward_backward_and_merge_match_with_unequal_gammas(self):
        for mode in ('net', 'gross', 'magnitude', 'w_pre', 'random', 'uniform'):
            layer = self.make_layer(mode)
            layer.slora_gamma, layer.plora_gamma = .35, .9
            x = torch.randn(2, 5, 4)
            state = layer._composed_conflict_state()
            expected = F.linear(x, state['net'] * state['gate'])
            actual = layer._contrib_from_units(x, 1)
            torch.testing.assert_close(actual, expected)
            actual.square().mean().backward()
            for unit in (layer.S_lora[1], layer.P_lora[1]):
                self.assertIsNone(unit.A_weight.grad)
                self.assertGreater(unit.B_weight.grad.norm().item(), 0.)
            before = layer.qkv(x) + actual.detach()
            anchor = layer.pretrained_weight.clone()
            layer.after_task(1)
            torch.testing.assert_close(layer.qkv(x), before, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(layer.pretrained_weight, anchor))
            self.assertIsNone(layer.S_lora[1])
            self.assertIsNone(layer.P_lora[1])
            self.assertEqual(layer._contrib_from_units(x, 1).abs().sum().item(), 0.)

    def test_regularization_reuses_actual_forward_gate(self):
        layer = self.make_layer()
        layer._contrib_from_units(torch.randn(2, 3, 4), 1)
        gate = layer._composed_forward_gate.clone()
        for isolated, unit in ((False, layer.S_lora[1]), (True, layer.P_lora[1])):
            delta = unit.B_weight @ unit.A_weight
            safe = delta * layer._composed_base_gate(delta, isolated) * gate
            score, _ = layer._joint_conflict(delta)
            expected = ((layer.w0_importance + score.detach()) * safe.square()).mean()
            torch.testing.assert_close(layer._joint_conflict_regularization(unit, isolated), expected)

    def test_random_selection_is_task_fixed_and_does_not_change_rng(self):
        layer = self.make_layer('random')
        before = torch.get_rng_state().clone()
        first = layer._composed_conflict_state()['selected']
        with torch.no_grad():
            layer.P_lora[1].B_weight.mul_(1.2)
        second = layer._composed_conflict_state()['selected']
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(first, second))

    def test_off_and_task0_preserve_original_outputs_and_rng(self):
        original = self.make_layer('off')
        original.cur_task = 0
        original.before_task(0)
        original.set_task_and_stage(0, 0)
        with torch.no_grad():
            original.S_lora[0].B_weight.normal_()
        candidate = copy.deepcopy(original)
        candidate.dual_mask_composed_conflict = 'net'
        x = torch.randn(2, 3, 4)
        rng = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(original._contrib_from_units(x, 0),
                                   candidate._contrib_from_units(x, 0)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        layer = self.make_layer('off')
        self.assertFalse(layer._composed_conflict_active())

    def test_merge_telemetry_uses_actual_mask(self):
        layer = self.make_layer()
        with self.assertLogs(level='INFO') as logs:
            layer.after_task(1)
        self.assertTrue(any('ComposedConflict ' in row for row in logs.output))
        self.assertFalse(any('P applied merge diagnostic:' in row for row in logs.output))

    def test_zero_updates_report_zero_suppression(self):
        layer = self.make_layer()
        with torch.no_grad():
            layer.S_lora[1].B_weight.zero_()
            layer.P_lora[1].B_weight.zero_()
        layer.after_task(1)
        self.assertEqual(layer.last_conflict_gate_suppression.item(), 0.)
        self.assertEqual(layer.last_safe_suppression.item(), 0.)
        self.assertEqual(layer.last_private_conflict_gate_suppression.item(), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware required')
    def test_cuda_full_attention_logits_match_after_merge(self):
        for mode in ('net', 'gross', 'magnitude', 'w_pre', 'random', 'uniform'):
            layer = self.make_layer(mode, 'cuda').eval()
            x = torch.randn(2, 5, 4, device='cuda')
            with torch.no_grad(), torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True):
                before = layer(x, 1)
                layer.after_task(1)
                after = layer(x, 1)
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
