import copy
import unittest

import torch

from test import test_composed_conflict as fixtures
from utils.p_direction_score import norm_matched_gate, spectral_score


class SpectralScoreTests(unittest.TestCase):
    def test_signed_distinguishes_enhancement_from_attenuation(self):
        identity = torch.eye(3).unsqueeze(0)
        singular = torch.tensor([[3., 2., 1.]])
        delta = torch.diag(torch.tensor([.3, -.3, .2])).unsqueeze(0)
        absolute, _, _ = spectral_score(delta, identity, singular, identity)
        signed, fractions, _ = spectral_score(delta, identity, singular, identity, True)
        self.assertGreater(absolute[0, 0, 0], 0)
        self.assertEqual(signed[0, 0, 0], 0)
        torch.testing.assert_close(signed[0, 1, 1], absolute[0, 1, 1])
        torch.testing.assert_close(fractions.sum(-1), torch.ones(1))

    def test_off_diagonal_mixing_is_not_dropped(self):
        identity = torch.eye(3).unsqueeze(0)
        delta = torch.zeros(1, 3, 3)
        delta[0, 0, 1] = .5
        signed, fractions, _ = spectral_score(delta, identity, torch.ones(1, 3), identity, True)
        self.assertGreater(signed[0, 0, 1], 0)
        torch.testing.assert_close(fractions, torch.tensor([[0., 0., 1.]]))

    def test_svd_pair_sign_changes_do_not_change_score(self):
        torch.manual_seed(91)
        weight, delta = torch.randn(3, 4, 4), torch.randn(3, 4, 4)
        left, singular, vh = torch.linalg.svd(weight)
        right = vh.transpose(-2, -1)
        signs = torch.tensor([1., -1., 1., -1.])
        for signed in (False, True):
            expected = spectral_score(delta, left, singular, right, signed)[0]
            actual = spectral_score(delta, left * signs, singular, right * signs, signed)[0]
            torch.testing.assert_close(expected, actual)

    def test_norm_match_respects_plasticity_and_matches_each_projection(self):
        base = torch.tensor([[[1., 3.], [2., 4.]]] * 3)
        plastic = torch.ones_like(base, dtype=torch.bool)
        plastic[:, 1, 1] = False
        base = base * plastic
        reference = torch.zeros_like(plastic)
        reference[:, 0, 1] = True
        score = torch.tensor([[[4., 1.], [3., 9.]]] * 3)
        gate, selected, strength = norm_matched_gate(base, score, reference, .5, plastic)
        self.assertTrue(bool((selected & ~plastic).sum() == 0))
        self.assertTrue(bool((strength <= 1).all() & (strength >= 0).all()))
        torch.testing.assert_close((base * (1 - gate)).norm(dim=(-2, -1)),
                                   (base * reference * .5).norm(dim=(-2, -1)))
        self.assertEqual(int(selected[0].sum()), 2)  # top-1 cannot supply target norm.

    def test_zero_update_and_zero_budget_are_finite(self):
        base = torch.zeros(3, 4, 4)
        gate, selected, strength = norm_matched_gate(base, base, base.bool(), .5, torch.ones_like(base))
        self.assertTrue(bool(torch.isfinite(gate).all() & torch.isfinite(strength).all()))
        self.assertFalse(bool(selected.any()))
        torch.testing.assert_close(gate, torch.ones_like(gate))


class PDirectionIntegrationTests(unittest.TestCase):
    def make_layer(self, mode, device='cpu'):
        layer = fixtures.ComposedConflictTests().make_layer('off', device)
        layer.args['p_direction_score'] = mode
        return layer

    def test_s_is_bitwise_unchanged_for_all_modes(self):
        layer = self.make_layer('off')
        raw = layer.S_lora[1].B_weight @ layer.S_lora[1].A_weight
        expected = layer._safe_delta(raw, False)
        for mode in ('coordinate', 'spectral', 'signed'):
            layer.args['p_direction_score'] = mode
            self.assertTrue(torch.equal(layer._safe_delta(raw, False), expected))

    def test_qkv_norm_matches_original_conflict_at_same_weights(self):
        for mode in ('coordinate', 'spectral', 'signed'):
            layer = self.make_layer(mode)
            raw = layer.P_lora[1].B_weight @ layer.P_lora[1].A_weight
            safe, gate, selected = layer._safe_delta(raw, True, return_details=True)
            base = raw * (1 - layer.general_mask)
            _, reference = layer._branch_conflict(raw, True)
            target = base * reference * layer._conflict_parameters()[1]
            torch.testing.assert_close((base - safe).reshape(3, 4, 4).norm(dim=(-2, -1)),
                                       target.reshape(3, 4, 4).norm(dim=(-2, -1)))
            self.assertFalse(bool((selected.bool() & layer.general_mask.bool()).any()))
            self.assertTrue(bool((gate >= 0).all() & (gate <= 1).all()))

    def test_basis_uses_immutable_anchor_and_is_cached(self):
        layer = self.make_layer('signed')
        raw = layer.P_lora[1].B_weight @ layer.P_lora[1].A_weight
        layer._safe_delta(raw, True)
        expected = layer.p_direction_left.clone()
        ptr = layer.p_direction_left.data_ptr()
        with torch.no_grad():
            layer.qkv.weight.add_(10.)
        layer._safe_delta(raw, True)
        self.assertEqual(ptr, layer.p_direction_left.data_ptr())
        self.assertTrue(torch.equal(expected, layer.p_direction_left))
        layer.capture_pretrained_anchor(force=True)
        self.assertIsNone(layer.p_direction_left)

    def test_forward_gradient_and_merge_agree_with_unequal_gammas(self):
        for mode in ('coordinate', 'spectral', 'signed'):
            layer = self.make_layer(mode)
            layer.slora_gamma, layer.plora_gamma = .3, 1.2
            x = torch.randn(2, 5, 4)
            actual = layer._contrib_from_units(x, 1)
            actual.square().mean().backward()
            for unit in (layer.S_lora[1], layer.P_lora[1]):
                self.assertIsNone(unit.A_weight.grad)
                self.assertGreater(float(unit.B_weight.grad.norm()), 0)
            before = layer.qkv(x) + actual.detach()
            anchor = layer.pretrained_weight.clone()
            with self.assertLogs(level='INFO') as records:
                layer.after_task(1)
            torch.testing.assert_close(layer.qkv(x), before, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(anchor, layer.pretrained_weight))
            self.assertIsNone(layer.P_lora[1])
            self.assertEqual(float(layer._contrib_from_units(x, 1).abs().sum()), 0)
            self.assertTrue(any('PDirectionScore ' in row for row in records.output))
            self.assertFalse(any('P applied merge diagnostic:' in row for row in records.output))

    def test_regularization_uses_actual_forward_gate_without_recomputing(self):
        layer = self.make_layer('signed')
        layer._contrib_from_units(torch.randn(2, 3, 4), 1)
        gate = layer._p_direction_state['gate'].reshape(12, 4)
        unit = layer.P_lora[1]
        raw = unit.B_weight @ unit.A_weight
        score, _ = layer._joint_conflict(raw)
        safe = raw * (1 - layer.general_mask) * gate
        expected = ((layer.w0_importance + score.detach()) * safe.square()).mean()
        torch.testing.assert_close(layer._joint_conflict_regularization(unit, True), expected)

    def test_task0_and_rng_are_unchanged(self):
        layer = self.make_layer('off')
        layer.cur_task = 0
        layer.before_task(0)
        layer.set_task_and_stage(0, 0)
        candidate = copy.deepcopy(layer)
        candidate.args['p_direction_score'] = 'signed'
        x = torch.randn(2, 3, 4)
        rng = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(layer._contrib_from_units(x, 0), candidate._contrib_from_units(x, 0)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        candidate = self.make_layer('signed')
        before = torch.get_rng_state().clone()
        candidate._contrib_from_units(x, 1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware required')
    def test_cuda_full_attention_pre_post_merge(self):
        for mode in ('coordinate', 'spectral', 'signed'):
            layer = self.make_layer(mode, 'cuda').eval()
            x = torch.randn(2, 5, 4, device='cuda')
            with torch.no_grad(), torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True):
                before = layer(x, 1)
                layer.after_task(1)
                after = layer(x, 1)
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware required')
    def test_cuda_vit_sized_gate_and_merge(self):
        from models.attention import Attention_LoRA
        for mode in ('coordinate', 'spectral', 'signed'):
            args = dict(self.make_layer(mode).args)
            args['dual_mask_svd_rank'] = 768
            layer = Attention_LoRA(dim=768, num_heads=12, r=64, n_tasks=2).cuda()
            layer._init_params(args)
            layer.before_task(1)
            layer.set_task_and_stage(1, 0)
            layer.slora_gamma, layer.plora_gamma = .35, .9
            layer.eval()
            with torch.no_grad():
                layer.S_lora[1].B_weight.normal_(std=.01)
                layer.P_lora[1].B_weight.normal_(std=.01)
                x = torch.randn(2, 5, 768, device='cuda')
                before = layer.qkv(x) + layer._contrib_from_units(x, 1)
                gate = layer._p_direction_state['gate'].clone()
                raw = .9 * (layer.P_lora[1].B_weight @ layer.P_lora[1].A_weight)
                layer._safe_delta(raw, True)
                self.assertTrue(torch.equal(gate, layer._p_direction_state['gate']))
                layer.after_task(1)
                torch.testing.assert_close(layer.qkv(x), before, rtol=1e-4, atol=1e-5)
            del layer


if __name__ == '__main__':
    unittest.main()
