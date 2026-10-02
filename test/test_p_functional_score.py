import copy
import unittest

import torch

from test import test_p_direction_score as direction_tests
from utils.p_functional_score import functional_score, gram_risk, qk_risk


MODES = ('coordinate', 'wpre_product', 'wpre_input', 'wpre_output', 'wpre_qk', 'task_qk')


class FunctionalMathTests(unittest.TestCase):
    def test_gram_gradient_matches_autograd_with_fixed_s_context(self):
        torch.manual_seed(7)
        anchor, context = torch.randn(3, 4, 4), torch.randn(3, 4, 4) * .1
        for kind in ('input', 'output'):
            delta = (torch.randn_like(anchor) * .1).requires_grad_()
            penalty, gradient = gram_risk(anchor, delta.detach(), context, kind)
            actual = anchor + context + delta
            original = anchor
            if kind == 'input':
                actual = actual.transpose(-2, -1) @ actual
                original = original.transpose(-2, -1) @ original
            else:
                actual = actual @ actual.transpose(-2, -1)
                original = original @ original.transpose(-2, -1)
            expected = (actual - original).square().sum((-2, -1)) / original.square().sum((-2, -1))
            expected.sum().backward()
            torch.testing.assert_close(penalty, expected.detach())
            torch.testing.assert_close(gradient, delta.grad)

    def test_qk_gradient_matches_autograd_including_bias(self):
        torch.manual_seed(17)
        anchor, context = torch.randn(3, 4, 4), torch.randn(3, 4, 4) * .1
        bias = torch.randn(3, 4)
        delta = (torch.randn_like(anchor) * .1).requires_grad_()
        penalty, gradient = qk_risk(anchor, delta.detach(), context, 2, bias)
        augmented = torch.cat((anchor + context + delta, bias.unsqueeze(-1)), -1).reshape(3, 2, 2, 5)
        original = torch.cat((anchor, bias.unsqueeze(-1)), -1).reshape(3, 2, 2, 5)
        reference = original[0].transpose(-2, -1) @ original[1]
        changed = augmented[0].transpose(-2, -1) @ augmented[1] - reference
        expected = (changed.square().sum((-2, -1)) / reference.square().sum((-2, -1))).mean()
        expected.backward()
        torch.testing.assert_close(penalty, expected.detach())
        torch.testing.assert_close(gradient, delta.grad)

    def test_large_compensating_qk_updates_have_zero_joint_risk(self):
        anchor = torch.eye(3).repeat(3, 1, 1)
        delta = torch.zeros_like(anchor)
        delta[0] = anchor[0]
        delta[1] = -.5 * anchor[1]
        risk, gradient = qk_risk(anchor, delta, torch.zeros_like(delta), 1)
        self.assertGreater(float(delta.norm()), 1)
        torch.testing.assert_close(risk, torch.zeros_like(risk))
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_positive_signed_contribution_distinguishes_compensation(self):
        anchor = torch.eye(2).repeat(3, 1, 1)
        context = .2 * anchor
        helpful, harmful = -.1 * anchor, .1 * anchor
        score_a, _ = functional_score('wpre_input', helpful, anchor, context, 1)
        score_b, _ = functional_score('wpre_input', harmful, anchor, context, 1)
        self.assertEqual(float(score_a.sum()), 0)
        self.assertGreater(float(score_b.sum()), 0)

    def test_qk_modes_leave_v_ranking_as_magnitude_and_are_rng_free(self):
        torch.manual_seed(3)
        anchor, delta = torch.randn(3, 4, 4), torch.randn(3, 4, 4)
        state = torch.get_rng_state().clone()
        for mode in ('wpre_qk', 'task_qk'):
            score, _ = functional_score(mode, delta, anchor, torch.zeros_like(delta), 2)
            torch.testing.assert_close(score[2], delta[2].abs())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


class FunctionalIntegrationTests(unittest.TestCase):
    def make_layer(self, mode, device='cpu'):
        return direction_tests.PDirectionIntegrationTests().make_layer(mode, device)

    def test_same_reference_norm_and_s_path_for_all_candidates(self):
        layer = self.make_layer('coordinate')
        p = layer.P_lora[1]
        raw = layer.plora_gamma * (p.B_weight @ p.A_weight)
        s = layer.S_lora[1]
        raw_s = layer.slora_gamma * (s.B_weight @ s.A_weight)
        expected_s = layer._safe_delta(raw_s, False)
        for mode in MODES:
            layer.args['p_direction_score'] = mode
            safe = layer._safe_delta(raw, True)
            state = layer._p_direction_state
            actual = (state['base'] - safe.reshape_as(state['base'])).norm(dim=(-2, -1))
            target = (state['base'] * state['reference'] * state['reference_strength']).norm(dim=(-2, -1))
            torch.testing.assert_close(actual, target)
            self.assertTrue(torch.equal(expected_s, layer._safe_delta(raw_s, False)))
            self.assertFalse(bool((state['selected'] & layer.general_mask.reshape_as(state['base']).bool()).any()))

    def test_forward_gradient_merge_and_task0_invariance(self):
        for mode in MODES:
            layer = self.make_layer(mode)
            layer.slora_gamma, layer.plora_gamma = .3, 1.2
            x = torch.randn(2, 5, 4)
            before = layer.qkv(x) + layer._contrib_from_units(x, 1)
            before.square().mean().backward()
            self.assertGreater(float(layer.P_lora[1].B_weight.grad.norm()), 0)
            anchor = layer.pretrained_weight.clone()
            with self.assertLogs(level='INFO'):
                layer.after_task(1)
            torch.testing.assert_close(layer.qkv(x), before.detach(), rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(anchor, layer.pretrained_weight))
            baseline = self.make_layer('off')
            baseline.cur_task = 0
            baseline.before_task(0)
            baseline.set_task_and_stage(0, 0)
            candidate = copy.deepcopy(baseline)
            candidate.args['p_direction_score'] = mode
            self.assertTrue(torch.equal(baseline._contrib_from_units(x, 0), candidate._contrib_from_units(x, 0)))

    def test_wpre_proxy_includes_previously_merged_weight_changes(self):
        layer = self.make_layer('wpre_input')
        with torch.no_grad():
            layer.qkv.weight.add_(.2)
        p, s = layer.P_lora[1], layer.S_lora[1]
        raw = layer.plora_gamma * (p.B_weight @ p.A_weight)
        layer._p_direction_gate(raw)
        state = layer._p_direction_state
        anchor = layer.pretrained_weight.reshape_as(state['base'])
        context = (layer.qkv.weight.detach().reshape_as(anchor) - anchor +
                   layer._safe_delta(layer.slora_gamma * (s.B_weight @ s.A_weight), False).reshape_as(anchor))
        expected, _ = gram_risk(anchor, state['base'], context, 'input')
        torch.testing.assert_close(state['functional']['risk'], expected)

    def test_zero_new_update_separates_pretrained_and_task_start_references(self):
        layer = self.make_layer('wpre_qk')
        with torch.no_grad():
            layer.qkv.weight.add_(.2)
            layer.S_lora[1].B_weight.zero_()
        raw = torch.zeros_like(layer.qkv.weight)
        layer._p_direction_gate(raw)
        self.assertGreater(float(layer._p_direction_state['functional']['risk']), 0)
        layer.args['p_direction_score'] = 'task_qk'
        layer._p_direction_gate(raw)
        torch.testing.assert_close(layer._p_direction_state['functional']['risk'], torch.tensor(0.))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware required')
    def test_cuda_vit_sized_math_attention_and_merge(self):
        from models.attention import Attention_LoRA
        for mode in MODES:
            args = dict(self.make_layer(mode).args)
            layer = Attention_LoRA(768, num_heads=12, qkv_bias=True, r=64, n_tasks=2).cuda()
            layer._init_params(args)
            layer.before_task(1)
            layer.set_task_and_stage(1, 0)
            layer.eval()
            with torch.no_grad(), torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True):
                layer.S_lora[1].B_weight.normal_(std=.01)
                layer.P_lora[1].B_weight.normal_(std=.01)
                x = torch.randn(1, 4, 768, device='cuda')
                before = layer(x, 1)
                layer.after_task(1)
                torch.testing.assert_close(layer(x, 1), before, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
