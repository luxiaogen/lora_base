"""Catch wrong depth direction, mass/count confusion, and ineffective installed gates."""
import copy
import json
import subprocess
import unittest

import torch
from torch.nn import functional as F


class DepthPermissionTests(unittest.TestCase):
    def api(self):
        try:
            from utils.depth_permissions import depth_value, calibrate_center, install_depth_permissions
        except ImportError:
            self.fail('The depth permission implementation is missing')
        return depth_value, calibrate_center, install_depth_permissions

    def make_modules(self, count=4, task=1):
        from test.test_protect_position import ProtectPositionTests
        modules = []
        for layer in range(count):
            module = ProtectPositionTests().make(task=task)
            module.args.update(dual_mask_fixed_coverage=.9, dual_mask_fixed_protect_strength=.5,
                dual_mask_conflict_exact_topk=True, dual_mask_conflict_score_mode='magnitude')
            module.layer_idx = layer
            module.n_tasks = 3
            module.S_lora.append(None)
            module.P_lora.append(None)
            module.before_task(task)
            module.set_task_and_stage(task, layer)
            modules.append(module)
        return modules

    def test_depth_endpoints_and_single_layer_midpoint(self):
        value, _, _ = self.api()
        self.assertEqual(value(.5, .25, 0, 12), .75)
        self.assertEqual(value(.5, .25, 11, 12), .25)
        self.assertEqual(value(.9, .05, 0, 1), .9)
        self.assertAlmostEqual(value(.5, .25, 1, 3), .5)

    def test_coverage_calibration_matches_mass_threshold_counts_not_density(self):
        _, calibrate, _ = self.api()
        from models.attention import _energy_coverage_mask
        scores = [torch.arange(10, 0, -1).float() for _ in range(4)]
        center = calibrate(scores, .9, .05)
        actual = [_energy_coverage_mask(score, center + offset) for score, offset in
                  zip(scores, [.05, .05/3, -.05/3, -.05])]
        self.assertEqual(sum(int(mask.sum()) for mask in actual), 32)
        self.assertGreaterEqual(int(actual[0].sum()), int(actual[-1].sum()))
        # 并列分数也沿用实际质量前缀的数量，而不是全选或坐标分位数。
        ties = [torch.ones(10), torch.ones(10)]
        tie_center = calibrate(ties, .9, .05)
        self.assertEqual(sum(int(_energy_coverage_mask(score, tie_center + offset).sum())
                             for score, offset in zip(ties, [.05, -.05])), 18)
        self.assertAlmostEqual(calibrate([torch.zeros(10)] * 2, .9, .05), .9)

    def test_off_and_task0_preserve_state_and_rng(self):
        _, _, install = self.api()
        for task, coverage, strength in [(1, 'off', 'off'), (0, 'shallow_high', 'shallow_high')]:
            modules = self.make_modules(task=task)
            snapshots = [copy.deepcopy(module.state_dict()) for module in modules]
            strengths = [module.effective_protect_strength for module in modules]
            state = torch.get_rng_state().clone()
            install(modules, task, dict(dual_mask_depth_coverage=coverage,
                dual_mask_depth_strength=strength))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            for module, snapshot, alpha in zip(modules, snapshots, strengths):
                self.assertEqual(module.effective_protect_strength, alpha)
                for key, value in snapshot.items():
                    torch.testing.assert_close(module.state_dict()[key], value, rtol=0, atol=0)

    def test_strength_only_changes_s_not_masks_or_p(self):
        _, _, install = self.api()
        modules = self.make_modules()
        reference = copy.deepcopy(modules)
        install(modules, 1, dict(dual_mask_depth_strength='shallow_high'))
        self.assertEqual([module.effective_protect_strength for module in modules],
                         [.75, .5 + .25/3, .5 - .25/3, .25])
        raw = torch.arange(48).reshape(12, 4).float() / 10
        for module, original in zip(modules, reference):
            torch.testing.assert_close(module.general_mask, original.general_mask, rtol=0, atol=0)
            torch.testing.assert_close(module._safe_delta(raw, True), original._safe_delta(raw, True), rtol=0, atol=0)
        self.assertFalse(torch.equal(modules[0]._safe_delta(raw, False), reference[0]._safe_delta(raw, False)))

    def test_coverage_only_preserves_importance_and_installs_complement_across_tasks(self):
        _, _, install = self.api()
        modules = self.make_modules()
        reference_counts = sum(int(module.general_mask.sum()) for module in modules)
        scores = [module.w0_importance.clone() for module in modules]
        rng = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO') as captured:
            center = install(modules, 1, dict(dual_mask_depth_coverage='shallow_high', dual_mask_depth_audit=True))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        records = [json.loads(line.split('DepthPermission ', 1)[1]) for line in captured.output
                   if 'DepthPermission ' in line]
        self.assertEqual(len(records), 4)
        self.assertGreater(records[0]['coverage_target'], records[-1]['coverage_target'])
        self.assertLessEqual(abs(sum(int(m.general_mask.sum()) for m in modules) - reference_counts), 1)
        masks = [module.general_mask.clone() for module in modules]
        for module, score in zip(modules, scores):
            torch.testing.assert_close(module.w0_importance, score, rtol=0, atol=0)
            torch.testing.assert_close(module.isolated_mask, 1-module.general_mask, rtol=0, atol=0)
            self.assertEqual(module.effective_protect_strength, .5)
            module.before_task(2)
        install(modules, 2, dict(dual_mask_depth_coverage='shallow_high'), center=center)
        for module, mask in zip(modules, masks):
            torch.testing.assert_close(module.general_mask, mask, rtol=0, atol=0)

    def test_zero_b_gradient_and_actual_forward_merge_once(self):
        _, _, install = self.api()
        for coverage, strength in [('shallow_high', 'off'), ('off', 'shallow_high'), ('shallow_high', 'shallow_high')]:
            modules = self.make_modules()
            install(modules, 1, dict(dual_mask_depth_coverage=coverage, dual_mask_depth_strength=strength))
            for module in modules:
                x = torch.randn(5, 4)
                module._contrib_from_units(x, 1).sum().backward()
                for unit in (module.S_lora[1], module.P_lora[1]):
                    self.assertGreater(unit.B_weight.grad.norm().item(), 0)
                    self.assertFalse(unit.A_weight.requires_grad)
                    with torch.no_grad():
                        unit.B_weight.normal_(std=.03)
                expected = module.qkv(x) + module._contrib_from_units(x, 1)
                anchor = module.pretrained_weight.clone()
                module.after_task(1)
                torch.testing.assert_close(module.qkv(x), expected, atol=2e-6, rtol=2e-6)
                merged = module.qkv.weight.clone()
                module.after_task(1)
                torch.testing.assert_close(module.qkv.weight, merged, rtol=0, atol=0)
                torch.testing.assert_close(module.pretrained_weight, anchor, rtol=0, atol=0)

    def test_default_matches_prechange_forward_loss_gradient_step_and_merge(self):
        from models.attention import Attention_LoRA
        from test.test_global_conflict_budget import GlobalBudgetSelectionTests
        source = subprocess.check_output(['git', 'show',
            'd5cb5bc0c36f9783c688de54a0974e70aa40f977:models/attention.py'], text=True)
        namespace = dict(__name__='depth_regression_reference')
        exec(compile(source, 'prechange_attention.py', 'exec'), namespace)
        args = dict(GlobalBudgetSelectionTests._make_attention('layer').args,
            dual_mask_fixed_coverage=.9, dual_mask_fixed_protect_strength=.5,
            dual_mask_task0_gate_mode='unmasked', dual_mask_conflict_exact_topk=True,
            dual_mask_conflict_score_mode='magnitude', dual_mask_conflict_ratio=.1)
        for task in (0, 1):
            values = []
            for cls in (namespace['Attention_LoRA'], Attention_LoRA):
                torch.manual_seed(1993)
                module = cls(dim=4, num_heads=1, r=2, n_tasks=2)
                module._init_params(args.copy())
                module.w0_importance.copy_(torch.arange(48).reshape(12, 4).float() / 47)
                module.before_task(task)
                module.set_task_and_stage(task, 0)
                if cls is Attention_LoRA:
                    self.api()[2]([module], task, dict(dual_mask_depth_audit=True))
                x, target = torch.randn(5, 4), torch.randn(5, 12)
                output = module.qkv(x) + module._contrib_from_units(x, task)
                loss = (output - target).square().mean()
                loss.backward()
                gradients = {name: p.grad.clone() for name, p in module.named_parameters() if p.grad is not None}
                torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.01).step()
                module.after_task(task)
                values.append((output.detach(), loss.detach(), gradients,
                               copy.deepcopy(module.state_dict()), torch.get_rng_state().clone()))
            for left, right in zip(*values):
                if isinstance(left, dict):
                    self.assertEqual(left.keys(), right.keys())
                    for key in left:
                        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
