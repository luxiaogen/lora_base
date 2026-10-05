import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import models.attention as attention
from test.test_protect_position import ProtectPositionTests
from utils.dualmask_core_audit import epoch_updates


class CompactDualMaskTests(unittest.TestCase):
    def template(self):
        module = ProtectPositionTests().make()
        module.args.update(dual_mask_permission_mode='symmetric_soft',
                           dual_mask_fixed_protect_strength=.5)
        module.slora_gamma, module.plora_gamma = .5, .75
        module.dual_mask_private_rank = 1
        module.dual_mask_conflict_exact_topk = True
        module.dual_mask_conflict_score_mode = 'magnitude'
        module.dual_mask_conflict_ratio = .1
        return module

    def pair(self, task=1):
        dual = self.template()
        single = copy.deepcopy(dual)
        single.args['dual_mask_branch_layout'] = 'single'
        rng = torch.get_rng_state().clone()
        dual.before_task(task)
        final_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        single.before_task(task)
        self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))
        for module in (dual, single):
            module.set_task_and_stage(task, 2)
        return dual, single

    def test_task0_and_default_layout_exactly_unchanged(self):
        dual, single = self.pair(task=0)
        for name, value in dual.state_dict().items():
            torch.testing.assert_close(value, single.state_dict()[name], atol=0, rtol=0)
        self.assertTrue(single.S_lora[0].A_weight.requires_grad)
        x = torch.randn(5, 4)
        torch.testing.assert_close(dual._contrib_from_units(x, 0), single._contrib_from_units(x, 0), atol=0, rtol=0)
        default = self.template()
        explicit = copy.deepcopy(default)
        explicit.args['dual_mask_branch_layout'] = 'dual'
        rng = torch.get_rng_state().clone()
        default.before_task(1)
        torch.set_rng_state(rng)
        explicit.before_task(1)
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], atol=0, rtol=0)

    def test_single_capacity_initialization_and_active_parameters(self):
        dual, single = self.pair()
        s, p, merged = dual.S_lora[1], dual.P_lora[1], single.S_lora[1]
        self.assertIsNone(single.P_lora[1])
        self.assertEqual(merged.r, s.r + p.r)
        self.assertEqual(merged.A.out_features, s.r + p.r)
        self.assertEqual(merged.B.in_features, s.r + p.r)
        torch.testing.assert_close(merged.A_weight, torch.cat((.5 * s.A_weight, .75 * p.A_weight)), atol=0, rtol=0)
        self.assertEqual(merged.B_weight.count_nonzero().item(), 0)
        self.assertFalse(merged.A_weight.requires_grad)
        self.assertTrue(merged.B_weight.requires_grad)
        self.assertEqual(sum(x.numel() for x in merged.parameters()),
                         sum(x.numel() for u in (s, p) for x in u.parameters()))
        self.assertEqual(sum(x.numel() for x in single.parameters() if x.requires_grad),
                         sum(x.numel() for x in dual.parameters() if x.requires_grad))

    def test_unmasked_raw_update_and_b_gradient_scale_match(self):
        dual, single = self.pair()
        for module in (dual, single):
            module.effective_protect_strength = 0
            module.dual_mask_s_conflict_enabled = False
            module.dual_mask_p_conflict_enabled = False
        with torch.no_grad():
            s, p = dual.S_lora[1], dual.P_lora[1]
            s.B_weight.normal_()
            p.B_weight.normal_()
            single.S_lora[1].B_weight.copy_(torch.cat((s.B_weight, p.B_weight), dim=1))
        x = torch.randn(5, 4)
        first, second = [m._contrib_from_units(x, 1) for m in (dual, single)]
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
        first.sum().backward()
        second.sum().backward()
        torch.testing.assert_close(single.S_lora[1].B_weight.grad,
                                  torch.cat((s.B_weight.grad, p.B_weight.grad), dim=1))

    def test_exact_magnitude_budget_skips_importance_product(self):
        module = self.template()
        module.before_task(1)
        value = torch.arange(48).reshape(12, 4).float()
        module.w0_importance.fill_(float('nan'))
        with patch.object(attention, '_normalize_score', wraps=attention._normalize_score) as normalize:
            score, mask = module._joint_conflict(value)
        self.assertEqual(normalize.call_count, 1)
        self.assertTrue(torch.isfinite(score).all())
        self.assertEqual(int(mask.sum()), int(.1 * value.numel()))
        self.assertEqual(set(mask.flatten().nonzero().flatten().tolist()), {44, 45, 46, 47})

    def test_exact_budget_ties_zeros_and_plastic_denominator(self):
        module = self.template()
        module.before_task(1)
        for value in (torch.zeros(12, 4), torch.ones(12, 4)):
            _, mask = module._joint_conflict(value)
            self.assertEqual(int(mask.sum()), 4)
        module.set_task_and_stage(1, 2)
        module._contrib_from_units(torch.randn(3, 4), 1).sum().backward()
        for unit in (module.S_lora[1], module.P_lora[1]):
            self.assertGreater(unit.B_weight.grad.norm().item(), 0)
            self.assertIsNone(unit.A_weight.grad)
        value = torch.arange(48).reshape(12, 4).float()
        _, selected = module._branch_conflict(value, isolated=True)
        self.assertEqual(int(selected.sum()), 4)  # Full QKV count, not the P complement.

    def test_single_nonzero_forward_merge_once_and_telemetry(self):
        for match in ('off', 'paired_min'):
            for position in ('wpre', 'permuted'):
                _, module = self.pair()
                module.args.update(dual_mask_position_norm_match=match,
                                   dual_mask_protect_position=position,
                                   dual_mask_mechanism_audit=True,
                                   dual_mask_position_audit=True)
                module.rebuild_dual_masks()
                with torch.no_grad():
                    module.S_lora[1].B_weight.normal_()
                learner = SimpleNamespace(_cur_task=1, args=module.args, _iter_lora_modules=lambda: [module])
                with self.assertLogs(level='INFO') as logs:
                    epoch_updates(learner, 1)
                rows = [json.loads(line.split('CoreEpochUpdate ', 1)[1])
                        for line in logs.output if 'CoreEpochUpdate ' in line]
                self.assertEqual(len(rows), 3)
                self.assertEqual({row['branch'] for row in rows}, {'Single'})
                raw = module.S_lora[1].B_weight @ module.S_lora[1].A_weight
                for block, row in zip(raw.chunk(3), rows):
                    self.assertAlmostEqual(block.norm().item(), row['raw_norm'], places=5)
                x = torch.randn(5, 4)
                before = module.qkv(x) + module._contrib_from_units(x, 1)
                module.after_task(1)
                torch.testing.assert_close(before, module.qkv(x), atol=2e-6, rtol=1e-5)
                weight = module.qkv.weight.detach().clone()
                module.after_task(1)
                torch.testing.assert_close(weight, module.qkv.weight, atol=0, rtol=0)
                self.assertEqual(module._contrib_from_units(x, 1).count_nonzero().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_vit_shape_cuda_single_startup_budget_and_merge(self):
        for strength in (0, .5):
            module = attention.Attention_LoRA(dim=768, num_heads=12, qkv_bias=True, r=64, n_tasks=2).cuda()
            module._init_params(dict(use_slora=True, use_plora=True, dual_mask_branch_layout='single',
                dual_mask_permission_mode='symmetric_soft', dual_mask_fixed_protect_strength=strength,
                dual_mask_private_rank=40, dual_mask_fixed_coverage=.825, dual_mask_fixed_conflict_strength=.5,
                dual_mask_conflict_exact_topk=True, dual_mask_conflict_score_mode='magnitude',
                dual_mask_conflict_ratio=.1, dual_mask_task0_gate_mode='unmasked', seed=1993))
            module.layer_idx = 2
            module.w0_importance.uniform_()
            module.before_task(1)
            module.set_task_and_stage(1, 2)
            self.assertEqual(module.S_lora[1].r, 104)
            x = torch.randn(2, 5, 768, device='cuda')
            module._contrib_from_units(x, 1).sum().backward()
            self.assertGreater(module.S_lora[1].B_weight.grad.norm().item(), 0)
            self.assertIsNone(module.S_lora[1].A_weight.grad)
            with torch.no_grad():
                module.S_lora[1].B_weight.normal_(std=.01)
                raw = module.S_lora[1].B_weight @ module.S_lora[1].A_weight
                _, selected = module._joint_conflict(raw)
                self.assertEqual(int(selected.sum()), int(.1 * raw.numel()))
                before = module.qkv(x) + module._contrib_from_units(x, 1)
                module.after_task(1)
                torch.testing.assert_close(before, module.qkv(x), atol=5e-5, rtol=5e-5)
                self.assertIsNone(module.S_lora[1])
                self.assertIsNone(module.P_lora[1])


if __name__ == '__main__':
    unittest.main()
