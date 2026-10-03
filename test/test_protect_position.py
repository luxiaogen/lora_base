import copy
import json
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from test import test_global_conflict_budget
from utils.protect_position import permute_protect_mask


class ProtectPositionTests(unittest.TestCase):
    def test_per_projection_counts_and_degree_distributions(self):
        mask = (torch.arange(192).reshape(24, 8) % 5 < 2).float()
        before = mask.clone()
        permuted = permute_protect_mask(mask, seed=1993, layer=3)
        for source, actual in zip(mask.chunk(3), permuted.chunk(3)):
            self.assertEqual(source.sum().item(), actual.sum().item())
            torch.testing.assert_close(source.sum(0).sort().values, actual.sum(0).sort().values)
            torch.testing.assert_close(source.sum(1).sort().values, actual.sum(1).sort().values)
        self.assertTrue(torch.equal(mask, before))
        self.assertFalse(torch.equal(mask, permuted))
        self.assertTrue(torch.equal(permuted, permute_protect_mask(mask, 1993, 3)))
        self.assertFalse(torch.equal(permuted, permute_protect_mask(mask, 1993, 4)))

    def test_private_rng_and_fixed_mapping_across_tasks(self):
        score = torch.arange(192).reshape(24, 8)
        first, second = (score > 90).float(), (score > 120).float()
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        first = permute_protect_mask(first, 1993, 3)
        second = permute_protect_mask(second, 1993, 3)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        if cuda_rng is not None:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())))
        self.assertTrue(torch.all(second <= first))

    def make(self, mode='wpre', task=1, audit=False, score='conflict'):
        module = test_global_conflict_budget.GlobalBudgetSelectionTests._make_attention('layer',
            dual_mask_protect_position=mode, dual_mask_position_audit=audit,
            dual_mask_conflict_score_mode=score, dual_mask_task0_gate_mode='unmasked', seed=1993)
        module.layer_idx = 2
        module.cur_task = task
        importance = torch.arange(48).reshape(12, 4).float() / 47
        module.w0_importance.copy_(importance)
        module._combined_importance = lambda: importance.clone()
        return module

    def test_default_and_task0_paths_are_unchanged(self):
        explicit = self.make()
        default = copy.deepcopy(explicit)
        default.args.pop('dual_mask_protect_position')
        explicit.rebuild_dual_masks()
        default.rebuild_dual_masks()
        self.assertTrue(torch.equal(explicit.general_mask, default.general_mask))
        for mode in ('wpre', 'permuted'):
            module = self.make(mode, task=0)
            reference = copy.deepcopy(module)
            reference.args['dual_mask_protect_position'] = 'wpre'
            rng = torch.get_rng_state().clone()
            module.before_task(0)
            state = torch.get_rng_state().clone()
            torch.set_rng_state(rng)
            reference.before_task(0)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            for key, value in reference.state_dict().items():
                self.assertTrue(torch.equal(value, module.state_dict()[key]), key)
            x = torch.randn(5, 4)
            torch.testing.assert_close(module._contrib_from_units(x, 0), reference._contrib_from_units(x, 0))

    def test_only_protection_placement_changes(self):
        reference, candidate = self.make(), self.make('permuted')
        rng = torch.get_rng_state().clone()
        reference.rebuild_dual_masks()
        candidate.rebuild_dual_masks()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        expected = permute_protect_mask(reference.general_mask, 1993, 2)
        self.assertTrue(torch.equal(expected, candidate.general_mask))
        self.assertTrue(torch.equal(candidate.isolated_mask, 1 - candidate.general_mask))
        self.assertTrue(torch.equal(reference.w0_importance, candidate.w0_importance))
        for name in ('current_private_rank', 'effective_protect_strength', 'effective_energy_coverage'):
            self.assertEqual(getattr(reference, name), getattr(candidate, name))
        candidate.cur_task = 2
        candidate.rebuild_dual_masks()
        self.assertTrue(torch.equal(expected, candidate.general_mask))

    def test_initialization_and_freezing_are_identical(self):
        reference, candidate = self.make(), self.make('permuted')
        rng = torch.get_rng_state().clone()
        reference.before_task(1)
        final_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        candidate.before_task(1)
        self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))
        for module in (reference, candidate):
            module.set_task_and_stage(1, 2)
        for units in zip((reference.S_lora[1], reference.P_lora[1]),
                         (candidate.S_lora[1], candidate.P_lora[1])):
            self.assertTrue(torch.equal(units[0].A_weight, units[1].A_weight))
            self.assertTrue(torch.equal(units[0].B_weight, units[1].B_weight))
            self.assertFalse(units[1].A_weight.requires_grad)
            self.assertTrue(units[1].B_weight.requires_grad)

    def test_ranking_switch_does_not_change_regularization_weights(self):
        original = self.make()
        original.rebuild_dual_masks()
        original.dual_mask_s_conflict_enabled = False
        original.dual_mask_p_conflict_enabled = False
        candidate = copy.deepcopy(original)
        candidate.dual_mask_conflict_score_mode = 'magnitude'
        candidate.args['dual_mask_conflict_reg_original_score'] = True
        for isolated in (False, True):
            first = torch.arange(48).reshape(12, 4).float().flip(0).requires_grad_()
            second = first.detach().clone().requires_grad_()
            original_loss = original._joint_conflict_regularization(
                SimpleNamespace(A_weight=torch.eye(4), B_weight=first), isolated)
            candidate_loss = candidate._joint_conflict_regularization(
                SimpleNamespace(A_weight=torch.eye(4), B_weight=second), isolated)
            torch.testing.assert_close(original_loss, candidate_loss, atol=0, rtol=0)
            original_loss.backward()
            candidate_loss.backward()
            torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)

    def test_four_cells_forward_merge_and_actual_update_diagnostics(self):
        for position in ('wpre', 'permuted'):
            for score in ('conflict', 'magnitude'):
                with self.subTest(position=position, score=score):
                    module = self.make(position, audit=True, score=score)
                    module.before_task(1)
                    module.set_task_and_stage(1, 2)
                    with torch.no_grad():
                        for unit in (module.S_lora[1], module.P_lora[1]):
                            unit.B_weight.normal_()
                    x = torch.randn(5, 4)
                    before = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
                    raw = module.plora_gamma * (module.P_lora[1].B_weight @ module.P_lora[1].A_weight)
                    safe = module._safe_delta(raw, True)
                    rng = torch.get_rng_state().clone()
                    with self.assertLogs(level='INFO') as logs:
                        module.after_task(1)
                    rows = [json.loads(line.split('ProtectionPositionUpdate ', 1)[1])
                            for line in logs.output if 'ProtectionPositionUpdate ' in line]
                    self.assertEqual(len(rows), 6)
                    for index, row in enumerate(r for r in rows if r['branch'] == 'P'):
                        self.assertAlmostEqual(row['raw_norm'], raw.chunk(3)[index].norm().item(), places=6)
                        self.assertAlmostEqual(row['effective_norm'], safe.chunk(3)[index].norm().item(), places=6)
                        self.assertAlmostEqual(row['total_removed_norm'], (raw - safe).chunk(3)[index].norm().item(), places=6)
                        self.assertLess(row['merge_error'], 1e-6)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    torch.testing.assert_close(before, F.linear(x, module.qkv.weight, module.qkv.bias), atol=2e-6, rtol=1e-5)
                    self.assertEqual(module._contrib_from_units(x, 1).count_nonzero().item(), 0)
                    self.assertIsNone(module.S_lora[1])
                    self.assertIsNone(module.P_lora[1])

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_real_vit_shape_forward_and_merge_on_cuda(self):
        from models.attention import Attention_LoRA
        for position in ('wpre', 'permuted'):
            for score in ('conflict', 'magnitude'):
                with self.subTest(position=position, score=score):
                    module = Attention_LoRA(dim=768, num_heads=12, qkv_bias=True, r=64, n_tasks=2).cuda()
                    module._init_params(dict(use_slora=True, use_plora=True,
                        dual_mask_protect_position=position, dual_mask_conflict_score_mode=score,
                        dual_mask_conflict_energy_adaptive=False, dual_mask_conflict_ratio=.1,
                        dual_mask_competence_adaptive=False, dual_mask_task0_gate_mode='unmasked', seed=1993))
                    module.layer_idx = 2
                    module.w0_importance.uniform_()
                    module.before_task(1)
                    module.set_task_and_stage(1, 2)
                    with torch.no_grad():
                        for unit in (module.S_lora[1], module.P_lora[1]):
                            unit.B_weight.normal_(std=.01)
                        x = torch.randn(2, 5, 768, device='cuda')
                        before = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
                        module.after_task(1)
                        after = F.linear(x, module.qkv.weight, module.qkv.bias)
                    torch.testing.assert_close(before, after, atol=5e-5, rtol=5e-5)
                    self.assertIsNone(module.S_lora[1])
                    self.assertIsNone(module.P_lora[1])


if __name__ == '__main__':
    unittest.main()
