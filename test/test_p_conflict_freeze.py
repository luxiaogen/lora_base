"""Freeze coordinates, not LoRA parameters; keep forward and merge identical."""
import copy
import json
from pathlib import Path
import shlex
import subprocess
import unittest

import torch
from torch.nn import functional as F

class PConflictFreezeTests(unittest.TestCase):
    def make_attention(self, freeze=5, task=1):
        from test.test_global_conflict_budget import GlobalBudgetSelectionTests
        module = GlobalBudgetSelectionTests._make_attention(
            'layer', p_conflict_freeze_epoch=freeze,
            dual_mask_conflict_exact_topk=True, dual_mask_conflict_ratio=.1)
        module.before_task(task)
        module.set_task_and_stage(task, 0)
        with torch.no_grad():
            module.general_mask.zero_()
            module.general_mask[:, 3] = 1
            module.w0_importance.fill_(1)
            for unit in (module.S_lora[task], module.P_lora[task]):
                unit.A_weight.copy_(torch.eye(2, 4))
                unit.A_weight.requires_grad_(False)
                unit.B_weight.copy_(torch.arange(24).reshape(12, 2) / 24)
        return module

    def update(self, module, epoch):
        self.assertTrue(hasattr(module, 'update_p_conflict_freeze'), 'epoch-end freeze hook is missing')
        module.update_p_conflict_freeze(epoch)

    def p_mask(self, module):
        unit = module.P_lora[module.cur_task]
        return module._safe_delta(unit.B_weight @ unit.A_weight, True, return_details=True)[2]

    def test_freeze_after_fifth_epoch_not_before_and_only_p(self):
        module = self.make_attention()
        self.update(module, 4)
        self.assertIsNone(module.frozen_p_conflict_mask)
        self.update(module, 5)
        saved_mask = self.p_mask(module).clone()
        reference = copy.deepcopy(module)
        reference.frozen_p_conflict_mask = None
        with torch.no_grad():
            for target in (module, reference):
                target.P_lora[1].B_weight.copy_(target.P_lora[1].B_weight.flip(0))
                target.S_lora[1].B_weight.copy_(target.S_lora[1].B_weight.flip(0))
        self.assertTrue(torch.equal(saved_mask, self.p_mask(module)))
        self.assertFalse(torch.equal(saved_mask, self.p_mask(reference)))
        x = torch.randn(3, 4)
        self.assertTrue(torch.equal(module._masked_unit_forward(x, module.S_lora[1], False),
                                    reference._masked_unit_forward(x, reference.S_lora[1], False)))

    def test_both_b_still_train_with_frozen_mask(self):
        module = self.make_attention()
        self.update(module, 5)
        mask = self.p_mask(module).clone()
        units = [module.S_lora[1], module.P_lora[1]]
        before = [u.B_weight.detach().clone() for u in units]
        optimizer = torch.optim.SGD([u.B_weight for u in units], lr=.02)
        module._contrib_from_units(torch.randn(6, 4), 1).square().mean().backward()
        optimizer.step()
        for unit, initial in zip(units, before):
            self.assertIsNotNone(unit.B_weight.grad)
            self.assertFalse(torch.equal(initial, unit.B_weight))
        self.assertTrue(torch.equal(mask, self.p_mask(module)))

    def test_adaptive_energy_mask_matches_forward_after_capture_and_merge(self):
        module = self.make_attention()
        module.dual_mask_conflict_exact_topk = False
        module.dual_mask_conflict_energy_adaptive = True
        module.dual_mask_conflict_energy_ratio_floor = True
        self.update(module, 5)
        frozen = self.p_mask(module).clone()
        with torch.no_grad():
            module.P_lora[1].B_weight.mul_(1.7)
        self.assertTrue(torch.equal(frozen, self.p_mask(module)))
        x = torch.randn(7, 4)
        before = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
        module.after_task(1)
        torch.testing.assert_close(before, F.linear(x, module.qkv.weight, module.qkv.bias), atol=1e-6, rtol=1e-5)

    def test_frozen_mask_used_in_merge_and_cleared_for_next_task(self):
        module = self.make_attention()
        self.update(module, 5)
        with torch.no_grad():
            module.P_lora[1].B_weight.copy_(module.P_lora[1].B_weight.flip(0))
        x = torch.randn(7, 4)
        pre = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
        raw = module.plora_gamma * (module.P_lora[1].B_weight @ module.P_lora[1].A_weight)
        mask = self.p_mask(module)
        expected_overlap = (mask * (1 - module.general_mask)).sum() / mask.sum()
        base = raw * (1 - module.general_mask)
        safe = base * (1 - module._conflict_parameters()[1] * mask)
        expected_suppression = 1 - safe.norm() / base.norm()
        module.after_task(1)
        self.assertAlmostEqual(module.last_private_conflict_mask_overlap.item(), expected_overlap.item(), places=6)
        self.assertAlmostEqual(module.last_private_conflict_gate_suppression.item(), expected_suppression.item(), places=6)
        post = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
        torch.testing.assert_close(pre, post, atol=1e-6, rtol=1e-5)
        self.assertIsNone(module.frozen_p_conflict_mask)
        self.assertIsNone(module.previous_p_conflict_mask)
        merged = module.qkv.weight.clone()
        self.assertEqual(torch.count_nonzero(module._contrib_from_units(x, 1)).item(), 0)
        self.assertTrue(torch.equal(merged, module.qkv.weight))
        module.frozen_p_conflict_mask = torch.ones_like(module.general_mask, dtype=torch.bool)
        module.before_task(0)
        self.assertIsNone(module.frozen_p_conflict_mask)

    def test_disabled_and_task0_preserve_output_rng_and_state(self):
        for freeze, task in ((0, 1), (5, 0)):
            module = self.make_attention(freeze, task)
            x = torch.randn(4, 4)
            before = module._contrib_from_units(x, task).detach().clone()
            state = copy.deepcopy(module.state_dict())
            rng = torch.get_rng_state().clone()
            self.update(module, 5)
            self.assertTrue(torch.equal(before, module._contrib_from_units(x, task)))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(state[k], v) for k, v in module.state_dict().items()))
            self.assertIsNone(module.frozen_p_conflict_mask)

    def test_telemetry_detects_stale_mask_without_changing_parameters(self):
        module = self.make_attention()
        self.update(module, 5)
        with torch.no_grad():
            module.P_lora[1].B_weight.copy_(module.P_lora[1].B_weight.flip(0))
        before = module.P_lora[1].B_weight.clone()
        rng = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO') as logs:
            self.update(module, 6)
        record = json.loads(next(s.split('PConflictFreeze ', 1)[1] for s in logs.output if 'PConflictFreeze ' in s))
        self.assertTrue(record['frozen'])
        self.assertEqual(record['applied_switch_fraction'], 0)
        self.assertEqual(record['applied_jaccard_previous'], 1)
        self.assertLess(record['applied_jaccard_dynamic'], 1)
        self.assertGreater(record['dynamic_suppression_ratio'], record['applied_suppression_ratio'])
        self.assertTrue(torch.equal(before, module.P_lora[1].B_weight))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_script_is_one_three_task_candidate_without_other_methods_or_saves(self):
        output = subprocess.check_output(['bash', 'scripts/9_29_imgr10_p_mask_freeze_3090.sh', '--dry-run'], text=True)
        commands = [shlex.split(line[len('Command: '):]) for line in output.splitlines() if line.startswith('Command: ')]
        self.assertEqual(len(commands), 1)
        args = commands[0]
        settings = dict(args[i + 1].split('=', 1) for i, arg in enumerate(args) if arg == '--set')
        expected = {'seed': '[1993]', 'max_tasks': '3', 'epochs': '20', 'init_epoch': '20',
                    'ca_epochs': '5', 'p_conflict_freeze_epoch': '5', 'sp_staged_s_epochs': '0',
                    'late_weight_average_epochs': '0', 'save_task_weights': 'false',
                    'dual_mask_anchor_reg_weight': '2.5', 'disable_fused_sdpa': 'true',
                    'dual_mask_conflict_granularity': 'layer', 'dual_mask_private_conflict_mode': 'global',
                    'ca_two_centers': 'false', 'old_competition_weight': '0'}
        for key, value in expected.items():
            self.assertEqual(settings[key], value, key)
        self.assertNotIn('data_path', settings)
        spec = json.loads(Path('scripts/sweeps/imgr10_p_mask_freeze_3090.json').read_text())
        self.assertEqual(spec['variants'][0]['overrides'], {'p_conflict_freeze_epoch': 5})


if __name__ == '__main__':
    unittest.main()
