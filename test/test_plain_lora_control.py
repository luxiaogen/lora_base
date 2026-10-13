"""普通 LoRA 对照的参数预算、Task0、梯度与交付回归。"""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

from test import test_protect_position as position_fixtures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class PlainLoRAControlTests(unittest.TestCase):
    def template(self, plain=False):
        module = position_fixtures.ProtectPositionTests().make(score='magnitude')
        module.slora_gamma, module.plora_gamma = .5, .75
        module.dual_mask_private_rank = module.rank
        module.args.update(dual_mask_permission_mode='asymmetric',
                           dual_mask_fixed_protect_strength=.5)
        if plain:
            module.args.update(dual_mask_branch_layout='plain_lora',
                               dual_mask_permission_mode='symmetric_soft',
                               dual_mask_fixed_protect_strength=0)
            module.dual_mask_s_conflict_enabled = False
            module.dual_mask_p_conflict_enabled = False
        return module

    def pair(self, task=1):
        first = self.template()
        second = copy.deepcopy(first)
        second.args.update(dual_mask_branch_layout='plain_lora',
                           dual_mask_permission_mode='symmetric_soft',
                           dual_mask_fixed_protect_strength=0)
        second.dual_mask_s_conflict_enabled = False
        second.dual_mask_p_conflict_enabled = False
        initial = torch.get_rng_state().clone()
        first.before_task(task)
        final = torch.get_rng_state().clone()
        torch.set_rng_state(initial)
        second.before_task(task)
        self.assertTrue(torch.equal(final, torch.get_rng_state()))
        for module in (first, second):
            module.set_task_and_stage(task, 2)
        return first, second

    def test_task0_weights_gradients_anchor_and_rng_unchanged(self):
        original, plain = self.pair(task=0)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, plain.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(original.S_lora[0].r, plain.S_lora[0].r)
        with torch.no_grad():
            original.S_lora[0].B_weight.normal_()
            plain.S_lora[0].B_weight.copy_(original.S_lora[0].B_weight)
        x = torch.randn(5, 4)
        outputs = [m._contrib_from_units(x, 0) for m in (original, plain)]
        torch.testing.assert_close(*outputs, atol=0, rtol=0)
        torch.testing.assert_close(original.anchor_regularization(),
                                   plain.anchor_regularization(), atol=0, rtol=0)
        for value in outputs:
            value.square().mean().backward()
        for name in ('A_weight', 'B_weight'):
            torch.testing.assert_close(getattr(original.S_lora[0], name).grad,
                                       getattr(plain.S_lora[0], name).grad, atol=0, rtol=0)

    def test_ab_trainable_budget_and_initialization(self):
        original, plain = self.pair()
        unit = plain.S_lora[1]
        self.assertIsNone(plain.P_lora[1])
        self.assertEqual(unit.r, 3 * (original.rank + original.dual_mask_private_rank) // 4)
        expected = torch.cat((original.S_lora[1].A_weight, original.P_lora[1].A_weight))[:unit.r]
        torch.testing.assert_close(unit.A_weight, expected, atol=0, rtol=0)
        self.assertEqual(unit.B_weight.count_nonzero().item(), 0)
        self.assertTrue(unit.A_weight.requires_grad)
        self.assertTrue(unit.B_weight.requires_grad)
        self.assertEqual(sum(p.numel() for p in plain.parameters() if p.requires_grad),
                         sum(p.numel() for p in original.parameters() if p.requires_grad))
        self.assertEqual(96 * (768 + 2304) * 12, 2 * 64 * 2304 * 12)

    def test_unmasked_update_gradients_and_merge_once(self):
        _, module = self.pair()
        unit = module.S_lora[1]
        x = torch.randn(5, 4)
        module._contrib_from_units(x, 1).sum().backward()
        self.assertGreater(unit.B_weight.grad.norm().item(), 0)
        self.assertEqual(unit.A_weight.grad.norm().item(), 0)
        unit.zero_grad()
        with torch.no_grad():
            unit.B_weight.normal_()
        expected = .5 * F.linear(F.linear(x, unit.A_weight), unit.B_weight)
        actual = module._contrib_from_units(x, 1)
        torch.testing.assert_close(actual, expected)
        expected_grads = torch.autograd.grad(expected.square().mean(), (unit.A_weight, unit.B_weight))
        actual_grads = torch.autograd.grad(actual.square().mean(), (unit.A_weight, unit.B_weight))
        for first, second in zip(expected_grads, actual_grads):
            torch.testing.assert_close(first, second)
        before = module.qkv(x) + actual
        module.after_task(1)
        torch.testing.assert_close(before, module.qkv(x), atol=2e-6, rtol=1e-5)
        merged = module.qkv.weight.detach().clone()
        module.after_task(1)
        torch.testing.assert_close(module.qkv.weight, merged, atol=0, rtol=0)
        self.assertEqual(module._contrib_from_units(x, 1).count_nonzero().item(), 0)

    def test_only_one_declared_run_and_recipe_difference(self):
        runner = importlib.import_module('run_plain_lora_control')
        runner.validate_settings('3090')
        self.assertEqual(runner.modes('3090'), ['imgr10_PlainLoRA_seed1993'])
        original = runner.final_f.settings_for('3090', 'imgr10_C11_seed1993')
        candidate = runner.settings_for('3090', runner.MODE)
        changed = {key for key in original.keys() | candidate.keys()
                   if original.get(key) != candidate.get(key)}
        self.assertEqual(changed, {'dual_mask_branch_layout', 'dual_mask_permission_mode',
            'dual_mask_fixed_protect_strength', 'dual_mask_s_conflict_enabled',
            'dual_mask_p_conflict_enabled', 'wandb_group'})
        smoke = runner.settings_for('3090', runner.MODE, True)
        self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2, 1, 1))
        with self.assertRaises(ValueError):
            runner.validate_settings('5090')

    def test_plain_configuration_rejects_gates_and_staged_training(self):
        runner = importlib.import_module('run_plain_lora_control')
        config = runner.effective_config('3090', runner.MODE)
        for changed in ({'dual_mask_s_conflict_enabled': True},
                        {'dual_mask_reg_weight': .01}, {'sp_staged_s_epochs': 2}):
            with self.assertRaises(ValueError):
                runner.normalize_dualmask_config(dict(config, **changed))

    def test_single_seed_summary_handles_missing_and_complete_results(self):
        runner = importlib.import_module('run_plain_lora_control')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            pending = dict(mode=runner.MODE, status='pending')
            runner.summarize_safely(directory, '3090', [pending])
            self.assertEqual(json.loads((directory / 'results.json').read_text()), [pending])
            path = directory / runner.MODE
            path.mkdir()
            (path / 'run.json').write_text(json.dumps(dict(phase='formal',
                effective_config={'dual_mask_branch_layout': 'plain_lora'})))
            lines = []
            for task in range(10):
                lines.append('[trainer.py] => CNN: ' + repr(dict(total=80., old=79., new=81.)))
                for epoch in range(1, 21):
                    for layer in range(12):
                        for projection in ('Q', 'K', 'V'):
                            lines.append('CoreEpochUpdate ' + json.dumps(dict(task=task,
                                epoch=epoch, layer=layer, projection=projection,
                                branch='S' if task == 0 else 'Single')))
            lines += ['[trainer.py] => Average Accuracy: 80.', '[trainer.py] => Forgetting: 1.']
            (path / 'training.log').write_text('\n'.join(lines))
            runner.summarize_safely(directory, '3090', [dict(pending, status='completed', exit_code=0)])
            result = json.loads((directory / 'results.json').read_text())[0]
            self.assertTrue(result['valid_performance'])
            self.assertTrue(result['epoch_records_complete'])
            self.assertIsNone(result['std'])
            self.assertFalse(result['complete_three_seeds'])


if __name__ == '__main__':
    unittest.main()
