import copy
import json
from pathlib import Path
import subprocess
import types
from types import SimpleNamespace
import unittest

import torch

from models.attention import _exact_top_ratio_mask, _normalize_score
from test.test_compact_dualmask import CompactDualMaskTests
from utils.dualmask_core_audit import epoch_updates


class RelativeConflictTests(unittest.TestCase):
    def make(self, mode='wpre_relative', task=1):
        module = CompactDualMaskTests().template()
        module.dual_mask_conflict_score_mode = mode
        with torch.no_grad():
            module.pretrained_weight.copy_(torch.arange(1, 13).float()[:, None].expand(12, 4))
            module.qkv.weight.copy_(module.pretrained_weight.flip(0))
        module.before_task(task)
        module.set_task_and_stage(task, 2)
        return module

    def test_score_uses_raw_update_and_reference_row_rms(self):
        for mode, reference in (('wpre_relative', 'pretrained_weight'), ('task_relative', 'qkv')):
            module = self.make(mode)
            weight = module.pretrained_weight if reference == 'pretrained_weight' else module.qkv.weight
            delta = torch.arange(1, 49).reshape(12, 4).float().requires_grad_()
            expected = _normalize_score(delta.detach().abs() / weight.detach().square().mean(1, keepdim=True).sqrt())
            score, selected = module._joint_conflict(delta)
            torch.testing.assert_close(score, expected)
            torch.testing.assert_close(selected, _exact_top_ratio_mask(expected, .1))
            self.assertFalse(score.requires_grad)
            self.assertEqual(int(selected.sum()), 4)

    def test_reference_cached_without_rng_and_refreshed_at_task_start(self):
        module = self.make('task_relative')
        reference = module.relative_conflict_scale.clone()
        rng = torch.get_rng_state().clone()
        module._joint_conflict(torch.ones(12, 4))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with torch.no_grad():
            module.qkv.weight.mul_(2)
        torch.testing.assert_close(reference, module.relative_conflict_scale)
        module.before_task(1)
        torch.testing.assert_close(module.relative_conflict_scale, 2 * reference)
        module.dual_mask_conflict_score_mode = 'wpre_relative'
        module.before_task(1)
        torch.testing.assert_close(module.relative_conflict_scale,
                                   module.pretrained_weight.square().mean(1, keepdim=True).sqrt())

    def test_task0_initialization_gradient_and_default_are_unchanged(self):
        reference = self.make('magnitude', task=0)
        for mode in ('wpre_relative', 'task_relative'):
            candidate = copy.deepcopy(reference)
            candidate.dual_mask_conflict_score_mode = mode
            state = torch.get_rng_state().clone()
            reference.before_task(0)
            final = torch.get_rng_state().clone()
            torch.set_rng_state(state)
            candidate.before_task(0)
            self.assertTrue(torch.equal(final, torch.get_rng_state()))
            self.assertIsNone(candidate.relative_conflict_scale)
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(candidate.state_dict()[key], value, atol=0, rtol=0)
            x = torch.randn(3, 4)
            first, second = [m._contrib_from_units(x, 0) for m in (reference, candidate)]
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            first.sum().backward()
            second.sum().backward()
            torch.testing.assert_close(reference.S_lora[0].B_weight.grad, candidate.S_lora[0].B_weight.grad)
        reference.before_task(1)
        self.assertIsNone(reference.relative_conflict_scale)
        delta = torch.randn(12, 4)
        score, selected = reference._joint_conflict(delta)
        torch.testing.assert_close(score, _normalize_score(delta.abs()), atol=0, rtol=0)
        torch.testing.assert_close(selected, _exact_top_ratio_mask(score, .1), atol=0, rtol=0)

    def test_zero_rows_ties_and_b_gradient_are_finite(self):
        for mode in ('wpre_relative', 'task_relative'):
            module = self.make(mode)
            with torch.no_grad():
                module.pretrained_weight.zero_()
                module.qkv.weight.zero_()
            module.before_task(1)
            for delta in (torch.zeros(12, 4), torch.ones(12, 4)):
                score, selected = module._joint_conflict(delta)
                self.assertTrue(torch.isfinite(score).all())
                self.assertEqual(int(selected.sum()), 4)
            module._contrib_from_units(torch.randn(3, 4), 1).sum().backward()
            for unit in (module.S_lora[1], module.P_lora[1]):
                self.assertTrue(torch.isfinite(unit.B_weight.grad).all())
                self.assertGreater(float(unit.B_weight.grad.norm()), 0)

    def test_forward_merge_once_and_readonly_telemetry(self):
        for mode in ('wpre_relative', 'task_relative'):
            module = self.make(mode).double()
            module.args['dual_mask_permission_mode'] = 'asymmetric'
            with torch.no_grad():
                for unit in (module.S_lora[1], module.P_lora[1]):
                    unit.B_weight.normal_()
            x = torch.randn(4, 4, dtype=torch.float64)
            before = module.qkv(x) + module._contrib_from_units(x, 1)
            reference = copy.deepcopy(module)
            rng = torch.get_rng_state().clone()
            learner = SimpleNamespace(_cur_task=1, args=module.args, _iter_lora_modules=lambda: [module])
            with self.assertLogs(level='INFO') as logged:
                epoch_updates(learner, 1)
            rows = [json.loads(line.split('CoreEpochUpdate ', 1)[1]) for line in logged.output
                    if 'CoreEpochUpdate ' in line]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(0 <= row['magnitude_mask_jaccard'] <= 1 for row in rows))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for current in (reference, module):
                current._contrib_from_units(x, 1).square().sum().backward()
            torch.testing.assert_close(reference.P_lora[1].B_weight.grad, module.P_lora[1].B_weight.grad)
            p_delta = module.P_lora[1].B_weight @ module.P_lora[1].A_weight
            self.assertEqual(int((module._safe_delta(p_delta, True) * module.general_mask).count_nonzero()), 0)
            module.after_task(1)
            torch.testing.assert_close(before, module.qkv(x), atol=1e-9, rtol=1e-9)
            weight = module.qkv.weight.clone()
            module.after_task(1)
            torch.testing.assert_close(weight, module.qkv.weight, atol=0, rtol=0)

    def test_default_matches_actual_historical_training_source(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(['git', 'show',
            'f43ccf986a7a1c18316f46132e64f11841ae7123:models/attention.py'], cwd=root, text=True)
        legacy = types.ModuleType('legacy_attention')
        exec(compile(source, 'legacy_attention.py', 'exec'), legacy.__dict__)
        for score in ('conflict', 'magnitude'):
            current = self.make(score, task=0)
            old = copy.deepcopy(current)
            old.__class__ = legacy.Attention_LoRA
            for task in (0, 1):
                rng = torch.get_rng_state().clone()
                current.before_task(task)
                final = torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                old.before_task(task)
                self.assertTrue(torch.equal(final, torch.get_rng_state()))
                for module in (current, old):
                    module.set_task_and_stage(task, 2)
                for _ in range(3):
                    x = torch.randn(3, 4)
                    actual = current.qkv(x) + current._contrib_from_units(x, task)
                    expected = old.qkv(x) + old._contrib_from_units(x, task)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    actual.square().mean().backward()
                    expected.square().mean().backward()
                    for (_, first), (_, second) in zip(current.named_parameters(), old.named_parameters()):
                        if first.grad is not None:
                            torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)
                            with torch.no_grad():
                                first.add_(first.grad, alpha=-.02)
                                second.add_(second.grad, alpha=-.02)
                    current.zero_grad()
                    old.zero_grad()
                current.after_task(task)
                old.after_task(task)
                torch.testing.assert_close(current.qkv.weight, old.qkv.weight, atol=0, rtol=0)

    def test_merge_uses_raw_factor_for_tied_relative_scores(self):
        with torch.random.fork_rng():
            torch.manual_seed(3)
            for mode in ('wpre_relative', 'task_relative'):
                module = self.make(mode)
                module.args['dual_mask_permission_mode'] = 'asymmetric'
                with torch.no_grad():
                    module.pretrained_weight.uniform_(.01, .1)
                    module.qkv.weight.uniform_(.01, .1)
                module.before_task(1)
                with torch.no_grad():
                    module.P_lora[1].A_weight.fill_(1)
                    module.P_lora[1].B_weight.copy_(module.relative_conflict_scale)
                x = torch.eye(4)
                before = module.qkv(x) + module._contrib_from_units(x, 1)
                module.after_task(1)
                torch.testing.assert_close(before, module.qkv(x), atol=1e-7, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
