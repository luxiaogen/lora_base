"""任务起点谱位置和冲突来源的梯度、缓存与交付回归。"""
import copy
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from test import test_protection_search as fixtures
from utils.protect_position import permute_protect_mask


class TaskStartPriorTests(unittest.TestCase):
    def module(self, position='wpre', source='wpre', score='magnitude', task=1):
        module = fixtures.ProtectionSearchTests().module('E1')
        module.args.update(dual_mask_task_start_probe=True,
                          dual_mask_protect_position=position,
                          dual_mask_conflict_importance_source=source)
        module.dual_mask_conflict_score_mode = score
        module.before_task(task)
        return module

    def test_task_start_protect_uses_new_score_and_reference_counts(self):
        reference = self.module()
        candidate = copy.deepcopy(reference)
        candidate.args['dual_mask_protect_position'] = 'task_start_svd'
        candidate.rebuild_dual_masks()
        self.assertTrue(hasattr(candidate, 'task_start_importance'))
        for score, mask, original in zip(candidate.task_start_importance.chunk(3),
                                       candidate.general_mask.chunk(3), reference.general_mask.chunk(3)):
            count = int(original.sum())
            expected = torch.zeros_like(mask).flatten()
            expected[score.flatten().argsort(descending=True, stable=True)[:count]] = 1
            torch.testing.assert_close(mask, expected.reshape_as(mask))
        torch.testing.assert_close(candidate.w0_importance, reference.w0_importance, rtol=0, atol=0)
        torch.testing.assert_close(candidate.isolated_mask, 1 - candidate.general_mask)

    def test_cache_is_task_fixed_refreshes_and_preserves_wpre_metadata(self):
        module = self.module()
        first = module.task_start_importance.clone()
        original = module.pretrained_weight.clone(), module.w0_importance.clone()
        metadata = module.last_svd_rank, module.last_svd_energy_coverage
        with torch.no_grad():
            module.qkv.weight[0].mul_(9)
        module.S_lora.append(None)
        module.P_lora.append(None)
        with patch.object(module, '_svd_importance', wraps=module._svd_importance) as svd:
            module.rebuild_dual_masks()
            self.assertEqual(svd.call_count, 0)
            torch.testing.assert_close(first, module.task_start_importance)
            module.before_task(2)
            self.assertEqual(svd.call_count, 1)
        self.assertFalse(torch.equal(first, module.task_start_importance))
        torch.testing.assert_close(module.pretrained_weight, original[0])
        torch.testing.assert_close(module.w0_importance, original[1])
        self.assertEqual(metadata, (module.last_svd_rank, module.last_svd_energy_coverage))
        self.assertNotIn('task_start_importance', module.state_dict())

    def test_permutation_is_once_fixed_and_does_not_draw_training_rng(self):
        module = self.module('task_start_svd_permuted')
        state = torch.get_rng_state().clone()
        module.rebuild_dual_masks()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(module.general_mask,
            permute_protect_mask(module.task_start_protect, 1993, module.layer_idx))
        torch.testing.assert_close(module.task_start_permuted_importance,
            permute_protect_mask(module.task_start_importance, 1993, module.layer_idx))
        self.assertEqual(module.core_diagnostic_positions,
                         ('task_start_svd', 'task_start_svd_permuted'))

    def test_four_rankings_exact_counts_and_use_distinct_sources(self):
        module = self.module()
        raw = torch.arange(1., 49.).reshape(12, 4).flip(0)
        importance = torch.arange(48.).reshape(12, 4)
        module.task_start_importance = importance
        module.task_start_permuted_importance = importance.flip(0)
        module.w0_importance.copy_(importance.flip(1))
        from models.attention import _exact_top_ratio_mask, _normalize_score
        for score, source in (('magnitude', 'wpre'), ('conflict', 'wpre'),
                              ('conflict', 'task_start'), ('conflict', 'task_start_permuted')):
            module.dual_mask_conflict_score_mode = score
            module.args['dual_mask_conflict_importance_source'] = source
            value = _normalize_score(raw.abs())
            if score == 'conflict':
                base = {'wpre': module.w0_importance, 'task_start': importance,
                        'task_start_permuted': importance.flip(0)}[source]
                value = _normalize_score(base * value)
            actual_score, selected = module._joint_conflict(raw)
            torch.testing.assert_close(actual_score, value)
            torch.testing.assert_close(selected, _exact_top_ratio_mask(value, .1))
            self.assertEqual(int(selected.sum()), 4)
        for raw in (torch.zeros_like(raw), torch.ones_like(raw)):
            self.assertEqual(int(module._joint_conflict(raw)[1].sum()), 4)

    def test_all_variants_zero_gradient_permissions_budget_merge_once(self):
        from utils.protection_strength import protection_update
        cells = [('wpre', 'wpre', 'magnitude'), ('task_start_svd', 'wpre', 'magnitude'),
                 ('task_start_svd_permuted', 'wpre', 'magnitude'),
                 ('wpre', 'wpre', 'conflict'), ('wpre', 'task_start', 'conflict'),
                 ('wpre', 'task_start_permuted', 'conflict')]
        for position, source, score in cells:
            module = self.module(position, source, score)
            module.set_task_and_stage(1, 0)
            x = torch.randn(3, 4)
            module._contrib_from_units(x, 1).sum().backward()
            for unit in (module.S_lora[1], module.P_lora[1]):
                self.assertGreater(float(unit.B_weight.grad.norm()), 0)
                with torch.no_grad():
                    unit.B_weight.normal_(std=.01)
            raw = module.S_lora[1].B_weight @ module.S_lora[1].A_weight
            state = protection_update(module, raw)
            mask = module.general_mask
            self.assertLessEqual(float((state['safe'].detach() * mask).norm()),
                                 float((state['safe'].detach() * (1 - mask)).norm()) + 1e-6)
            self.assertFalse(state['alpha'].requires_grad)
            p = module.P_lora[1].B_weight @ module.P_lora[1].A_weight
            self.assertEqual(float((module._safe_delta(p, True).detach() * mask).abs().max()), 0)
            before = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
            module.after_task(1)
            torch.testing.assert_close(before, F.linear(x, module.qkv.weight, module.qkv.bias),
                                       rtol=1e-5, atol=2e-6)
            once = module.qkv.weight.clone()
            module.after_task(1)
            torch.testing.assert_close(once, module.qkv.weight, rtol=0, atol=0)

    def test_default_and_task0_forward_rng_and_weights_match_old_source(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(['git', 'show',
            '2832659024419b5813bb0472eb5e9874baf28bf5:models/attention.py'], cwd=root, text=True)
        namespace = {'__name__': 'old_attention'}
        exec(compile(source, '<old_attention>', 'exec'), namespace)
        for task in (0, 1):
            current = fixtures.ProtectionSearchTests().module('E1')
            previous = copy.deepcopy(current)
            previous.__class__ = namespace[current.__class__.__name__]
            if task == 0:
                current.args.update(dual_mask_task_start_probe=True,
                    dual_mask_protect_position='task_start_svd',
                    dual_mask_conflict_importance_source='task_start')
            state = torch.get_rng_state().clone()
            previous.before_task(task)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(state)
            current.before_task(task)
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            for key, value in previous.state_dict().items():
                torch.testing.assert_close(value, current.state_dict()[key], rtol=0, atol=0)
            for a, b in zip(previous.parameters(), current.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            x = torch.randn(3, 4)
            values = [m._contrib_from_units(x, task) for m in (previous, current)]
            torch.testing.assert_close(*values, rtol=0, atol=0)
            for value in values:
                value.square().mean().backward()
            for a, b in zip(previous.parameters(), current.parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            for module in (previous, current):
                optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.02)
                optimizer.step()
                module.after_task(task)
            torch.testing.assert_close(previous.qkv.weight, current.qkv.weight, rtol=0, atol=0)

    def test_task_start_diagnostic_labels_and_next_training_step_are_preserved(self):
        from test import test_dualmask_core_audit as audit
        from utils.dualmask_core_audit import position_diagnostic
        first = audit.TinyNetwork()
        first.attention = self.module('task_start_svd')
        first.attention.args['dual_mask_budget_audit'] = True
        with torch.no_grad():
            first.attention.S_lora[1].B_weight.normal_()
            first.attention.P_lora[1].B_weight.normal_()
        second = copy.deepcopy(first)
        learner = SimpleNamespace(_network=first, _device=torch.device('cpu'),
            args={'dual_mask_budget_audit': True}, run_epoch=20, _cur_task=1,
            _known_classes=2, _total_classes=4, batch_size=3,
            _iter_lora_modules=lambda: [first.attention])
        rng = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO') as output:
            position_diagnostic(learner, audit.Images(), 20)
        rows = [json.loads(line.split('CorePositionDiagnostic ', 1)[1])
                for line in output.output if 'CorePositionDiagnostic ' in line]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all((r['reference_position'], r['permuted_position']) ==
                           ('task_start_svd', 'task_start_svd_permuted') for r in rows))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(first.attention._core_audit_position)
        for model in (first, second):
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            optimizer.zero_grad()
            model.interface(torch.arange(12).reshape(3, 4).float()).square().mean().backward()
            optimizer.step()
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
