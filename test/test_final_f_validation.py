"""最终 F 的单 seed 因素、三位置整层控制与真实合并回归。"""
import copy
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from test import test_protect_position as position_fixtures
from test import test_dualmask_core_audit as audit_fixtures
from utils import dualmask_core
from utils.protect_position import update_rows

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class FinalFQueueTests(unittest.TestCase):
    def runner(self):
        self.assertIsNotNone(importlib.util.find_spec('run_final_f_validation'),
                             '最终 F 队列入口尚未实现')
        return importlib.import_module('run_final_f_validation')

    def test_only_nine_authorized_single_seed_groups(self):
        runner = self.runner()
        self.assertEqual(runner.modes('3090'), ['imgr10_' + name + '_seed1993'
            for name in ('C11', 'C01', 'C10', 'C00', 'Soft', 'Bfixed')])
        self.assertEqual(runner.modes('5090'), ['imgr10_' + name + '_seed1993'
            for name in ('Lspectrum', 'Labs', 'Lperm')])
        for machine in ('3090', '5090'):
            runner.validate_settings(machine)
            for name in runner.modes(machine):
                config = runner.effective_config(machine, name)
                for key, value in dict(seed=[1993], rank=64, dual_mask_private_rank=64,
                        dual_mask_fixed_coverage=.9, dual_mask_reg_weight=0,
                        dual_mask_conflict_exact_topk=True, dual_mask_conflict_ratio=.1,
                        dual_mask_conflict_score_mode='magnitude', dual_mask_protection_rule='static',
                        init_epoch=20, epochs=20, ca_epochs=5, save_task_weights=False,
                        dual_mask_gradient_route='off', dual_mask_prototype_position_probe=False,
                        dual_mask_position_signal_audit=False, p_fixed_support='off').items():
                    self.assertEqual(config[key], value, (name, key))
                smoke = runner.effective_config(machine, name, True)
                self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2, 1, 1))

    def test_only_predeclared_factors_change(self):
        runner = self.runner()
        first = runner.settings_for('3090', 'imgr10_C11_seed1993')
        for name, wanted in {
                'C01': {'dual_mask_permission_mode', 'dual_mask_fixed_protect_strength'},
                'C10': {'dual_mask_s_conflict_enabled', 'dual_mask_p_conflict_enabled'},
                'C00': {'dual_mask_permission_mode', 'dual_mask_fixed_protect_strength',
                        'dual_mask_s_conflict_enabled', 'dual_mask_p_conflict_enabled'},
                'Soft': {'dual_mask_permission_mode'},
                'Bfixed': {'dual_mask_fixed_conflict_strength'}}.items():
            other = runner.settings_for('3090', 'imgr10_' + name + '_seed1993')
            self.assertEqual({k for k in first.keys() | other.keys() if first.get(k) != other.get(k)}, wanted)
        first = runner.settings_for('5090', 'imgr10_Lspectrum_seed1993')
        for name in ('Labs', 'Lperm'):
            other = runner.settings_for('5090', 'imgr10_' + name + '_seed1993')
            self.assertEqual({k for k in first if first[k] != other[k]}, {'dual_mask_protect_position'})
            self.assertEqual(other['dual_mask_position_norm_match'], 'prior_layer_min')

    def test_missing_cells_never_form_a_complete_matrix(self):
        self.runner()
        analyzer = importlib.import_module('analyze_final_f_validation')
        cells = {name: {key: value for key in analyzer.METRICS}
                 for name, value in [('C11', 10.), ('C01', 7.), ('C10', 6.), ('C00', 2.)]}
        full = analyzer.factor_effects(cells)
        self.assertEqual(full['Average'], dict(permission_effect=3.5, gate_effect=4.5, interaction=-1.))
        self.assertEqual(analyzer.factor_effects({k: v for k, v in cells.items() if k != 'C00'}), {})

    def test_update_and_delivered_gate_evidence_must_match_configuration(self):
        self.runner()
        analyzer = importlib.import_module('analyze_final_f_validation')
        self.assertTrue(hasattr(analyzer, 'update_evidence_issues'), '生效门证据检查尚未实现')
        config = dict(dual_mask_protect_position='wpre', dual_mask_permission_mode='asymmetric',
                      dual_mask_position_norm_match='off', dual_mask_s_conflict_enabled=True,
                      dual_mask_p_conflict_enabled=True, embd_dim=6, dual_mask_conflict_ratio=.1)
        updates = [dict(task=t, layer=l, branch=b, projection=p, position='wpre',
                        permission_mode='asymmetric', position_norm_match='off', score_mode='magnitude',
                        merge_error=0., merge_error_definition='same_factor_cached_gate_arithmetic')
                   for t in range(10) for l in range(12)
                   for b in (('S',) if t == 0 else ('S', 'P')) for p in ('Q', 'K', 'V')]
        budgets = [dict(task=t, layer=l, branch=b, gate_enabled=True, reference_k=10, applied_k=10,
                        granularity='layer', removed_norm_definition='conflict_only',
                        merge_error=0., merge_error_definition='same_factor_cached_gate_arithmetic')
                   for t in range(1, 10) for l in range(12) for b in ('S', 'P')]
        self.assertEqual(analyzer.update_evidence_issues(updates, budgets, config), [])
        self.assertIn('merge_update_records', analyzer.update_evidence_issues(updates[:-1], budgets, config))
        wrong = copy.deepcopy(updates)
        wrong[-1]['permission_mode'] = 'symmetric_soft'
        self.assertIn('merge_update_policy', analyzer.update_evidence_issues(wrong, budgets, config))
        wrong = copy.deepcopy(budgets)
        wrong[-1]['gate_enabled'] = False
        self.assertIn('delivered_gate_policy', analyzer.update_evidence_issues(updates, wrong, config))
        wrong[-1].update(gate_enabled=True, reference_k=0, applied_k=0)
        self.assertIn('delivered_gate_policy', analyzer.update_evidence_issues(updates, wrong, config))
        epochs = [dict(task=1, epoch=1, layer=0, branch='S', selected_coordinates=n)
                  for n in (3, 3, 4)]
        self.assertEqual(analyzer.epoch_selection_issues(epochs, config), [])
        epochs[-1]['selected_coordinates'] = 3
        self.assertEqual(analyzer.epoch_selection_issues(epochs, config), ['epoch_selected_count'])

    def test_pair_csv_is_cleared_for_unreadable_partial_runs(self):
        self.runner()
        analyzer = importlib.import_module('analyze_final_f_validation')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'manifest.json').write_text(json.dumps(dict(machine='3090', revision='r')))
            (directory / 'queue.json').write_text(json.dumps([
                dict(mode='imgr10_C11_seed1993', status='failed', exit_code=1),
                dict(mode='imgr10_C01_seed1993', status='pending')]))
            (directory / 'pairs.csv').write_text('stale pair')
            analyzer.summarize_saved(directory)
            self.assertEqual((directory / 'pairs.csv').read_text(), '')
            self.assertEqual(json.loads((directory / 'pairs.json').read_text()), [])

    def test_analysis_failure_does_not_stop_remaining_training(self):
        runner = self.runner()
        analyzer = importlib.import_module('analyze_final_f_validation')
        calls = []
        def train(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run', side_effect=train), \
                patch.object(analyzer, 'summarize_saved', side_effect=RuntimeError('analysis broke')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute('3090', runner.modes('3090'), Path(temp), 'r'), 0)
            self.assertTrue((Path(temp) / 'analysis_errors.jsonl').exists())
        self.assertEqual(len(calls), 12)
        self.assertEqual([n for n, smoke in calls if not smoke], runner.modes('3090'))

    def test_smoke_failure_and_deadline_do_not_launch_formal_runs(self):
        runner = self.runner()
        for expired in (False, True):
            calls = []
            def train(machine, name, directory, revision, smoke=False, dry_run=False):
                calls.append((name, smoke))
                return dict(mode=name, status='failed', exit_code=7, minutes=1)
            with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                    patch.object(runner.budget.tail.engine, 'run', side_effect=train), \
                    patch.object(runner.budget.tail, 'summarize_safely'), \
                    contextlib.redirect_stdout(io.StringIO()):
                directory = Path(temp)
                if expired:
                    (directory / 'budget.json').write_text(json.dumps(dict(deadline_unix=0)))
                code = runner.execute('3090', runner.modes('3090'), directory, 'r')
                self.assertEqual(code, 0 if expired else 7)
            self.assertEqual(calls, [] if expired else [('imgr10_C11_seed1993', True)])


class PriorLayerNormTests(unittest.TestCase):
    def setUp(self):
        state = torch.get_rng_state().clone()
        self.addCleanup(torch.set_rng_state, state)
        torch.manual_seed(1993)

    def match(self, raw, *gates):
        self.assertTrue(hasattr(dualmask_core, 'prior_layer_min_gates'),
                        '三位置共同最小范数尚未实现')
        return dualmask_core.prior_layer_min_gates(raw, *gates)

    def module(self, position='wpre', task=1):
        module = position_fixtures.ProtectPositionTests().make(position, task=task, score='magnitude')
        module.args.update(dual_mask_position_norm_match='prior_layer_min',
                           dual_mask_fixed_protect_strength=.5)
        module.before_task(task)
        return module

    def test_three_candidates_have_one_whole_layer_target(self):
        raw = torch.tensor([[3., -4.], [1., 2.], [5., -6.]], requires_grad=True)
        gates = [torch.ones_like(raw), torch.full_like(raw, .5),
                 torch.tensor([[1., 0.], [0., 1.], [1., 0.]])]
        matched, norms, target, scales = self.match(raw, *gates)
        expected_target = min(float((raw.detach() * gate).norm()) for gate in gates)
        self.assertAlmostEqual(float(target), expected_target, places=6)
        self.assertTrue(torch.all(scales <= 1))
        self.assertFalse(scales.requires_grad)
        for gate in matched:
            self.assertAlmostEqual(float((raw * gate).detach().norm()), expected_target, places=5)
        (raw * matched[0]).sum().backward()
        torch.testing.assert_close(raw.grad, matched[0])

    def test_zero_update_preserves_all_initial_gates(self):
        raw = torch.zeros(3, 2, requires_grad=True)
        gates = [torch.full_like(raw, x) for x in (1., .5, .25)]
        matched, _, _, scales = self.match(raw, *gates)
        torch.testing.assert_close(scales, torch.ones(3), rtol=0, atol=0)
        for first, second in zip(gates, matched):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_position_counts_rng_p_complement_and_matching(self):
        self.match(torch.ones(3, 2), *(torch.ones(3, 2) for _ in range(3)))
        module = self.module()
        masks = (module.core_reference_protect, module.core_magnitude_protect, module.core_permuted_protect)
        raw = torch.arange(1., 49.).reshape(12, 4)
        state = torch.get_rng_state().clone()
        for isolated in (False, True):
            values = []
            for position, mask in zip(('wpre', 'wpre_magnitude', 'permuted'), masks):
                candidate = copy.deepcopy(module)
                candidate.args['dual_mask_protect_position'] = position
                candidate.general_mask.copy_(mask)
                candidate.isolated_mask.copy_(1 - mask)
                safe = candidate._safe_delta(raw, isolated)
                values.append(safe.norm())
                self.assertTrue(torch.all(safe.abs() <= raw.abs()))
                if isolated:
                    self.assertEqual(float((safe * mask).abs().max()), 0.)
                for first, second in zip(mask.chunk(3), masks[0].chunk(3)):
                    self.assertEqual(int(first.sum()), int(second.sum()))
            torch.testing.assert_close(torch.stack(values), values[0].expand(3), rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_zero_b_gradient_and_forward_merge_once(self):
        self.match(torch.ones(3, 2), *(torch.ones(3, 2) for _ in range(3)))
        for position in ('wpre', 'wpre_magnitude', 'permuted'):
            module = self.module(position)
            x = torch.arange(1., 13.).reshape(3, 4)
            module._contrib_from_units(x, 1).sum().backward()
            for unit in (module.S_lora[1], module.P_lora[1]):
                self.assertGreater(float(unit.B_weight.grad.norm()), 0.)
                with torch.no_grad():
                    unit.B_weight.normal_(std=.01)
            expected = module._contrib_from_units(x, 1)
            original = module.qkv.weight.detach().clone()
            module.after_task(1)
            torch.testing.assert_close(torch.nn.functional.linear(x, module.qkv.weight - original),
                                       expected, rtol=1e-4, atol=1e-6)
            once = module.qkv.weight.detach().clone()
            module.after_task(1)
            torch.testing.assert_close(module.qkv.weight, once, rtol=0, atol=0)

    def test_task0_is_unchanged(self):
        candidate = self.module(task=0)
        reference = copy.deepcopy(candidate)
        reference.args['dual_mask_position_norm_match'] = 'off'
        x = torch.arange(1., 13.).reshape(3, 4)
        torch.testing.assert_close(candidate._contrib_from_units(x, 0), reference._contrib_from_units(x, 0), rtol=0, atol=0)

    def test_norm_control_is_not_reported_as_conflict_only(self):
        raw = torch.ones(3, 2)
        row = update_rows(raw, raw, raw * .25, torch.zeros_like(raw), torch.zeros_like(raw),
                          branch='S', position_norm_match='prior_layer_min')[0]
        self.assertIsNone(row['conflict_removed_norm'])
        self.assertGreater(row['post_permission_removed_norm'], 0)

    def test_read_only_diagnostic_keeps_next_optimizer_step(self):
        from types import SimpleNamespace
        from utils.dualmask_core_audit import position_diagnostic
        first = audit_fixtures.TinyNetwork()
        first.attention = self.module('wpre_magnitude')
        with torch.no_grad():
            first.attention.S_lora[1].B_weight.normal_(std=.03)
            first.attention.P_lora[1].B_weight.normal_(std=.03)
        second = copy.deepcopy(first)
        learner = SimpleNamespace(_network=first, _device=torch.device('cpu'),
            args={'dual_mask_position_norm_match': 'prior_layer_min'}, run_epoch=20,
            _cur_task=1, _known_classes=2, _total_classes=4, batch_size=3,
            _iter_lora_modules=lambda: [first.attention])
        state = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO'):
            position_diagnostic(learner, audit_fixtures.Images(), 20)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for model in (first, second):
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            optimizer.zero_grad()
            model.interface(torch.arange(12).reshape(3, 4).float()).square().mean().backward()
            optimizer.step()
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)

    def test_symmetric_permissions_report_actual_p_merge_and_removal(self):
        import re
        for alpha, enabled in ((0., True), (0., False), (.5, True)):
            module = self.module()
            module.args.update(dual_mask_permission_mode='symmetric_soft', dual_mask_position_norm_match='off')
            module.effective_protect_strength = alpha
            module.dual_mask_s_conflict_enabled = enabled
            module.dual_mask_p_conflict_enabled = enabled
            with torch.no_grad():
                module.P_lora[1].B_weight.normal_(std=.02)
            factor = module.P_lora[1].B_weight @ module.P_lora[1].A_weight
            safe = module.plora_gamma * module._safe_delta(factor, True)
            base, _ = module._merge_base_and_conflict(factor, True, .1, compute_conflict=False)
            wanted = float((module.plora_gamma * base - safe).detach().norm())
            with self.assertLogs(level='INFO') as logs:
                module.after_task(1)
            line = next(line for line in logs.output if 'P applied merge diagnostic:' in line)
            self.assertLess(float(re.search(r'merge_error=([\d.e+-]+)', line)[1]), 1e-6)
            self.assertAlmostEqual(float(re.search(r'removed_norm=([\d.e+-]+)', line)[1]), wanted, places=5)

    def test_default_f_forward_gradient_step_and_merge_match_base_commit(self):
        source = subprocess.check_output(['git', 'show',
            'd5cb5bc0c36f9783c688de54a0974e70aa40f977:models/attention.py'], cwd=ROOT, text=True)
        namespace = {'__name__': 'base_attention'}
        exec(compile(source, '<base_attention>', 'exec'), namespace)
        current = self.module()
        current.args['dual_mask_position_norm_match'] = 'off'
        with torch.no_grad():
            current.S_lora[1].B_weight.normal_(std=.02)
            current.P_lora[1].B_weight.normal_(std=.02)
        previous = copy.deepcopy(current)
        previous.__class__ = namespace[current.__class__.__name__]
        x = torch.arange(12).reshape(3, 4).float()
        values = [m._contrib_from_units(x, 1) for m in (current, previous)]
        torch.testing.assert_close(*values, rtol=0, atol=0)
        for module, value in zip((current, previous), values):
            optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.02)
            optimizer.zero_grad()
            value.square().mean().backward()
            optimizer.step()
        for a, b in zip(current.parameters(), previous.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        current.after_task(1)
        previous.after_task(1)
        torch.testing.assert_close(current.qkv.weight, previous.qkv.weight, rtol=0, atol=0)
