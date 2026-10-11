"""预算、均匀对照与整层位置控制的数学和真实合并回归。"""
import copy
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test import test_protection_search as protection_fixtures
from test import test_dualmask_core_audit as audit_fixtures
from utils import dualmask_core
from utils.dualmask_core_audit import budget_update_values, position_diagnostic
from utils.protection_strength import protection_update

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_budget_verification as runner
import analyze_budget_verification as analyzer


class BudgetUpdateTests(unittest.TestCase):
    def module(self, rule='energy_equal', paired=False):
        with patch.object(protection_fixtures.runner, 'SPEC', protection_fixtures.runner.SPEC):
            module = protection_fixtures.ProtectionSearchTests().module('E1')
        module.args.update(dual_mask_protection_rule=rule,
                           dual_mask_position_norm_match='paired_layer_min' if paired else 'off')
        module.core_reference_protect = module.general_mask.clone()
        module.core_permuted_protect = torch.roll(module.general_mask, 1, 0)
        return module

    def test_uniform_matches_reference_removal_without_claiming_region_budget(self):
        for rule in ('energy_equal', 'energy_half'):
            module = self.module(rule)
            raw = torch.ones_like(module.qkv.weight, requires_grad=True)
            with torch.no_grad():
                raw[module.general_mask.bool()] = 9.
            reference = protection_update(module, raw)
            module.args['dual_mask_protection_rule'] = rule + '_uniform'
            actual = protection_update(module, raw)
            _, selected = module._branch_conflict(raw, False)
            value = raw * (1 - module._conflict_parameters()[1] * selected)
            torch.testing.assert_close((value - actual['safe']).norm(),
                                       (value - reference['safe']).norm(), rtol=1e-6, atol=1e-6)
            self.assertFalse(actual['uniform_coefficient'].requires_grad)
            torch.testing.assert_close(actual['alpha'], reference['alpha'])
            self.assertTrue(torch.all(actual['safe'].abs() <= raw.abs()))
            actual['safe'].sum().backward()
            torch.testing.assert_close(raw.grad, actual['gate'], rtol=0, atol=0)

    def test_uniform_zero_update_preserves_reference_first_gradient(self):
        for rule in ('energy_equal', 'energy_half'):
            module = self.module(rule)
            raw = torch.zeros_like(module.qkv.weight, requires_grad=True)
            reference = protection_update(module, raw)
            module.args['dual_mask_protection_rule'] = rule + '_uniform'
            actual = protection_update(module, raw)
            torch.testing.assert_close(actual['gate'], reference['gate'], rtol=0, atol=0)
            actual['safe'].sum().backward()
            torch.testing.assert_close(raw.grad, reference['gate'], rtol=0, atol=0)

    def test_layer_match_equalizes_whole_qkv_and_preserves_each_budget(self):
        for rule, ratio in (('energy_equal', 1.), ('energy_half', .5)):
            module = self.module(rule, paired=True)
            raw = torch.arange(1., module.qkv.weight.numel() + 1).reshape_as(module.qkv.weight)
            raw[::2] *= 7
            masks = (module.core_reference_protect, module.core_permuted_protect)
            gates = [protection_update(module, raw, protect=mask)['gate'] for mask in masks]
            matched, _, _, scales = dualmask_core.paired_layer_min_gates(raw, *gates)
            self.assertTrue(torch.all(scales <= 1))
            torch.testing.assert_close((raw * matched[0]).norm(), (raw * matched[1]).norm(), rtol=1e-6, atol=1e-6)
            for mask, gate in zip(masks, matched):
                safe = raw * gate
                self.assertLessEqual(float((safe * mask).norm()),
                                     ratio * float((safe * (1 - mask)).norm()) + 1e-4)
            zeros = torch.zeros_like(raw)
            zero_gates, _, _, scales = dualmask_core.paired_layer_min_gates(zeros, *gates)
            torch.testing.assert_close(scales, torch.ones_like(scales), rtol=0, atol=0)
            for original, zero in zip(gates, zero_gates):
                torch.testing.assert_close(original, zero, rtol=0, atol=0)

    def test_new_paths_forward_merge_once_and_p_permissions(self):
        for rule, paired in (('energy_equal_uniform', False), ('energy_half_uniform', False),
                             ('energy_equal', True), ('energy_half', True)):
            module = self.module(rule, paired)
            x = torch.randn(4, 4)
            module._contrib_from_units(x, 1).sum().backward()
            for unit in (module.S_lora[1], module.P_lora[1]):
                self.assertGreater(float(unit.B_weight.grad.norm()), 0.)
            with torch.no_grad():
                module.S_lora[1].B_weight.normal_(std=.02)
                module.P_lora[1].B_weight.normal_(std=.02)
            p = module.P_lora[1].B_weight @ module.P_lora[1].A_weight
            self.assertEqual(float((module._safe_delta(p, True) * module.general_mask).detach().abs().max()), 0.)
            expected = module._contrib_from_units(x, 1)
            original = module.qkv.weight.detach().clone()
            module.after_task(1)
            torch.testing.assert_close(torch.nn.functional.linear(x, module.qkv.weight - original),
                                       expected, rtol=1e-4, atol=1e-7)
            once = module.qkv.weight.detach().clone()
            module.after_task(1)
            torch.testing.assert_close(module.qkv.weight, once, rtol=0, atol=0)

    def test_existing_protection_rules_match_previous_commit(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(['git', 'show',
            '3f73843c4aa198c1216feda465dd92aa00f90d1c:utils/protection_strength.py'], cwd=root, text=True)
        namespace = {}
        exec(compile(source, '<previous-protection>', 'exec'), namespace)
        for rule in ('static', 'energy_equal', 'energy_half'):
            module = self.module(rule)
            raw = torch.randn_like(module.qkv.weight)
            first, second = (raw.clone().requires_grad_() for _ in range(2))
            old, new = namespace['protection_update'](module, first), protection_update(module, second)
            torch.testing.assert_close(old['safe'], new['safe'], rtol=0, atol=0)
            old['safe'].square().mean().backward()
            new['safe'].square().mean().backward()
            torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)

    def test_task0_and_p_updates_are_unchanged(self):
        for rule in ('energy_equal_uniform', 'energy_half_uniform', 'energy_equal', 'energy_half'):
            module = self.module(rule, paired=not rule.endswith('_uniform'))
            module.cur_task = 0
            raw = torch.randn_like(module.qkv.weight, requires_grad=True)
            torch.testing.assert_close(module._safe_delta(raw, False), raw, rtol=0, atol=0)
            module._safe_delta(raw, False).sum().backward()
            torch.testing.assert_close(raw.grad, torch.ones_like(raw), rtol=0, atol=0)
        reference = self.module('energy_equal')
        actual = copy.deepcopy(reference)
        actual.args['dual_mask_protection_rule'] = 'energy_equal_uniform'
        raw = torch.randn_like(reference.qkv.weight)
        torch.testing.assert_close(actual._safe_delta(raw, True), reference._safe_delta(raw, True), rtol=0, atol=0)

    def test_audit_uses_dynamic_budget_and_matches_actual_effective_update(self):
        for rule, paired in (('energy_equal', True), ('energy_half', True),
                             ('energy_equal_uniform', False), ('energy_half_uniform', False)):
            module = self.module(rule, paired)
            raw = torch.ones_like(module.qkv.weight)
            raw[module.general_mask.bool()] = 9
            effective, _, selected = module._safe_delta(raw, False, return_details=True)
            row = budget_update_values(module, raw, effective, selected, False, .5)
            self.assertNotEqual(row['reference_alpha'], module.effective_protect_strength)
            self.assertEqual(row['region_budget_applies'], paired)
            self.assertLess(row['uniform_match_residual'] or 0., 1e-5)
            self.assertLess(row.get('pair_matching_residual', 0.), 1e-5)
            if paired:
                self.assertLess(row['budget_violation'], 1e-5)
                state = protection_update(module, raw)
                base, _ = module._merge_base_and_conflict(raw, False, .1, compute_conflict=False)
                torch.testing.assert_close(base, state['base'], rtol=0, atol=0)

    def test_read_only_budget_position_probe_keeps_next_training_step(self):
        first = audit_fixtures.TinyNetwork()
        first.attention = self.module('energy_half', paired=True)
        first.attention.args['dual_mask_budget_audit'] = True
        with torch.no_grad():
            first.attention.S_lora[1].B_weight.normal_()
            first.attention.P_lora[1].B_weight.normal_()
        second = copy.deepcopy(first)
        learner = SimpleNamespace(_network=first, _device=torch.device('cpu'),
            args={'dual_mask_budget_audit': True}, run_epoch=20, _cur_task=1,
            _known_classes=2, _total_classes=4, batch_size=3,
            _iter_lora_modules=lambda: [first.attention])
        before = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO') as logs:
            position_diagnostic(learner, audit_fixtures.Images(), 20)
        self.assertTrue(any('same_state_paired_layer_min' in line for line in logs.output))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for model in (first, second):
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            optimizer.zero_grad()
            model.interface(torch.arange(12).reshape(3, 4).float()).square().mean().backward()
            optimizer.step()
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)

    def test_paired_p_merge_diagnostic_reconstructs_delivered_gate(self):
        module = self.module('energy_half', paired=True)
        module.args['dual_mask_protect_position'] = 'permuted'
        module.general_mask = module.core_permuted_protect.clone()
        factor = torch.ones_like(module.qkv.weight)
        factor[module.core_reference_protect.bool()] = 9.
        safe, gate, applied = module._safe_delta(factor, True, return_details=True)
        reference = .75 * (factor * gate)
        row = module._private_merge_diagnostic(.75 * factor, .75 * safe, .1,
            module._conflict_parameters()[1], applied_mask=applied, merge_reference=reference)
        self.assertLess(row['merge_error'], 1e-7)
        self.assertEqual(row['removed_definition'], 'conflict_and_layer_norm')

    def test_budget_audit_resets_peak_before_adapter_setup(self):
        from methods.dlora import Learner
        network = SimpleNamespace(numtask=2, to=lambda device: None, named_parameters=lambda: [])
        def stop():
            raise RuntimeError('stop before GPU setup')
        learner = SimpleNamespace(_network=network, args={'dual_mask_budget_audit': True},
                                  _device=torch.device('cuda'), _iter_lora_modules=stop)
        with patch('methods.dlora.stage_cost', return_value=contextlib.nullcontext()), \
                patch.object(torch.cuda, 'reset_peak_memory_stats') as reset, \
                self.assertRaisesRegex(RuntimeError, 'stop before GPU setup'):
            Learner._train(learner, None, None)
        reset.assert_called_once_with(learner._device)


class BudgetQueueTests(unittest.TestCase):
    def test_fixed_order_and_dataset_specific_training_recipe(self):
        for machine, datasets, count in (('3090', ['imgr10'], 9), ('5090', ['imga10', 'cub10'], 12)):
            runner.validate_settings(machine)
            self.assertEqual(len(runner.modes(machine)), count)
            self.assertEqual([runner.identity(machine, n)[0]['name'] for n in runner.modes(machine)],
                             [d for d in datasets for _ in range(9 if machine == '3090' else 6)])
            for name in runner.modes(machine):
                dataset, variant = runner.identity(machine, name)
                source = json.loads((ROOT / dataset['config']).read_text())
                actual = runner.effective_config(machine, name)
                for key in ('init_lr', 'lrate', 'optim', 'margin', 'slora_gamma', 'plora_gamma',
                            'data_path', 'init_cls', 'increment', 'rank'):
                    self.assertEqual(actual[key], source[key])
                self.assertEqual(actual['dual_mask_private_rank'], 0 if variant['recipe'] == 'O' else dataset['rank'])
                self.assertFalse(actual['dual_mask_prototype_position_probe'])
                self.assertFalse(actual['save_task_weights'])
                smoke = runner.effective_config(machine, name, True)
                self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2, 1, 1))
        first = runner.settings_for('3090', 'imgr10_R3_seed1993')
        second = runner.settings_for('3090', 'imgr10_R4_seed1993')
        self.assertEqual({k for k in first if first[k] != second[k]}, {'dual_mask_protect_position'})

    def simulate(self, directory, machine='5090', failure=None):
        calls = []
        def run(m, name, path, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            failed = failure is not None and smoke == failure
            return dict(mode=name, status='failed' if failed else 'completed',
                        exit_code=7 if failed else 0, minutes=1)
        with patch.object(runner.tail.engine, 'run', side_effect=run), \
                patch.object(runner, 'summarize_safely'), contextlib.redirect_stdout(io.StringIO()):
            result = runner.execute(machine, runner.modes(machine), directory, 'r')
        return result, calls

    def test_all_dataset_smokes_precede_formal_and_completed_runs_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            result, calls = self.simulate(directory)
            self.assertEqual(result, 0)
            names = runner.modes('5090')
            self.assertEqual(calls, [(n, True) for n in names] + [(n, False) for n in names])
            self.assertEqual(self.simulate(directory)[1], [])

    def test_shared_deadline_covers_both_datasets_and_survives_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(runner.tail.time, 'monotonic', side_effect=[0] + [0] * 18 + [36001]):
                result, calls = self.simulate(directory)
            self.assertEqual(result, 0)
            self.assertEqual(sum(not smoke for _, smoke in calls), 6)
            records = json.loads((directory / 'queue.json').read_text())
            self.assertTrue(all(row['status'] == 'time_budget_pending' for row in records[6:]))
            (directory / 'budget.json').write_text(json.dumps({'deadline_unix': 100}))
            with patch.object(runner.tail.time, 'time', return_value=200):
                self.assertEqual(self.simulate(directory)[1], [])

    def test_failures_pause_but_analysis_errors_do_not_stop_planned_training(self):
        for smoke_failure in (True, False):
            with tempfile.TemporaryDirectory() as temp:
                result, calls = self.simulate(Path(temp), failure=smoke_failure)
                self.assertEqual(result, 7)
                self.assertEqual(len(calls), 1 if smoke_failure else 13)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(analyzer, 'summarize_saved', side_effect=RuntimeError('analysis error')), \
                patch.object(runner.tail.engine, 'run', side_effect=lambda m, n, p, r, smoke=False, dry_run=False:
                    dict(mode=n, status='completed', exit_code=0, minutes=1)), \
                contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temp)
            self.assertEqual(runner.execute('3090', runner.modes('3090'), directory, 'r'), 0)
            self.assertTrue((directory / 'analysis_errors.jsonl').exists())
            self.assertEqual(len(json.loads((directory / 'queue.json').read_text())), 9)

    def test_resume_requires_healthy_matching_source_config_and_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            names = runner.modes('3090')
            (directory / 'manifest.json').write_text(json.dumps(dict(machine='3090', revision='r', order=names)))
            with self.assertRaises(ValueError):
                runner.check_resume(directory, '3090', 'other', names)
            record = dict(mode=names[0], status='completed', exit_code=0)
            (directory / 'queue.json').write_text(json.dumps([record]))
            snapshot = dict(machine='3090', code_revision='r', phase='formal',
                effective_config=runner.effective_config('3090', names[0]), source_sha256={})
            measured = dict(runtime_error=False, tasks_reported=10, **{k: 1 for k in runner.METRICS})
            with patch.object(runner, 'read_run', return_value=(measured, snapshot, {})):
                runner.check_resume(directory, '3090', 'r', names)
                snapshot['source_sha256'] = {'models/attention.py': 'bad'}
                with self.assertRaises(ValueError):
                    runner.check_resume(directory, '3090', 'r', names)
                snapshot['source_sha256'] = {}
                snapshot['effective_config']['dual_mask_fixed_coverage'] = .825
                with self.assertRaises(ValueError):
                    runner.check_resume(directory, '3090', 'r', names)
            record['status'] = 'failed'
            (directory / 'queue.json').write_text(json.dumps([record]))
            with self.assertRaises(ValueError):
                runner.check_resume(directory, '3090', 'r', names)

    def test_missing_or_mismatched_pairs_never_get_attribution(self):
        rows = [dict(mode=n, dataset='imgr10', valid_performance=True, **{k: 1 for k in runner.METRICS})
                for n in runner.modes('3090')[:2]]
        pairs = analyzer.pairs_for(rows, [], '3090')
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['first'], rows[1]['mode'])
        self.assertEqual(analyzer.pairs_for(rows, [{'dataset': 'imgr10', 'field': 'source'}], '3090'), [])

    def test_summary_keeps_metrics_but_blocks_attribution_when_budget_evidence_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            name = runner.modes('3090')[0]
            record = dict(mode=name, status='completed', exit_code=0)
            (directory / 'manifest.json').write_text(json.dumps(dict(machine='3090', revision='r')))
            (directory / 'queue.json').write_text(json.dumps([record]))
            path = directory / name
            path.mkdir()
            epochs = ['LoRA learning rates: task=' + str(t) + ', epoch=' + str(e)
                      for t in range(10) for e in range(1, 21)]
            budgets = [dict(task=t, epoch=e, layer=l, branch=b, region_budget_applies=False,
                            uniform_match_residual=None, pair_matching_residual=None)
                       for t in range(1, 10) for e in (1, 5, 10, 20)
                       for l in range(12) for b in ('S', 'P')]
            telemetry = {key: [] for key in ('tasks', 'epochs', 'updates', 'masks', 'costs', 'storage')}
            telemetry['diagnostics'] = [dict(task=t, epoch=20, partition=p, scope='all_seen',
                norm_control='same_state_paired_layer_min', count=10) for t in (1, 5, 9) for p in ('old', 'new')]
            telemetry['epochs'] = [dict(task=t, epoch=e, layer=l, branch=b, projection=p)
                for t in range(10) for e in (1, 5, 10, 20) for l in range(12)
                for b in (('S',) if t == 0 else ('S', 'P')) for p in ('Q', 'K', 'V')]
            snapshot = dict(code_revision='r', machine='3090', phase='formal',
                effective_config=runner.effective_config('3090', name), source_sha256={}, software={}, hardware={})
            def read(*args):
                return dict(record, valid_performance=True, Task0=1, **{k: 1 for k in runner.METRICS}), snapshot, telemetry
            with patch.object(analyzer, 'read_run', side_effect=read), patch.object(analyzer, 'draw'):
                for complete in (True, False):
                    content = epochs + ['BudgetUpdateAudit ' + json.dumps(r) for r in (budgets if complete else budgets[:-1])]
                    (path / 'training.log').write_text('\n'.join(content))
                    analyzer.summarize_saved(directory)
                    issues = json.loads((directory / 'matching_issues.json').read_text())
                    self.assertEqual(bool(issues), not complete)
                    result = json.loads((directory / 'results.json').read_text())[0]
                    self.assertTrue(result['valid_performance'])
                    self.assertIsNone(result['std'])
                    self.assertFalse(result['complete_three_seeds'])
                    self.assertEqual(result['budget_records_complete'], complete)


if __name__ == '__main__':
    unittest.main()
