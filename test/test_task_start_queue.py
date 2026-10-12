"""七组固定配方、GPU短测先行及部分结果归因的运行入口检查。"""
import json
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class TaskStartQueueTests(unittest.TestCase):
    def test_dry_run_executes_every_smoke_before_formal_without_writing(self):
        for machine, count in (('3090', 3), ('5090', 4)):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / 'no-write'
                result = subprocess.run([sys.executable, 'scripts/run_task_start_prior.py',
                    '--machine', machine, '--mode', 'dry-run', '--resume', str(directory)],
                    cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.count('GPU SMOKE, NOT PERFORMANCE'), count)
                self.assertEqual(result.stdout.count('REAL FULL T10'), count)
                self.assertLess(result.stdout.rfind('GPU SMOKE, NOT PERFORMANCE'),
                                result.stdout.find('REAL FULL T10'))
                self.assertFalse(directory.exists())

    def test_only_declared_factors_differ_and_protection_rule_accepts_products(self):
        import run_task_start_prior as runner
        for machine, factors in (('3090', {'dual_mask_protect_position'}),
                ('5090', {'dual_mask_conflict_score_mode', 'dual_mask_conflict_importance_source'})):
            first = runner.settings_for(machine, runner.modes(machine)[0])
            runner.validate_settings(machine)
            for name in runner.modes(machine):
                actual = runner.settings_for(machine, name)
                self.assertLessEqual({k for k in first.keys() | actual.keys()
                    if first.get(k) != actual.get(k)}, factors)
                self.assertEqual(actual['dual_mask_protection_rule'], 'energy_equal')
                self.assertEqual(actual['dual_mask_private_rank'], 64)
                self.assertTrue(actual['dual_mask_task_start_probe'])

    def test_invalid_budget_score_combinations_remain_rejected(self):
        import run_task_start_prior as runner
        from dualmask_config import normalize_dualmask_config
        base = runner.effective_config('5090', 'imgr10_S2_seed1993')
        for changes in ({'dual_mask_protection_rule': 'energy_half'},
                        {'dual_mask_task_start_probe': False},
                        {'dual_mask_conflict_exact_topk': False},
                        {'dual_mask_reg_weight': .01},
                        {'dual_mask_conflict_importance_source': 'invalid'}):
            with self.assertRaises(ValueError):
                normalize_dualmask_config(dict(base, **changes))

    def test_pairs_need_complete_healthy_matching_members(self):
        import analyze_task_start_prior as analyzer
        from analyze_core_evidence import METRICS
        rows = [dict(mode='imgr10_' + name + '_seed1993', valid_performance=True,
                     dataset='imgr10', **{k: 1. for k in METRICS}) for name in ('R0', 'R1', 'R2')]
        self.assertEqual(len(analyzer.pairs_for(rows, [], '3090')), 2)
        self.assertEqual(analyzer.pairs_for(rows, [{'dataset': 'imgr10'}], '3090'), [])
        rows[1]['valid_performance'] = False
        self.assertEqual(analyzer.pairs_for(rows, [], '3090'), [])

    def test_analysis_error_restores_runner_and_does_not_escape(self):
        import run_task_start_prior as runner
        import run_budget_verification as budget
        import analyze_task_start_prior as analyzer
        original = budget.SPEC, budget.settings_for
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(analyzer, 'summarize_saved', side_effect=RuntimeError('analysis-only')):
                runner.summarize_safely(Path(temporary), '3090', [])
            errors = (Path(temporary) / 'analysis_errors.jsonl').read_text()
            self.assertIn('analysis-only', errors)
        with runner.configured_runner():
            self.assertEqual(budget.SPEC, runner.SPEC)
        self.assertEqual(original, (budget.SPEC, budget.settings_for))

    def test_deadline_completed_skip_and_smoke_failure_with_new_runner(self):
        import run_task_start_prior as runner
        import run_budget_verification as budget
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temporary, runner.configured_runner(), \
                patch.object(budget.tail.engine, 'run', side_effect=run), \
                patch.object(budget, 'summarize_safely'), contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            names = runner.modes('5090')
            with patch.object(budget.tail.time, 'monotonic', side_effect=[0] * 6 + [28801]):
                self.assertEqual(runner.execute('5090', names, directory, 'r'), 0)
            self.assertEqual(calls, [(n, True) for n in names] + [(names[0], False)])
            rows = json.loads((directory / 'queue.json').read_text())
            self.assertTrue(all(r['status'] == 'time_budget_pending' for r in rows[1:]))
            calls.clear()
            runner.execute('5090', names, directory, 'r')
            self.assertEqual(calls, [(n, False) for n in names[1:]])
            calls.clear()
            runner.execute('5090', names, directory, 'r')
            self.assertEqual(calls, [])
        with tempfile.TemporaryDirectory() as temporary, runner.configured_runner(), \
                patch.object(budget.tail.engine, 'run', return_value=dict(mode='imgr10_R0_seed1993',
                    status='failed', exit_code=7, minutes=1)) as run, \
                patch.object(budget, 'summarize_safely'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute('3090', runner.modes('3090'), Path(temporary), 'r'), 7)
            self.assertEqual(run.call_count, 1)

    def test_missing_task_start_records_block_attribution_but_keep_measured_scores(self):
        import run_task_start_prior as runner
        import analyze_task_start_prior as analyzer
        import analyze_budget_verification as budget_analysis
        from analyze_core_evidence import METRICS
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            results = [dict(mode=n, status='completed', dataset='imgr10', valid_performance=True,
                            **{k: 1. for k in METRICS}) for n in runner.modes('3090')]
            (directory / 'manifest.json').write_text(json.dumps(dict(machine='3090')))
            for row in results:
                folder = directory / row['mode']
                folder.mkdir()
                (folder / 'training.log').write_text('')
            def base_analysis(directory):
                for name, value in (('results', results), ('matching_issues', [])):
                    (directory / (name + '.json')).write_text(json.dumps(value))
            with patch.object(budget_analysis, 'summarize_saved', side_effect=base_analysis):
                analyzer.summarize_saved(directory)
            self.assertEqual(json.loads((directory / 'pairs.json').read_text()), [])
            measured = json.loads((directory / 'results.json').read_text())
            self.assertTrue(all(r['Average'] == 1. and not r['task_start_evidence_complete'] for r in measured))

    def test_resume_rejects_missing_fingerprint_in_completed_reference(self):
        import run_task_start_prior as runner
        import run_budget_verification as budget
        from analyze_core_evidence import METRICS
        with tempfile.TemporaryDirectory() as temporary, runner.configured_runner():
            directory = Path(temporary)
            name = runner.modes('3090')[0]
            (directory / 'manifest.json').write_text(json.dumps(dict(
                machine='3090', revision='r', order=[name])))
            (directory / 'queue.json').write_text(json.dumps([dict(
                mode=name, status='completed', exit_code=0)]))
            (directory / name).mkdir()
            snapshot = dict(machine='3090', phase='formal', code_revision='r',
                effective_config=runner.effective_config('3090', name), source_sha256={})
            (directory / name / 'run.json').write_text(json.dumps(snapshot))
            measured = dict(runtime_error=False, tasks_reported=10, **{k: 1. for k in METRICS})
            with patch.object(budget, 'read_run', return_value=(measured, snapshot, {})):
                with self.assertRaises(ValueError):
                    budget.check_resume(directory, '3090', 'r', [name])

    def test_early_analysis_failure_never_publishes_unvalidated_pairs(self):
        import run_task_start_prior as runner
        import analyze_budget_verification as base
        from analyze_core_evidence import METRICS
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            results = [dict(mode=n, status='completed', dataset='imgr10', valid_performance=True,
                            **{k: 1. for k in METRICS}) for n in runner.modes('3090')]
            def fail_before_task_start_validation(directory):
                values = base.pairs_for(results, [], '3090')
                (directory / 'pairs.json').write_text(json.dumps(values))
                raise RuntimeError('base analysis interrupted')
            with patch.object(base, 'summarize_saved', side_effect=fail_before_task_start_validation):
                runner.summarize_safely(directory, '3090', [])
            self.assertEqual(json.loads((directory / 'pairs.json').read_text()), [])


if __name__ == '__main__':
    unittest.main()
