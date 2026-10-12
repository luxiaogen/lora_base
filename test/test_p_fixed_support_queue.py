"""固定四组、短测先行、截止和续跑复用现有引擎。"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_p_fixed_support as runner
from dualmask_config import normalize_dualmask_config


class PFixedSupportQueueTests(unittest.TestCase):
    def test_only_declared_factor_changes_and_smoke_trains_after_capture(self):
        runner.validate_settings('3090')
        names = runner.modes('3090')
        self.assertEqual(names, ['imgr10_' + n + '_seed1993' for n in 'ABCD'])
        reference = runner.effective_config('3090', names[0])
        for name in names[1:]:
            actual = runner.effective_config('3090', name)
            differences = {k for k in actual.keys() | reference.keys()
                           if actual.get(k) != reference.get(k)}
            self.assertEqual(differences, {'p_fixed_support'})
        for name in names:
            smoke = runner.effective_config('3090', name, True)
            self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs'],
                              smoke['p_fixed_support_warmup_epochs']), (2, 1, 2, 1))
        self.assertEqual(reference['dual_mask_fixed_protect_strength'], .5)
        self.assertIsNone(reference['dual_mask_fixed_conflict_strength'])
        self.assertEqual(reference['dual_mask_protection_rule'], 'static')

    def test_configuration_rejects_incompatible_paths_and_invalid_selection(self):
        base = runner.effective_config('3090', runner.modes('3090')[1])
        for key, value in dict(p_fixed_support='unknown', p_fixed_support_keep_ratio=0,
            p_fixed_support_warmup_epochs=20, dual_mask_branch_layout='single',
            dual_mask_update_rule='soft_tail', p_conflict_freeze_epoch=5,
            dual_mask_position_norm_match='paired_min', plora_train_a=True).items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_dualmask_config(dict(base, **{key: value}))

    def test_all_smokes_before_formal_and_analysis_failure_does_not_stop_queue(self):
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run', side_effect=run), \
                patch.object(runner, 'summarize_safely') as summary, \
                contextlib.redirect_stdout(io.StringIO()):
            runner.budget.summarize_safely = runner.summarize_safely
            folder = Path(temp)
            self.assertEqual(runner.execute('3090', runner.modes('3090'), folder, 'r'), 0)
            self.assertEqual(calls, [(n, True) for n in runner.modes('3090')]
                             + [(n, False) for n in runner.modes('3090')])
            calls.clear()
            self.assertEqual(runner.execute('3090', runner.modes('3090'), folder, 'r'), 0)
            self.assertEqual(calls, [])
            self.assertTrue(summary.called)

    def test_deadline_and_training_failure_preserve_pending(self):
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run') as run, \
                patch.object(runner.budget, 'summarize_safely'), \
                contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temp)
            (folder / 'budget.json').write_text(json.dumps(dict(deadline_unix=0)))
            runner.execute('3090', runner.modes('3090'), folder, 'r')
            self.assertFalse(run.called)
            self.assertTrue(all(r['status'] == 'time_budget_pending'
                                for r in json.loads((folder / 'queue.json').read_text())))
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run', return_value=dict(
                    mode='imgr10_A_seed1993', status='failed', exit_code=7, minutes=1)) as run, \
                patch.object(runner.budget, 'summarize_safely'), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute('3090', runner.modes('3090'), Path(temp), 'r'), 7)
            self.assertEqual(run.call_count, 1)

    def test_analysis_error_is_separate_and_missing_pair_is_not_published(self):
        import analyze_p_fixed_support as analysis
        from analyze_core_evidence import METRICS
        a = dict(mode='imgr10_A_seed1993', valid_performance=True,
                 support_evidence_complete=True, **{k: 1. for k in METRICS})
        b = dict(a, mode='imgr10_B_seed1993', support_evidence_complete=False)
        self.assertEqual(analysis.pairs_for([a, b], []), [])
        b['support_evidence_complete'] = True
        self.assertEqual(len(analysis.pairs_for([a, b], [])), 1)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(analysis, 'summarize_saved', side_effect=RuntimeError('plot error')), \
                contextlib.redirect_stdout(io.StringIO()):
            runner.summarize_safely(Path(temp), '3090', [])
            self.assertIn('plot error', (Path(temp) / 'analysis_errors.jsonl').read_text())

    def test_uniform_evidence_requires_metrics_and_scale_aware_consistency(self):
        from analyze_p_fixed_support import uniform_match_valid
        valid = dict(extra_removed_norm=35.54029, reference_removed_norm=35.54,
                     matching_residual=.00029, uniform_coefficient=.5,
                     p_plastic_norm=71.08058)
        self.assertTrue(uniform_match_valid(valid))
        for key in valid:
            self.assertFalse(uniform_match_valid(dict(valid, **{key: None})))
        self.assertFalse(uniform_match_valid(dict(valid, matching_residual=0)))
        self.assertFalse(uniform_match_valid(dict(valid, uniform_coefficient=.3)))
        zero = dict.fromkeys(valid, 0.)
        self.assertTrue(uniform_match_valid(zero))

    def test_unreadable_failed_run_rewrites_partial_report_without_false_pair(self):
        import analyze_p_fixed_support as analysis
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'manifest.json').write_text(json.dumps(dict(machine='3090', revision='r')))
            (directory / 'queue.json').write_text(json.dumps([
                dict(mode='imgr10_A_seed1993', status='failed', exit_code=1),
                dict(mode='imgr10_B_seed1993', status='pending')]))
            (directory / 'pairs.csv').write_text('old pair')
            (directory / 'accuracy_old_new.pdf').write_bytes(b'old generated plot')
            analysis.summarize_saved(directory)
            rows = json.loads((directory / 'results.json').read_text())
            self.assertEqual(rows[0]['evidence_error'].split(':')[0], '[Errno 2] No such file or directory')
            self.assertFalse(rows[0]['valid_performance'])
            self.assertEqual(json.loads((directory / 'pairs.json').read_text()), [])
            self.assertEqual((directory / 'pairs.csv').read_text(), '')
            self.assertFalse((directory / 'accuracy_old_new.pdf').exists())


if __name__ == '__main__':
    unittest.main()
