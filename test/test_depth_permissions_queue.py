"""四格身份、时间边界和不完整证据的回归测试。"""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class DepthQueueTests(unittest.TestCase):
    def runner(self):
        try:
            return importlib.import_module('run_depth_permissions')
        except ImportError:
            self.fail('Depth queue implementation missing')

    def test_four_groups_only_change_declared_factors(self):
        runner = self.runner()
        runner.validate_settings('3090')
        names = runner.modes('3090')
        self.assertEqual(names, ['imgr10_' + n + '_seed1993' for n in 'ABCD'])
        reference = runner.effective_config('3090', names[0])
        expected = [set(), {'dual_mask_depth_coverage'}, {'dual_mask_depth_strength'},
                    {'dual_mask_depth_coverage', 'dual_mask_depth_strength'}]
        for name, differences in zip(names, expected):
            actual = runner.effective_config('3090', name)
            self.assertEqual({k for k in actual.keys() | reference.keys()
                              if actual.get(k) != reference.get(k)}, differences)
            smoke = runner.effective_config('3090', name, True)
            self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2, 1, 1))
        self.assertTrue(reference['dual_mask_conflict_old_overlap_adaptive'])
        self.assertFalse(reference['dual_mask_position_signal_audit'])
        with self.assertRaises(ValueError):
            runner.validate_settings('5090')

    def test_reject_incompatible_depth_recipe(self):
        runner = self.runner()
        from dualmask_config import normalize_dualmask_config
        base = runner.effective_config('3090', runner.modes('3090')[1])
        for key, value in dict(dual_mask_depth_strength='unknown', dual_mask_fixed_coverage=.8,
            dual_mask_protection_rule='energy_equal', dual_mask_protect_position='permuted',
            p_fixed_support='magnitude', dual_mask_position_signal_audit=True).items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_dualmask_config(dict(base, **{key: value}))

    def test_smokes_before_formal_and_completed_runs_not_repeated(self):
        runner = self.runner()
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run', side_effect=run), \
                patch.object(runner, 'smoke_issues', return_value=[]), \
                patch.object(runner.budget, 'summarize_safely'), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute('3090', runner.modes('3090'), Path(temp), 'r'), 0)
            self.assertEqual(calls, [(n, True) for n in runner.modes('3090')]
                             + [(n, False) for n in runner.modes('3090')])
            calls.clear()
            runner.execute('3090', runner.modes('3090'), Path(temp), 'r')
            self.assertEqual(calls, [])

    def test_deadline_and_smoke_failure_pause(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run') as run, \
                patch.object(runner.budget, 'summarize_safely'), \
                contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temp)
            (folder / 'budget.json').write_text(json.dumps(dict(deadline_unix=0)))
            runner.execute('3090', runner.modes('3090'), folder, 'r')
            self.assertFalse(run.called)
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run', return_value=dict(
                    mode='imgr10_A_seed1993', status='failed', exit_code=7, minutes=1)) as run, \
                patch.object(runner.budget, 'summarize_safely'), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute('3090', runner.modes('3090'), Path(temp), 'r'), 7)
            self.assertEqual(run.call_count, 1)

    def test_incomplete_or_mismatched_grid_has_no_interaction(self):
        self.runner()
        from analyze_depth_permissions import contrasts
        from analyze_core_evidence import METRICS
        rows = [dict(mode='imgr10_' + name + '_seed1993', valid_performance=True,
                     depth_evidence_complete=True, **{key: value for key in METRICS})
                for name, value in zip('ABCD', [1, 2, 3, 5])]
        self.assertNotIn('interaction', contrasts(rows[:3], []))
        self.assertEqual(contrasts(rows, [])['interaction']['Average'], 1)
        self.assertNotIn('interaction', contrasts(rows, [dict(mode=rows[-1]['mode'])]))
        self.assertNotIn('interaction', contrasts([dict(row, depth_evidence_complete=False)
                                                  for row in rows], []))

    def test_analysis_failure_is_saved_not_raised(self):
        runner = self.runner()
        import analyze_depth_permissions as analysis
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(analysis, 'summarize_saved', side_effect=RuntimeError('plot failed')), \
                contextlib.redirect_stdout(io.StringIO()):
            runner.summarize_safely(Path(temp), '3090', [])
            self.assertIn('plot failed', (Path(temp) / 'analysis_errors.jsonl').read_text())

    def test_smoke_guard_rejects_missing_permissions_and_bad_merge(self):
        runner = self.runner()
        self.assertTrue(hasattr(runner, 'smoke_issues'), 'GPU smoke evidence guard missing')
        import analyze_depth_permissions as analysis
        config = runner.effective_config('3090', runner.modes('3090')[0], True)
        depth = [dict(task=1, layer=layer, layers=12, coverage_mode='off', strength_mode='off',
                      coverage_target=.9, alpha=.5, s_rank=64, p_rank=64, reference_count=9,
                      protect_count=9, qkv_protect_counts=[3, 3, 3], protected_importance_fraction=.91)
                 for layer in range(12)]
        budgets = [dict(task=1, center=.9, reference_count=108, protect_count=108, count_residual=0)]
        self.assertEqual(analysis.permission_issues(depth, budgets, config, tasks=(1,)), [])
        self.assertTrue(analysis.permission_issues(depth[:-1], budgets, config, tasks=(1,)))
        self.assertTrue(analysis.permission_issues([dict(row, alpha=.75) for row in depth], budgets, config, tasks=(1,)))
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            snapshot = dict(effective_config=config, source_sha256={key: 'h' for key in runner.REQUIRED_SOURCE})
            (folder / 'run.json').write_text(json.dumps(snapshot))
            content = '\n'.join('DepthPermission ' + json.dumps(row) for row in depth)
            content += '\nDepthPermissionBudget ' + json.dumps(budgets[0])
            applied = [dict(task=task, layer=layer, branch=branch, merge_error=0.)
                       for task in (0, 1) for layer in range(12)
                       for branch in (('S',) if task == 0 else ('S', 'P'))]
            content += '\n' + '\n'.join('AppliedConflictBudget ' + json.dumps(row) for row in applied)
            (folder / 'training.log').write_text(content)
            self.assertEqual(runner.smoke_issues(folder, runner.modes('3090')[0]), [])
            applied[0]['merge_error'] = .01
            (folder / 'training.log').write_text(content + '\nAppliedConflictBudget ' + json.dumps(applied[0]))
            self.assertIn('merge_evidence', runner.smoke_issues(folder, runner.modes('3090')[0]))


if __name__ == '__main__':
    unittest.main()
