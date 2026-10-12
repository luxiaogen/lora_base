"""只跑一组F；短测先行，诊断不完整不能成为机制结论。"""
import contextlib
import importlib
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class PositionSignalQueueTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('run_protection_position_diagnostic'),
                             '缺少单组F诊断入口')
        self.runner = importlib.import_module('run_protection_position_diagnostic')

    def test_recipe_only_changes_diagnostic_metadata_and_has_one_formal_group(self):
        import run_p_fixed_support as old
        runner = self.runner
        runner.validate_settings('3090')
        self.assertEqual(runner.modes('3090'), ['imgr10_F_seed1993'])
        previous = old.effective_config('3090', 'imgr10_A_seed1993')
        current = runner.effective_config('3090', 'imgr10_F_seed1993')
        changed = {k for k in previous.keys() | current.keys() if previous.get(k) != current.get(k)}
        self.assertLessEqual(changed, {'dual_mask_position_signal_audit', 'position_signal_tasks',
            'position_signal_epochs', 'p_fixed_support_audit', 'wandb_group'})
        self.assertEqual(current['p_fixed_support'], 'off')
        self.assertEqual(current['position_signal_tasks'], [1, 5, 9])
        smoke = runner.effective_config('3090', 'imgr10_F_seed1993', True)
        self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2, 1, 3))
        self.assertEqual(smoke['position_signal_epochs'], [2, 3])
        with self.assertRaises(ValueError):
            runner.validate_settings('5090')

    def test_runs_smoke_before_formal_and_never_retries_failure(self):
        runner = self.runner
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append(smoke)
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run', side_effect=run), \
                patch.object(runner, 'verify_diagnostics', return_value=[]), \
                patch.object(runner, 'smoke_fingerprint_issues', return_value=[]), \
                patch.object(runner.budget, 'summarize_safely'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute('3090', runner.modes('3090'), Path(temp), 'r'), 0)
        self.assertEqual(calls, [True, False])
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run', return_value=dict(mode='imgr10_F_seed1993',
                    status='failed', exit_code=7, minutes=1)) as run, \
                patch.object(runner.budget, 'summarize_safely'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute('3090', runner.modes('3090'), Path(temp), 'r'), 7)
            self.assertEqual(run.call_count, 1)

    def test_summary_keeps_incomplete_diagnostics_out_of_evidence(self):
        api = importlib.import_module('analyze_protection_position_diagnostic')
        complete = {
            'PositionSignalPrepared': [dict(task=t, score_old_disjoint=True,
                sample_counts={'score': 2, 'old': 2, 'new': 2}, seconds=1., fisher_seconds=.5, cache_bytes=1)
                for t in (1, 5, 9)],
            'PositionSignalFisher': [dict(task=t, layer=l, samples=2, draws_per_image=2, scale=20.,
                classifier_scope='previous_seen_classes', source='old_train_score_only',
                s_b_norm=0., p_b_norm=0., draw_correlation=.9, draw_mask_jaccard=.8,
                fisher_sum=1., residual_norm=.1) for t in (1, 5, 9) for l in range(12)],
            'PositionSignalScore': [dict(task=t, layer=l, source=s, mass=1.,
                quantiles=[0., .2, .4, .6, .8, 1.], fisher_correlation=.5)
                for t in (1, 5, 9) for l in range(12) for s in ('spectrum', 'residual')],
            'PositionSignalCost': [dict(task=t, epoch=e, seconds=1.) for t in (1, 5, 9) for e in (2, 20)],
            'PositionSignalMask': [dict(task=t, layer=l, position=p, qkv_counts=[3, 4, 5],
                reference_counts=[3, 4, 5]) for t in (1, 5, 9) for l in range(12)
                for p in ('W', 'WR', 'WRperm', 'MC', 'Random')],
            'PositionSignalUpdate': [dict(task=t, epoch=e, layer=l, position=p, scale=.9,
                s_effective_norm=1., p_effective_norm=2.) for t in (1, 5, 9) for e in (2, 20)
                for l in range(12) for p in ('W', 'WR', 'WRperm', 'MC', 'Random')],
            'PositionSignalIntervention': [dict(task=t, epoch=e, partition=a, position=p, count=2,
                source='independent_train_diagnostic', prediction_reference='W_matched', accuracy=50.,
                loss=1., margin=.1) for t in (1, 5, 9) for e in (2, 20) for a in ('old', 'new')
                for p in ('W', 'WR', 'WRperm', 'MC', 'Random', 'ActualF')]
        }
        self.assertEqual(api.evidence_issues(complete), [])
        for marker in ('PositionSignalScore', 'PositionSignalCost'):
            missing = dict(complete, **{marker: []})
            self.assertIn(marker, api.evidence_issues(missing))
        complete['PositionSignalFisher'][0].pop('draw_mask_jaccard')
        self.assertIn('invalid_fisher_evidence', api.evidence_issues(complete))
        complete['PositionSignalFisher'][0]['draw_mask_jaccard'] = .8
        complete['PositionSignalFisher'][0]['p_b_norm'] = .1
        self.assertIn('invalid_fisher_evidence', api.evidence_issues(complete))
        complete['PositionSignalFisher'][0]['p_b_norm'] = 0.
        complete['PositionSignalUpdate'][0]['p_effective_norm'] = 3.
        self.assertIn('p_changed', api.evidence_issues(complete))
        complete['PositionSignalMask'].pop()
        self.assertIn('PositionSignalMask', api.evidence_issues(complete))

    def test_changed_smoke_formal_fingerprints_block_attribution(self):
        api = importlib.import_module('analyze_protection_position_diagnostic')
        self.assertTrue(hasattr(api, 'fingerprint_issues'), '缺少短测/正式源码环境配对检查')
        smoke = dict(code_revision='rev', machine='3090', phase='smoke',
            source_sha256={path: 'a' * 64 for path in self.runner.REQUIRED_SOURCE},
            hardware={'uuid': 'same_gpu'}, software={'torch': 'same_version'})
        formal = dict(smoke, phase='formal')
        self.assertEqual(api.fingerprint_issues(formal, smoke, 'rev'), [])
        for field in ('source_sha256', 'hardware', 'software'):
            changed = dict(formal, **{field: {'changed': 'value'}})
            self.assertIn(field, api.fingerprint_issues(changed, smoke, 'rev'))
        self.assertIn('missing_smoke_snapshot', api.fingerprint_issues(formal, None, 'rev'))

    def test_preflight_drift_stops_before_formal_process_is_created(self):
        runner = self.runner
        self.assertTrue(hasattr(runner, 'smoke_fingerprint_issues'), '正式启动前缺少短测指纹检查')
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append(smoke)
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run', side_effect=run), \
                patch.object(runner, 'verify_diagnostics', return_value=[]), \
                patch.object(runner, 'smoke_fingerprint_issues', return_value=['source_sha256']), \
                patch.object(runner.budget, 'summarize_safely'), contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temp)
            self.assertEqual(runner.execute('3090', runner.modes('3090'), directory, 'r'), 3)
            record = json.loads((directory / 'queue.json').read_text())[0]
            self.assertEqual(record['reason'], 'smoke_fingerprint_mismatch')
            self.assertEqual(record['issues'], ['source_sha256'])
        self.assertEqual(calls, [True])

    def test_preflight_reads_current_source_before_start(self):
        runner = self.runner
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = {}
            for name in runner.REQUIRED_SOURCE:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original')
                sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            formal = root / 'run' / 'imgr10_F_seed1993'
            smoke = formal.parent / ('smoke_' + formal.name)
            smoke.mkdir(parents=True)
            snapshot = dict(code_revision='rev', machine='3090', phase='smoke', source_sha256=sources,
                hardware=dict(hostname='host', cuda_visible_devices='0', gpus=['gpu']),
                software=dict(python=sys.version, platform='platform',
                    packages={p: 'test' for p in ('torch', 'torchvision', 'timm', 'numpy')}))
            (smoke / 'run.json').write_text(json.dumps(snapshot))
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner.subprocess, 'check_output', side_effect=lambda cmd, **kw:
                        'rev\n' if cmd[0] == 'git' else 'gpu\n'), \
                    patch.object(runner.platform, 'node', return_value='host'), \
                    patch.object(runner.platform, 'platform', return_value='platform'), \
                    patch.object(runner.importlib.metadata, 'version', return_value='test'), \
                    patch.dict(runner.os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
                self.assertEqual(runner.smoke_fingerprint_issues(formal, 'rev'), [])
                (root / 'models/attention.py').write_text('changed')
                self.assertIn('source_sha256', runner.smoke_fingerprint_issues(formal, 'rev'))

    def test_deadline_and_analysis_error_are_separate(self):
        runner = self.runner
        with tempfile.TemporaryDirectory() as temp, runner.configured_runner(), \
                patch.object(runner.budget.tail.engine, 'run') as run, \
                patch.object(runner.budget, 'summarize_safely'), contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temp)
            (directory / 'budget.json').write_text(json.dumps(dict(deadline_unix=0)))
            runner.execute('3090', runner.modes('3090'), directory, 'r')
            self.assertFalse(run.called)
        with tempfile.TemporaryDirectory() as temp, \
                patch('analyze_protection_position_diagnostic.summarize_saved', side_effect=RuntimeError('plot failure')):
            runner.summarize_safely(Path(temp), '3090', [])
            self.assertIn('plot failure', (Path(temp) / 'analysis_errors.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
