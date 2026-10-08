import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_region_prior as suite


class RegionPriorTests(unittest.TestCase):
    def fixture(self, name):
        config = dict(json.loads((suite.ROOT / 'exps/dlora/imgr10.json').read_text()),
                      **suite.settings_for('3090', name))
        snapshot = dict(code_revision='fixture', machine='3090', phase='formal',
                        effective_config=config,
                        source_sha256={p: hashlib.sha256((suite.ROOT / p).read_bytes()).hexdigest()
                                       for p in suite.REQUIRED_SOURCE},
                        hardware={'gpus': ['fixture-GPU']}, software={'python': 'fixture'})
        measured = dict(valid_performance=True, Task0=97., tasks_reported=10, runtime_error=False,
                        **{key: 90. for key in suite.METRICS})
        rows = dict(tasks=[],
            masks=[dict(mode=name, task=t, layer=l, qkv_protect_counts=[5, 5, 5])
                   for t in range(10) for l in range(12)],
            updates=[dict(mode=name, task=t, layer=l, branch=b, projection=p)
                     for t in range(10) for l in range(12)
                     for b in (('S',) if t == 0 else ('S', 'P')) for p in ('Q', 'K', 'V')])
        return measured, snapshot, rows

    def test_only_region_changes_in_original_recipe(self):
        self.assertEqual(len(suite.modes()), 9)
        for seed in (1993, 1996, 1997):
            reference = suite.baseline.settings_for('3090', 'imgr10_seed' + str(seed))
            for variant in ('A_spectrum', 'B_weight', 'C_permuted'):
                name = variant + '_seed' + str(seed)
                settings = suite.settings_for('3090', name)
                changed = {key for key in reference.keys() | settings.keys()
                           if reference.get(key) != settings.get(key)}
                self.assertLessEqual(changed, {'dual_mask_protect_position', 'dual_mask_position_audit',
                                               'wandb_group'})
                self.assertTrue(settings['dual_mask_position_audit'])
                self.assertFalse(settings['dual_mask_mechanism_audit'])
                self.assertFalse(settings['save_task_weights'])
                self.assertEqual(settings['seed'], [seed])
                effective = dict(json.loads((suite.ROOT / 'exps/dlora/imgr10.json').read_text()), **settings)
                self.assertEqual((effective['max_tasks'], effective['epochs'], effective['ca_epochs']), (10, 20, 5))
                self.assertEqual(settings['dual_mask_conflict_score_mode'], 'conflict')
                self.assertEqual(settings['dual_mask_anchor_reg_weight'], 2.5)
                command, _ = suite.command_for('3090', name, Path('/tmp/fixture/run'))
                self.assertEqual(command[1:4], ['main.py', '--config', 'exps/dlora/imgr10.json'])

    def test_three_smokes_then_nine_formal_units_without_time_cutoff(self):
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, patch.object(suite.engine, 'run', side_effect=run), \
                patch.object(suite, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(suite.execute(Path(temp), 'fixture'), 0)
        self.assertEqual(calls[:3], [(v + '_seed1993', True) for v in ('A_spectrum', 'B_weight', 'C_permuted')])
        self.assertEqual(calls[3:], [(name, False) for name in suite.modes()])

    def test_reuse_completed_units_and_stop_on_training_failure(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(suite.engine, 'run',
                return_value=dict(status='completed', exit_code=0)) as run, \
                patch.object(suite, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            (root / 'queue.json').write_text(json.dumps([dict(mode=suite.modes()[0], status='completed')]))
            (root / 'smoke_queue.json').write_text(json.dumps([
                dict(mode=v + '_seed1993', status='completed') for v in ('A_spectrum', 'B_weight', 'C_permuted')]))
            self.assertEqual(suite.execute(root, 'fixture'), 0)
            self.assertEqual(run.call_count, 8)
        with tempfile.TemporaryDirectory() as temp, patch.object(suite.engine, 'run',
                return_value=dict(status='failed', exit_code=7)) as run, \
                patch.object(suite, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(suite.execute(Path(temp), 'fixture'), 7)
            self.assertEqual(run.call_count, 1)

    def test_partial_results_do_not_produce_three_seed_aggregates_or_pairs(self):
        measured = dict(valid_performance=True, **{key: 90. for key in suite.METRICS})
        snapshot = dict(effective_config={'dual_mask_protect_position': 'wpre', 'seed': [1993]})
        rows = dict(tasks=[], updates=[], masks=[])
        with tempfile.TemporaryDirectory() as temp, patch.object(suite, 'read_run',
                return_value=(measured, snapshot, rows)):
            root = Path(temp)
            suite.summarize(root, [dict(mode=suite.modes()[0], status='completed', exit_code=0)])
            report = json.loads((root / 'aggregate.json').read_text())
            self.assertFalse(report['complete_nine_units'])
            self.assertEqual(report['paired_differences'], [])
            self.assertEqual(report['group_means'], [])

    def test_complete_grid_checks_variant_seed_machine_and_cross_seed_sources(self):
        records = [dict(mode=name, status='completed', exit_code=0) for name in suite.modes()]
        for wrong in (None, 'seed', 'position', 'machine', 'source', 'budget'):
            values = {name: self.fixture(name) for name in suite.modes()}
            target = values[suite.modes()[3]]
            if wrong == 'seed':
                target[1]['effective_config']['seed'] = [1993]
            elif wrong == 'position':
                values[suite.modes()[1]][1]['effective_config']['dual_mask_protect_position'] = 'wpre'
            elif wrong == 'machine':
                target[1]['machine'] = '5090'
            elif wrong == 'source':
                for name in suite.modes()[3:6]:
                    values[name][1]['source_sha256'] = {'different': 'source'}
            elif wrong == 'budget':
                values[suite.modes()[1]][2]['masks'][0]['qkv_protect_counts'][0] += 1
            with tempfile.TemporaryDirectory() as temp, patch.object(suite, 'read_run',
                    side_effect=lambda directory, record: copy.deepcopy(values[record['mode']])):
                root = Path(temp)
                suite.summarize(root, records)
                report = json.loads((root / 'aggregate.json').read_text())
                self.assertEqual(report['complete_nine_units'], wrong is None, wrong)
                self.assertEqual(len(report['group_means']), 3 if wrong is None else 0, wrong)
                if wrong is not None:
                    self.assertTrue(report['matching_issues'], wrong)

    def test_resume_requires_zero_exit_finite_metrics_and_complete_telemetry(self):
        name = suite.modes()[0]
        for wrong in (None, 'exit', 'finite', 'telemetry'):
            measured, snapshot, rows = self.fixture(name)
            record = dict(mode=name, status='completed', exit_code=7 if wrong == 'exit' else 0)
            if wrong == 'finite':
                measured['Average'] = float('nan')
            if wrong == 'telemetry':
                rows['updates'].pop()
            with tempfile.TemporaryDirectory() as temp, patch.object(suite, 'read_run',
                    return_value=(measured, snapshot, rows)):
                root = Path(temp)
                (root / 'manifest.json').write_text(json.dumps(dict(revision='fixture')))
                (root / 'queue.json').write_text(json.dumps([record]))
                if wrong is None:
                    suite.check_resume(root, 'fixture')
                else:
                    self.assertRaises(ValueError, suite.check_resume, root, 'fixture')

    def test_resume_does_not_overwrite_interrupted_or_running_unit(self):
        name = suite.modes()[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text(json.dumps(dict(revision='fixture')))
            (root / 'queue.json').write_text(json.dumps([dict(mode=name, status='pending')]))
            (root / name).mkdir()
            (root / name / 'training.log').write_text('preserve')
            self.assertRaisesRegex(ValueError, 'Interrupted or still-running unit preserved',
                                   suite.check_resume, root, 'fixture')
            self.assertEqual((root / name / 'training.log').read_text(), 'preserve')

    def test_invalid_reanalysis_clears_derived_csv_not_raw_records(self):
        records = [dict(mode=name, status='completed', exit_code=0) for name in suite.modes()]
        values = {name: self.fixture(name) for name in suite.modes()}
        with tempfile.TemporaryDirectory() as temp, patch.object(suite, 'read_run',
                side_effect=lambda directory, record: copy.deepcopy(values[record['mode']])):
            root = Path(temp)
            suite.summarize(root, records)
            self.assertIn('Average_mean', (root / 'aggregate.csv').read_text())
            for _, snapshot, _ in values.values():
                snapshot['source_sha256'] = {}
            suite.summarize(root, records)
            self.assertEqual((root / 'aggregate.csv').read_text(), 'variant,n\n')
            self.assertEqual((root / 'pairs.csv').read_text(), 'seed,contrast\n')

    def test_cached_pretrained_fingerprint_never_downloads_or_requires_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / 'checkpoints'
            cache.mkdir()
            fake_torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: temp))
            fake_vit = SimpleNamespace(resolve_pretrained_cfg=lambda name: {'url': 'https://fixture/a.npz'})
            with patch.dict(sys.modules, {'torch': fake_torch, 'models.vit': fake_vit}):
                missing = suite.pretrained_fingerprint()
                self.assertFalse(missing['exists'])
                self.assertIsNone(missing['sha256'])
                (cache / 'a.npz').write_bytes(b'cached fixture')
                present = suite.pretrained_fingerprint()
                self.assertTrue(present['exists'])
                self.assertEqual(present['sha256'], hashlib.sha256(b'cached fixture').hexdigest())


if __name__ == '__main__':
    unittest.main()
