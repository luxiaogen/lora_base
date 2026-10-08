import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_baseline_suite as suite


class BaselineSuiteTests(unittest.TestCase):
    def test_regression_wrapper_pins_one_group_and_selected_python(self):
        env = dict(os.environ, PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE='1')
        output = subprocess.check_output(['bash', 'scripts/10_08_imgr10_baseline_regression_3090.sh',
                                          '--mode', 'dry-run'], env=env, text=True)
        commands = [shlex.split(line.removeprefix('Command: ')) for line in output.splitlines()
                    if line.startswith('Command: ')]
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[1:4], ['main.py','--config','exps/dlora/imgr10.json'])
            overrides = dict(command[i+1].split('=',1) for i,arg in enumerate(command) if arg == '--set')
            self.assertEqual(overrides['seed'], '[1993]')
            self.assertEqual(overrides['dual_mask_conflict_old_overlap_adaptive'], 'true')
            self.assertEqual(overrides['save_task_weights'], 'false')
        self.assertIn('1 T10 runs', output)

    def test_single_selected_group_runs_only_its_smoke_and_formal(self):
        selected = ['imgr10_seed1993']
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, patch.object(suite.engine, 'run', side_effect=run), \
                patch.object(suite, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            self.assertEqual(suite.execute(root, 'revision', selected=selected), 0)
            self.assertEqual(calls, [('imgr10_seed1993', True), ('imgr10_seed1993', False)])
            self.assertEqual([r['mode'] for r in json.loads((root / 'queue.json').read_text())], selected)

    def test_selection_smokes_once_per_dataset_without_stride_assumption(self):
        selected = ['imgr10_seed1996', 'imgr10_seed1993', 'imga10_seed1997']
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, patch.object(suite.engine, 'run', side_effect=run), \
                patch.object(suite, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(suite.execute(Path(temp), 'revision', selected=selected), 0)
        self.assertEqual(calls[:2], [('imgr10_seed1996', True), ('imga10_seed1997', True)])
        self.assertEqual(calls[2:], [(name, False) for name in selected])

    def test_cli_selection_prechecks_only_selected_dataset_and_records_selection(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(suite, 'ROOT', Path(temp)), \
                patch.object(suite, 'check_data', side_effect=lambda config: dict(dataset=config['dataset'])), \
                patch.object(suite, 'execute', return_value=0), \
                patch.object(suite.subprocess, 'check_output', return_value='revision'), \
                patch.object(sys, 'argv', ['run_baseline_suite.py', '--modes', 'imgr10_seed1993']), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            # Fixtures contain only ImageNet-R: reading any unrelated config must fail.
            config = root / 'exps/dlora/imgr10.json'
            config.parent.mkdir(parents=True)
            config.write_text((Path(__file__).resolve().parents[1] / 'exps/dlora/imgr10.json').read_text())
            self.assertEqual(suite.main(), 0)
            manifest_path = next(root.glob('logs/shell_logs/baseline_4datasets_3090/*/manifest.json'))
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest['modes'], ['imgr10_seed1993'])
            self.assertEqual(len(manifest['datasets']), 1)
            self.assertEqual(manifest['datasets'][0]['dataset'], 'ImageNet_R')

    def test_cli_rejects_unknown_and_repeated_modes_before_running(self):
        for selected in (['imgr10_seed0'], ['imgr10_seed1993','imgr10_seed1993']):
            with patch.object(sys, 'argv', ['run_baseline_suite.py','--mode','dry-run','--modes',*selected]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                suite.main()
            self.assertNotEqual(error.exception.code, 0)

    def test_resume_selection_cannot_change_original_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text(json.dumps(dict(revision='revision', modes=['imgr10_seed1993'])))
            suite.check_resume(root, 'revision', selected=['imgr10_seed1993'])
            with self.assertRaisesRegex(ValueError, 'selection|modes|selected'):
                suite.check_resume(root, 'revision', selected=['imgr10_seed1996'])

    def test_exact_requested_matrix_and_dataset_recipes(self):
        expected = [dataset + '_seed' + str(seed)
                    for dataset in ('cifar100', 'imga10', 'cub10', 'imgr10')
                    for seed in (1993, 1996, 1997)]
        self.assertEqual(suite.modes(), expected)
        for name in expected:
            item, seed = suite.run_identity(name)
            source = json.loads((suite.ROOT / item['config']).read_text())
            source.update(item.get('overrides', {}))
            settings = suite.settings_for('3090', name)
            effective = dict(source, **settings)
            for field in suite.DATASET_FIELDS:
                if field in source:
                    self.assertEqual(effective[field], source[field], (name, field))
            self.assertEqual(settings['seed'], [seed])
            self.assertEqual(settings['dual_mask_anchor_reg_weight'], 2.5)
            self.assertEqual(settings['task0_margin'], source['margin'])
            self.assertEqual(settings['dual_mask_reg_weight'], .01)
            self.assertEqual(settings['dual_mask_conflict_score_mode'], 'conflict')
            self.assertFalse(settings['dual_mask_conflict_exact_topk'])
            self.assertEqual(settings['dual_mask_private_rank'], 0)
            self.assertTrue(settings['disable_fused_sdpa'])
            self.assertFalse(settings['save_task_weights'])
            self.assertTrue(effective['ca'])
            self.assertEqual(effective['ca_epochs'], 5)
            self.assertEqual((settings['max_tasks'], settings['init_epoch'], settings['epochs']), (10, 20, 20))
            self.assertIsNone(settings['dual_mask_fixed_coverage'])
            for flag in ('p_old_gradient_oracle', 'plora_train_a', 'ridge_fusion_enabled'):
                self.assertFalse(settings[flag])
            smoke = suite.settings_for('3090', name, True)
            self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2, 1, 1))
            command, _ = suite.command_for('3090', name, Path('/tmp/fixture/formal'))
            self.assertEqual(command[1:4], ['main.py', '--config', item['config']])

    def test_only_four_smokes_then_twelve_formal_runs(self):
        calls = []
        def run(machine, name, directory, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            return dict(mode=name, status='completed', exit_code=0, minutes=1)
        with tempfile.TemporaryDirectory() as temp, patch.object(suite.engine, 'run', side_effect=run), \
                patch.object(suite, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            self.assertEqual(suite.execute(root, 'revision'), 0)
            self.assertEqual(calls[:4], [(n, True) for n in suite.modes()[::3]])
            self.assertEqual(calls[4:], [(n, False) for n in suite.modes()])
            self.assertEqual(len(json.loads((root / 'queue.json').read_text())), 12)

    def test_training_failure_pauses_and_analysis_failure_does_not(self):
        for smoke_failure in (True, False):
            calls = []
            def run(machine, name, directory, revision, smoke=False, dry_run=False):
                calls.append((name, smoke))
                return dict(mode=name, status='failed', exit_code=7, minutes=0) if smoke == smoke_failure else \
                    dict(mode=name, status='completed', exit_code=0, minutes=0)
            with tempfile.TemporaryDirectory() as temp, patch.object(suite.engine, 'run', side_effect=run), \
                    patch.object(suite, 'summarize', side_effect=RuntimeError('analysis only')), \
                    contextlib.redirect_stdout(io.StringIO()):
                root = Path(temp)
                self.assertEqual(suite.execute(root, 'revision'), 7)
                self.assertEqual(len(calls), 1 if smoke_failure else 5)
        with tempfile.TemporaryDirectory() as temp, patch.object(suite.engine, 'run',
                side_effect=lambda m,n,d,r,smoke=False,dry_run=False:
                    dict(mode=n, status='completed', exit_code=0, minutes=0)), \
                patch.object(suite, 'summarize', side_effect=RuntimeError('plot only')), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            self.assertEqual(suite.execute(root, 'revision'), 0)
            self.assertTrue((root / 'analysis_errors.jsonl').exists())
            self.assertEqual(len(json.loads((root / 'queue.json').read_text())), 12)

    def test_completed_groups_not_repeated_and_dry_run_has_no_writes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(suite.engine, 'run',
                side_effect=lambda m,n,d,r,smoke=False,dry_run=False:
                    dict(mode=n, status='completed', exit_code=0, minutes=0)) as run, \
                patch.object(suite, 'summarize'), contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            first = suite.modes()[0]
            (root / 'queue.json').write_text(json.dumps([dict(mode=first,status='completed',exit_code=0)]))
            (root / 'smoke_queue.json').write_text(json.dumps([
                dict(mode=n,status='completed',exit_code=0) for n in suite.modes()[::3]]))
            self.assertEqual(suite.execute(root, 'revision'), 0)
            self.assertEqual(run.call_count, 11)
            self.assertNotIn(first, [c.args[1] for c in run.call_args_list])
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp) / 'not-created'
            self.assertEqual(suite.execute(root, 'revision', 'dry-run'), 0)
            self.assertFalse(root.exists())

    def test_original_recipe_cannot_be_replaced_by_compact_controls(self):
        for field, value in (('dual_mask_private_rank',40), ('dual_mask_conflict_score_mode','magnitude'),
                             ('dual_mask_fixed_coverage',.825), ('dual_mask_reg_weight',0)):
            bad = suite.settings_for('3090', suite.modes()[0])
            bad[field] = value
            with patch.object(suite, 'settings_for', return_value=bad):
                self.assertRaises(ValueError, suite.validate_settings)

    def test_missing_data_prevents_automatic_split_or_download(self):
        with tempfile.TemporaryDirectory() as temp:
            for dataset in ('ImageNet_R', 'ImageNet_A', 'CUB', 'cifar100'):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    suite.check_data(dict(dataset=dataset, data_path=temp))

    def test_partial_seeds_not_reported_as_three_seed_mean(self):
        measured = dict(mode=suite.modes()[0], status='completed', exit_code=0,
                        valid_performance=True, **{k:90 for k in suite.METRICS})
        with tempfile.TemporaryDirectory() as temp, patch.object(suite, 'read_run',
                return_value=(measured, {'effective_config':{'dataset':'cifar100','seed':[1993]}}, {'tasks':[]})):
            root = Path(temp)
            suite.summarize(root, [dict(mode=suite.modes()[0],status='completed',exit_code=0)])
            aggregate = json.loads((root / 'aggregate.json').read_text())
            self.assertFalse(aggregate[0]['complete_three_seeds'])
            self.assertNotIn('Average_mean', aggregate[0])

    def test_engine_snapshot_uses_actual_dataset_config_not_imagenet_r(self):
        class Process:
            pid = 12345
            stdout = ("[trainer.py] => CNN: {'total': 80., 'old': 79., 'new': 81.}\n" * 2
                + '[trainer.py] => Average Accuracy: 80.0\n[trainer.py] => Forgetting: 5.0\n').splitlines(True)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def wait(self):
                return 0
        with tempfile.TemporaryDirectory() as temp, patch.object(suite.engine, 'command_for', side_effect=suite.command_for), \
                patch.object(suite.engine.subprocess, 'Popen', return_value=Process()), \
                patch.object(suite.engine.subprocess, 'check_output', return_value='0,fixture-GPU,fixture'), \
                patch.object(suite.engine.importlib.metadata, 'version', return_value='fixture'), \
                patch.object(suite.engine.platform, 'platform', return_value='fixture'), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp) / 'smoke_cifar100_seed1993'
            suite.engine.run('3090', 'cifar100_seed1993', root, 'fixture', smoke=True)
            snapshot = json.loads((root / 'run.json').read_text())
            self.assertEqual(snapshot['effective_config']['dataset'], 'cifar100')
            self.assertEqual(snapshot['effective_config']['rank'], 32)
            self.assertEqual(snapshot['effective_config']['init_lr'], .005)
            self.assertTrue(snapshot['effective_config']['ca'])
            self.assertIn('exps/dlora/cifar10.json', snapshot['source_sha256'])

    def test_resume_requires_matching_recipe_and_source_and_finite_metrics(self):
        name = suite.modes()[0]
        item, _ = suite.run_identity(name)
        config = dict(json.loads((suite.ROOT / item['config']).read_text()), **suite.settings_for('3090', name))
        snapshot = dict(code_revision='rev', machine='3090', phase='formal',
                        effective_config=config, source_sha256={})
        measured = dict(tasks_reported=10, runtime_error=False, **{k:90 for k in suite.METRICS})
        with tempfile.TemporaryDirectory() as temp, patch.object(suite, 'read_run', return_value=(measured,snapshot,{})):
            root = Path(temp)
            (root / 'manifest.json').write_text(json.dumps(dict(revision='rev')))
            (root / 'queue.json').write_text(json.dumps([dict(mode=name,status='completed',exit_code=0)]))
            suite.check_resume(root, 'rev')
            measured['Average'] = float('nan')
            self.assertRaises(ValueError, suite.check_resume, root, 'rev')
            measured['Average'] = 90
            config['ca'] = False
            self.assertRaises(ValueError, suite.check_resume, root, 'rev')


if __name__ == '__main__':
    unittest.main()
