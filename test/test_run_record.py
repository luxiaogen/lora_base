import ast
import copy
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.run_record import start_run_record, update_run_record


class RunRecordTests(unittest.TestCase):
    def test_records_do_not_change_config_or_rng_and_never_reuse_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = dict(seed=1993, device=[torch.device('cpu')], logdir=tmp,
                        total_sessions=10, max_tasks=2, ca_epochs=5)
            original = copy.deepcopy(args)
            states = random.getstate(), np.random.get_state(), torch.get_rng_state()
            record = start_run_record(args, str(Path(tmp) / 'train.log'))
            other = start_run_record(args, str(Path(tmp) / 'train.log'))
            self.assertNotEqual(record, other)
            self.assertEqual(args, original)
            self.assertEqual(random.getstate(), states[0])
            np.testing.assert_equal(np.random.get_state(), states[1])
            self.assertTrue(torch.equal(torch.get_rng_state(), states[2]))
            config = json.loads((record / 'effective_config.json').read_text())
            self.assertEqual(config['device'], ['cpu'])
            self.assertEqual(config['ca_epochs'], 5)
            info = json.loads((record / 'run_info.json').read_text())
            self.assertEqual(info['train_log'], str((Path(tmp) / 'train.log').resolve()))
            self.assertIn('git_commit', info)
            self.assertIn('git_status', info)
            self.assertEqual(info['devices'], ['cpu'])
            result = json.loads((record / 'result.json').read_text())
            self.assertEqual(result['status'], 'running')
            self.assertEqual(result['tasks_completed'], 0)
            update_run_record(record, [dict(task=0, top1=90., old=None, new=90., forgetting=None)], 2, 3.)
            result = json.loads((record / 'result.json').read_text())
            self.assertEqual(result['status'], 'running')
            self.assertEqual(result['tasks_completed'], 1)
            tasks = [dict(task=0, top1=90., old=None, new=90., forgetting=None),
                     dict(task=1, top1=80., old=79., new=81., forgetting=2.)]
            update_run_record(record, tasks, 2, 5., completed=True)
            result = json.loads((record / 'result.json').read_text())
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['average_accuracy'], 85.)
            self.assertEqual(result['last_accuracy'], 80.)
            self.assertEqual(result['final_old'], 79.)
            self.assertEqual(result['stage_mean_new'], 81.)

    def test_unavailable_git_is_unknown_not_fake_clean(self):
        with tempfile.TemporaryDirectory() as tmp, patch('utils.run_record.subprocess.run', side_effect=OSError('no git')):
            path = start_run_record(dict(seed=1, device=['cpu'], logdir=tmp, total_sessions=1), 'train.log')
            info = json.loads((path / 'run_info.json').read_text())
            self.assertIsNone(info['git_commit'])
            self.assertIsNone(info['git_status'])

    def test_write_failure_does_not_interrupt_training(self):
        with patch('utils.run_record.Path.mkdir', side_effect=OSError('read only')), self.assertLogs(level='WARNING'):
            self.assertIsNone(start_run_record(dict(seed=1, logdir='/no-write', device=['cpu'], total_sessions=1), 'train.log'))
        update_run_record(None, [], 1, 0.)
        with tempfile.TemporaryDirectory() as tmp, patch('utils.run_record.Path.write_text', side_effect=OSError('full')), self.assertLogs(level='WARNING'):
            update_run_record(Path(tmp), [], 1, 0.)

    def test_frozen_recipe_matches_current_ca_reference(self):
        frozen = json.loads(Path('docs/experiments/baselines/imgr10_B0.json').read_text())
        self.assertEqual(frozen['seed'], [1993])
        self.assertNotIn('data_path', frozen)
        self.assertNotIn('device', frozen)
        for gpu in ('3090', '5090'):
            spec = json.loads(Path(f'scripts/sweeps/imgr10_ca_real_new_{gpu}.json').read_text())
            effective = {**spec['common_overrides'], **spec['variants'][0]['overrides']}
            for key, value in frozen.items():
                if key not in ('seed', 'dual_mask_conflict_exact_topk'):
                    self.assertEqual(effective[key], value, key)
        self.assertFalse(frozen['dual_mask_conflict_exact_topk'])
        self.assertFalse(frozen['ca_real_new_features'])

    def test_trainer_integration_uses_final_configuration_and_updates(self):
        tree = ast.parse(Path('trainer.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_train')
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        start = next(n for n in calls if n.func.id == 'start_run_record')
        device = next(n for n in calls if n.func.id == '_set_device')
        self.assertGreater(start.lineno, device.lineno)
        updates = [n for n in calls if n.func.id == 'update_run_record']
        self.assertEqual(len(updates), 2)
        self.assertTrue(any(any(k.arg == 'completed' and isinstance(k.value, ast.Constant) and k.value.value is True for k in n.keywords) for n in updates))

    def test_actual_trainer_loop_records_two_completed_tasks(self):
        import time
        tree = ast.parse(Path('trainer.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_train')
        class Model:
            _network = None
            acc_matrix = np.array([[90., 79.], [0., 81.]])
            task = 0
            def incremental_train(self, data):
                pass
            def eval_task(self):
                value = 90. if self.task == 0 else 80.
                return dict(top1=value, grouped={'old': 79., 'new': 81.}), dict(top1=95.), None, .9
            def after_task(self):
                self.task += 1
        model = Model()
        quiet_logging = SimpleNamespace(INFO=20, basicConfig=lambda **kw: None,
                                        FileHandler=lambda **kw: None, StreamHandler=lambda *a: None,
                                        info=lambda *a: None)
        namespace = dict(os=os, sys=__import__('sys'), time=time, logging=quiet_logging, np=np,
                         start_run_record=start_run_record, update_run_record=update_run_record,
                         _set_random=lambda args: None, _set_device=lambda args: None,
                         print_args=lambda args: None, count_parameters=lambda net: 0,
                         DataManager=lambda *a: SimpleNamespace(nb_tasks=2),
                         factory=SimpleNamespace(get_model=lambda *a: model),
                         _log_experiment_task=lambda *a: None, _log_experiment_summary=lambda *a: None)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'trainer.py', 'exec'), namespace)
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                args = dict(ca=True, prefix='paired', use_slora=True, use_plora=True,
                            dataset='fake', total_sessions=2, seed=1993, rank=2,
                            model_name='fake', optim='sgd', lrate=.02, device=['cpu'],
                            shuffle=True, init_cls=2, increment=2, max_tasks=2)
                namespace['_train'](args)
                result_path = next(Path(tmp).rglob('result.json'))
                result = json.loads(result_path.read_text())
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['tasks_completed'], 2)
                self.assertEqual(result['average_accuracy'], 85.)
                self.assertEqual(result['forgetting'], 11.)
            finally:
                os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
