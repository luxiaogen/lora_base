"""通过真实子进程检查四组队列；不导入训练依赖，也不占用 GPU。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


BUNDLE = Path(__file__).resolve().parents[1]
RUNNER = BUNDLE / 'scripts/run_ca_drift.py'
GROUPS = ['A_baseline', 'B_full', 'C_isotropic', 'D_permuted']

FAKE = r'''import argparse, hashlib, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--config'); p.add_argument('--data-path'); p.add_argument('--device')
p.add_argument('--seed', type=int); p.add_argument('--epochs', type=int)
p.add_argument('--ca-epochs', type=int); p.add_argument('--max-tasks', type=int)
p.add_argument('--output'); p.add_argument('--set', action='append', default=[])
a = p.parse_args()
cfg = json.loads(Path(a.config).read_text())
cfg.update(dict(x.split('=', 1) for x in a.set))
cfg = {k: (json.loads(v) if k == 'ca_drift_augmentation' else v) for k,v in cfg.items()}
cfg.update(data_path=a.data_path, device=['cuda:' + a.device], seed=a.seed,
           init_epoch=a.epochs, epochs=a.epochs, ca_epochs=a.ca_epochs,
           max_tasks=a.max_tasks, output=a.output, logdir=a.output)
mode = cfg['ca_drift_augmentation']
phase = 'smoke' if a.max_tasks == 2 else 'full'
with open(os.environ['FAKE_CALLS'], 'a') as f:
    f.write(json.dumps(dict(phase=phase, mode=mode, pid=os.getpid(), cfg=cfg)) + '\n')
time.sleep(float(os.environ.get('FAKE_SLEEP', '0')))
if phase == 'full':
    time.sleep(float(os.environ.get('FAKE_FULL_SLEEP', '0')))
if os.environ.get('FAKE_FAIL') == phase + ':' + mode:
    sys.exit(9)
out = Path(a.output); out.mkdir(parents=True)
bundle = Path(os.environ['FAKE_BUNDLE'])
root = Path(os.environ['FAKE_ROOT'])
source = {str(x.relative_to(bundle)): hashlib.sha256(x.read_bytes()).hexdigest()
          for x in bundle.rglob('*') if x.is_file() and x.suffix in ('.py','.json','.sh')
          and root not in x.parents and not {'.git','__pycache__','runs','logs'} & set(x.parts)}
(out / 'effective_config.json').write_text(json.dumps(cfg))
(out / 'run_info.json').write_text(json.dumps(dict(source_sha256=source,
    gpu_names=['NVIDIA GeForce RTX 5090'], effective_config=cfg)))
tasks = [dict(task=i, top1=90-i, old=77, new=97, forgetting=4 if i else None,
              ca_drift=dict(mode=mode, samples=20, drift_rms=0.25, increment_trace=1.5) if i else {})
         for i in range(a.max_tasks)]
result = dict(status='completed', tasks_completed=a.max_tasks, planned_tasks=a.max_tasks,
    tasks=tasks, summary=dict(Average=83, Last=79, Old=77, New=97,
    StageOld=78, StageNew=95, Forgetting=4))
if os.environ.get('FAKE_INCOMPLETE') == phase + ':' + mode:
    result.update(status='failed', tasks_completed=1)
if os.environ.get('FAKE_BAD_ANALYSIS') == mode and phase == 'full':
    result['summary'] = {}
if os.environ.get('FAKE_BAD_TASKS') == phase + ':' + mode:
    result['tasks'][1]['task'] = 0
(out / 'result.json').write_text('bad json' if os.environ.get('FAKE_BAD_RESULT') == phase + ':' + mode
                                  else json.dumps(result))
if os.environ.get('FAKE_BLOCK_SUMMARY') == mode and phase == 'full':
    (out.parent / 'result.json').mkdir()
print('CADrift ' + json.dumps(dict(mode=mode, eta=0.25)))
'''


class CADriftQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.root = self.path / 'queue'
        self.calls_path = self.path / 'calls.jsonl'
        self.entrypoint = self.path / 'fake_train.py'
        self.entrypoint.write_text(FAKE)
        self.config = self.path / 'config.json'
        self.config.write_text(json.dumps(dict(dataset='ImageNet_R', data_path='unused',
            seed=1993, device='0', init_cls=20, increment=20, total_sessions=10,
            rank=64, ca_drift_augmentation='off')))
        self.data = self.path / 'data'
        for split in ('train', 'test'):
            (self.data / split / 'class1').mkdir(parents=True)
            (self.data / split / 'class1/a.jpg').write_bytes(b'fixture')
        bin_path = self.path / 'bin'
        bin_path.mkdir()
        gpu = bin_path / 'nvidia-smi'
        gpu.write_text('#!' + sys.executable + '''
import os, time
from pathlib import Path
if os.environ.get('FAKE_GPU_STARTED'):
    Path(os.environ['FAKE_GPU_STARTED']).touch()
time.sleep(float(os.environ.get('FAKE_GPU_SLEEP', '0')))
print(os.environ.get('FAKE_GPU_NAME', 'NVIDIA GeForce RTX 5090'))
''')
        gpu.chmod(0o755)
        self.env = dict(os.environ, PATH=str(bin_path) + os.pathsep + os.environ['PATH'],
            FAKE_CALLS=str(self.calls_path), FAKE_BUNDLE=str(BUNDLE), FAKE_ROOT=str(self.root))

    def command(self, *extra):
        return [sys.executable, str(RUNNER), '--root', str(self.root), '--config',
                str(self.config), '--data-path', str(self.data), '--entrypoint',
                str(self.entrypoint), *extra]

    def run_queue(self, *extra, **env):
        return subprocess.run(self.command(*extra), env=dict(self.env, **env),
                              text=True, capture_output=True, timeout=30)

    def calls(self):
        return [json.loads(x) for x in self.calls_path.read_text().splitlines()] if self.calls_path.exists() else []

    def queue(self):
        return json.loads((self.root / 'queue.json').read_text())

    def test_all_four_smokes_precede_full_runs_and_metrics_are_preserved(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([(x['phase'], x['mode']) for x in calls], [
            ('smoke', 'baseline'), ('smoke', 'full'), ('smoke', 'isotropic'), ('smoke', 'permuted'),
            ('full', 'baseline'), ('full', 'full'), ('full', 'isotropic'), ('full', 'permuted')])
        self.assertEqual([x['cfg']['epochs'] for x in calls], [1, 1, 1, 1, 20, 20, 20, 20])
        self.assertEqual([x['cfg']['ca_epochs'] for x in calls], [1, 1, 1, 1, 5, 5, 5, 5])
        self.assertEqual(len({x['cfg']['output'] for x in calls}), 8)
        summary = json.loads((self.root / 'summary.json').read_text())
        self.assertEqual(len(summary['groups']), 4)
        self.assertEqual(summary['groups'][0]['Average'], 83)
        self.assertEqual(summary['groups'][0]['StageOld'], 78)
        self.assertEqual(len(summary['groups'][0]['tasks']), 10)
        self.assertEqual(summary['groups'][0]['tasks'][1]['ca_drift']['drift_rms'], 0.25)
        self.assertTrue((self.root / 'summary.csv').is_file())
        self.assertEqual(self.queue()['status'], 'completed')

    def test_failed_smoke_pauses_all_training_and_resume_never_retries(self):
        first = self.run_queue(FAKE_FAIL='smoke:full')
        self.assertNotEqual(first.returncode, 0)
        self.assertEqual([x['mode'] for x in self.calls()], ['baseline', 'full'])
        self.assertEqual(self.queue()['status'], 'paused')
        second = self.run_queue('--resume')
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(len(self.calls()), 2)

    def test_failed_full_is_not_retried_after_resume(self):
        first = self.run_queue(FAKE_FAIL='full:baseline')
        self.assertNotEqual(first.returncode, 0)
        self.assertEqual(len(self.calls()), 5)
        second = self.run_queue('--resume')
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(len(self.calls()), 5)

    def test_incomplete_smoke_result_pauses_even_if_entrypoint_exits_zero(self):
        result = self.run_queue(FAKE_INCOMPLETE='smoke:full')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([x['mode'] for x in self.calls()], ['baseline', 'full'])
        self.assertEqual(self.queue()['status'], 'paused')
        second = self.run_queue('--resume')
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(len(self.calls()), 2)

    def test_unreadable_or_nonconsecutive_task_result_never_passes_smoke(self):
        for variable in ('FAKE_BAD_RESULT', 'FAKE_BAD_TASKS'):
            with self.subTest(variable=variable):
                self.root = self.path / variable
                self.env['FAKE_ROOT'] = str(self.root)
                result = self.run_queue(**{variable: 'smoke:full'})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.queue()['status'], 'paused')
        self.assertEqual(len(self.calls()), 4)

    def test_analysis_error_is_recorded_without_stopping_later_groups(self):
        result = self.run_queue(FAKE_BAD_ANALYSIS='full')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 8)
        self.assertTrue(self.queue()['analysis_errors'])
        self.assertTrue((self.root / 'full/D_permuted/result.json').is_file())

    def test_summary_write_error_is_recorded_without_stopping_later_groups(self):
        result = self.run_queue(FAKE_BLOCK_SUMMARY='full')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 8)
        self.assertTrue(self.queue()['analysis_errors'])
        self.assertTrue((self.root / 'full/D_permuted/result.json').is_file())

    def test_resume_uses_recorded_data_path_when_override_is_omitted(self):
        first = self.run_queue('--modes', 'baseline')
        self.assertEqual(first.returncode, 0, first.stderr)
        command = self.command('--resume', '--modes', 'baseline')
        index = command.index('--data-path')
        del command[index:index + 2]
        resumed = subprocess.run(command, env=self.env, text=True, capture_output=True, timeout=30)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(len(self.calls()), 2)

    def test_resume_rebuilds_missing_or_corrupt_summary_from_verified_training_records(self):
        first = self.run_queue(FAKE_BLOCK_SUMMARY='full')
        self.assertEqual(first.returncode, 0, first.stderr)
        blocked = self.root / 'full/B_full/result.json'
        blocked.rmdir()
        corrupt = self.root / 'full/C_isotropic/result.json'
        corrupt.write_text('bad json')
        resumed = self.run_queue('--resume')
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(len(self.calls()), 8)
        self.assertEqual(json.loads(blocked.read_text())['Average'], 83)
        self.assertEqual(json.loads(corrupt.read_text())['Average'], 83)

    def test_resume_rejects_changed_frozen_config_before_starting_any_full(self):
        first = self.run_queue('--hours', '0')
        self.assertEqual(first.returncode, 0, first.stderr)
        frozen = self.root / 'base_config.json'
        cfg = json.loads(frozen.read_text()); cfg['rank'] = 32
        frozen.write_text(json.dumps(cfg))
        second = self.run_queue('--resume', '--hours', '0')
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(len(self.calls()), 4)

    def test_zero_budget_runs_smokes_but_starts_no_full_group(self):
        result = self.run_queue('--hours', '0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([x['phase'] for x in self.calls()], ['smoke'] * 4)
        self.assertEqual(self.queue()['status'], 'budget_exhausted')
        started = self.queue()['first_gpu_started_at']
        again = self.run_queue('--resume')
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(len(self.calls()), 4)
        self.assertEqual(self.queue()['first_gpu_started_at'], started)

    def test_started_full_finishes_after_budget_before_next_group_is_blocked(self):
        result = self.run_queue('--hours', '0.0004', FAKE_FULL_SLEEP='1.6')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([x['phase'] for x in self.calls()], ['smoke'] * 4 + ['full'])
        self.assertEqual(self.queue()['status'], 'budget_exhausted')
        self.assertEqual(json.loads((self.root / 'full/A_baseline/result.json').read_text())['status'], 'completed')

    def test_bash_wrapper_honors_explicit_python_and_non_5090_is_rejected(self):
        result = subprocess.run(['bash', str(BUNDLE / 'scripts/run_ca_drift_5090.sh'),
            *self.command('--modes', 'baseline')[2:]], env=dict(self.env, LODA_PYTHON=sys.executable),
            text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 2)
        rejected = self.run_queue('--root', str(self.path / 'other'), FAKE_GPU_NAME='NVIDIA RTX 3090')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(len(self.calls()), 2)

    def test_two_resumes_cannot_own_the_same_queue_during_gpu_preflight(self):
        self.assertEqual(self.run_queue('--hours', '0').returncode, 0)
        marker = self.path / 'gpu-started'
        first = subprocess.Popen(self.command('--resume'), env=dict(self.env,
            FAKE_GPU_STARTED=str(marker), FAKE_GPU_SLEEP='0.5'),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: first.kill() if first.poll() is None else None)
        deadline = time.time() + 10
        while not marker.exists() and first.poll() is None and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(marker.exists())
        self.assertEqual(json.loads((self.root / 'active.json').read_text())['queue_pid'], first.pid)
        second = self.run_queue('--resume')
        stdout, stderr = first.communicate(timeout=15)
        self.assertEqual(first.returncode, 0, stderr)
        self.assertNotEqual(second.returncode, 0)

    def test_resume_skips_only_matching_completed_source_and_config(self):
        first = self.run_queue('--modes', 'baseline')
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(len(self.calls()), 2)
        second = self.run_queue('--resume', '--modes', 'baseline')
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(len(self.calls()), 2)
        actual = self.root / 'full/A_baseline/training/effective_config.json'
        cfg = json.loads(actual.read_text()); cfg['seed'] = 42
        actual.write_text(json.dumps(cfg))
        mismatch = self.run_queue('--resume', '--modes', 'baseline')
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertEqual(len(self.calls()), 2)

    def test_new_root_and_valid_data_are_required_before_launch(self):
        self.root.mkdir()
        result = self.run_queue()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])
        self.root.rmdir()
        shutil.rmtree(self.data / 'test')
        result = self.run_queue()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_active_pids_are_real_while_child_runs_and_cleared_afterwards(self):
        child = subprocess.Popen(self.command('--modes', 'baseline'),
            env=dict(self.env, FAKE_SLEEP='0.3'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: child.kill() if child.poll() is None else None)
        deadline = time.time() + 10
        active = None
        while time.time() < deadline and child.poll() is None:
            try:
                value = json.loads((self.root / 'active.json').read_text())
                if value.get('child_pid'):
                    active = value
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.01)
        self.assertIsNotNone(active)
        self.assertEqual(active['queue_pid'], child.pid)
        os.kill(active['child_pid'], 0)
        stdout, stderr = child.communicate(timeout=15)
        self.assertEqual(child.returncode, 0, stderr)
        self.assertIsNone(json.loads((self.root / 'active.json').read_text())['child_pid'])


if __name__ == '__main__':
    unittest.main()
