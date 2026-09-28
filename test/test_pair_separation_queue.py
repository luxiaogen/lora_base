"""Execute the real queues with a recording training-process substitute."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class QueueTests(unittest.TestCase):
    def execute(self, gpu, mode, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            script = root / 'scripts' / f'9_29_imgr10_pair_separation_{gpu}.sh'
            shutil.copyfile(Path('scripts') / script.name, script)
            capture = root / 'calls.jsonl'
            stub = root / 'python'
            stub.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
settings = dict(arg.split('=', 1) for i, arg in enumerate(sys.argv) if i and sys.argv[i-1] == '--set')
with open(os.environ['RUN_CAPTURE'], 'a') as stream:
    stream.write(json.dumps(settings) + '\\n')
sys.exit(9 if os.environ.get('FAIL_FIRST') and 'p_w005' in settings['prefix'] else 0)
''')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       RUN_CAPTURE=str(capture))
            if fail:
                env['FAIL_FIRST'] = '1'
            result = subprocess.run(['bash', str(script)] + ([mode] if mode else []),
                                    env=env, capture_output=True, text=True)
            calls = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
            return result, calls

    def test_full_queues_preserve_machine_recipe_and_order(self):
        expected = {
            '3090': [('p', .05, 10), ('sp', .05, 10), ('p', 0, 10), ('p', .025, 10), ('sp', .025, 10)],
            '5090': [('p', .05, 3), ('sp', .025, 3), ('sp', .05, 10)],
        }
        for gpu, wanted in expected.items():
            result, calls = self.execute(gpu, '--full')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([(c['pair_separation_scope'], float(c['pair_separation_weight']),
                               int(c['max_tasks'])) for c in calls], wanted)
            for c in calls:
                self.assertEqual(float(c['dual_mask_anchor_reg_weight']), 2.5 if gpu == '3090' else 5)
                self.assertEqual(json.loads(c['seed']), [1993])
                self.assertNotIn('data_path', c)
                self.assertEqual(c['save_task_weights'], 'false')
                self.assertEqual(c['ca_epochs'], '5')
                self.assertEqual(c['init_epoch'], '20')
                self.assertEqual(c['disable_fused_sdpa'], 'true')
            self.assertEqual([c['plora_a_init_mode'] for c in calls],
                             ['off', 'off', 'random_orthogonal', 'off', 'off'] if gpu == '3090' else ['off'] * 3)

    def test_smoke_covers_task1_and_uses_distinct_prefix(self):
        result, smoke = self.execute('5090', '--smoke')
        self.assertEqual(result.returncode, 0, result.stderr)
        for c in smoke:
            self.assertEqual(c['max_tasks'], '2')
            self.assertEqual(c['epochs'], '1')
            self.assertEqual(c['init_epoch'], '1')
            self.assertEqual(c['ca_epochs'], '1')
            self.assertEqual(c['wandb_mode'], 'offline')
            self.assertIn('smoke', c['prefix'])

    def test_default_runs_smoke_then_all_full_experiments(self):
        result, calls = self.execute('5090', None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([int(c['max_tasks']) for c in calls], [2, 2, 2, 3, 3, 10])
        self.assertEqual(len({c['prefix'] for c in calls}), 6)

    def test_failed_full_run_does_not_cancel_later_candidates(self):
        result, calls = self.execute('3090', '--full', fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 5)

    def test_dry_run_does_not_start_training(self):
        result, calls = self.execute('3090', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])
        self.assertEqual(result.stdout.count('python main.py'), 5)

    def test_failed_smoke_does_not_launch_full_training(self):
        result, calls = self.execute('5090', None, fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([int(c['max_tasks']) for c in calls], [2, 2, 2])

    def test_misspelled_mode_cannot_accidentally_start_training(self):
        result, calls = self.execute('3090', '--dryrun')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
