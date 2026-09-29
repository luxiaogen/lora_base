import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class TwoCenterQueueTests(unittest.TestCase):
    def execute(self, gpu, mode=None, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            script = root / 'scripts' / f'9_29_imgr10_ca_two_centers_{gpu}.sh'
            shutil.copyfile(Path('scripts') / script.name, script)
            capture = root / 'calls.jsonl'
            stub = root / 'python'
            stub.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
settings = dict(arg.split('=', 1) for i, arg in enumerate(sys.argv) if i and sys.argv[i-1] == '--set')
with open(os.environ['RUN_CAPTURE'], 'a') as stream:
    stream.write(json.dumps(settings) + '\\n')
sys.exit(9 if os.environ.get('FAIL_RUN') else 0)
''')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       RUN_CAPTURE=str(capture))
            if fail:
                env['FAIL_RUN'] = '1'
            result = subprocess.run(['bash', str(script)] + ([mode] if mode else []),
                                    env=env, capture_output=True, text=True)
            calls = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
            return result, calls

    def test_each_machine_runs_smoke_then_one_full_candidate_without_weights(self):
        for gpu in ('3090', '5090'):
            result, calls = self.execute(gpu)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([c['max_tasks'] for c in calls], ['2', '10'])
            self.assertNotEqual(calls[0]['prefix'], calls[1]['prefix'])
            self.assertEqual([c['ca_epochs'] for c in calls], ['1', '5'])
            self.assertEqual([c['init_epoch'] for c in calls], ['1', '20'])
            spec = json.loads(Path(f'scripts/sweeps/imgr10_ca_two_centers_{gpu}.json').read_text())
            base = json.loads(Path(f'scripts/sweeps/imgr10_pair_separation_{gpu}.json').read_text())
            self.assertEqual(len(spec['variants']), 1)
            for c in calls:
                self.assertEqual(c['ca_two_centers'], 'true')
                self.assertEqual(c['save_task_weights'], 'false')
                self.assertEqual(float(c['dual_mask_anchor_reg_weight']), 2.5 if gpu == '3090' else 5)
                self.assertEqual(c['seed'], '[1993]')
                self.assertNotIn('data_path', c)
                self.assertNotIn('device', c)
            expected = dict(base['common_overrides'], ca_two_centers=True,
                            wandb_group=f'imgr10_ca_two_centers_{gpu}')
            actual = {**spec['common_overrides'], **spec['variants'][0]['overrides']}
            self.assertEqual(actual, expected)
            for k, v in expected.items():
                self.assertEqual(calls[1][k], v if isinstance(v, str) else json.dumps(v, separators=(',', ':')))

    def test_failed_smoke_stops_full_and_failed_full_is_reported(self):
        result, calls = self.execute('3090', fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([c['max_tasks'] for c in calls], ['2'])
        result, calls = self.execute('5090', '--full', fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([c['max_tasks'] for c in calls], ['10'])

    def test_dry_run_and_invalid_mode_do_not_start_training(self):
        for mode, code in (('--dry-run', 0), ('--dryrun', 2)):
            result, calls = self.execute('5090', mode)
            self.assertEqual(result.returncode, code)
            self.assertEqual(calls, [])
        self.assertEqual(self.execute('3090', '--dry-run')[0].stdout.count('python main.py'), 1)


if __name__ == '__main__':
    unittest.main()
