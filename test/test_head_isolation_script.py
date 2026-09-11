import json
import os
import shlex
import subprocess
import unittest


class HeadIsolationScriptTests(unittest.TestCase):
    def test_both_variants_have_valid_device_and_same_paired_checkpoint(self):
        for machine, mode in [('3090', 'task_local_head'), ('5090', 'task_local_head_replay')]:
            result = subprocess.run(['bash', 'scripts/9_11_head_isolation_pair.sh', '--dry-run', machine,
                                     '/data with spaces/imagenet-r'], check=True, capture_output=True, text=True,
                                    env=dict(os.environ, SEEDS='1993'))
            runs = []
            for line in result.stdout.splitlines():
                parts = shlex.split(line)
                if 'main.py' not in parts:
                    continue
                args = {}
                for i, part in enumerate(parts):
                    if part == '--set':
                        key, value = parts[i + 1].split('=', 1)
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                        args[key] = value
                runs.append(args)
            self.assertEqual(len(runs), 3)
            for args in runs:
                self.assertIsInstance(args['device'], str)  # trainer.train calls split(',').
                self.assertEqual(args['seed'], [1993])
                self.assertEqual(args['total_sessions'], 10)
                self.assertEqual(args['ca_epochs'], 5)
                self.assertEqual(args['data_path'], '/data with spaces/imagenet-r')
            self.assertEqual(runs[0]['max_tasks'], 1)
            self.assertEqual(runs[1]['max_tasks'], 3)
            self.assertEqual(runs[2]['max_tasks'], 3)
            self.assertEqual(runs[0]['task0_checkpoint_save'], runs[1]['task0_checkpoint_resume'])
            self.assertEqual(runs[1]['task0_checkpoint_resume'], runs[2]['task0_checkpoint_resume'])
            self.assertEqual(runs[1]['classification_training_mode'], 'task_local')
            self.assertEqual(runs[2]['classification_training_mode'], mode)
            allowed = {'prefix', 'classification_training_mode'}
            self.assertEqual({k: v for k, v in runs[1].items() if k not in allowed},
                             {k: v for k, v in runs[2].items() if k not in allowed})

    def test_multiple_seeds_generate_independent_pairs(self):
        result = subprocess.run(['bash', 'scripts/9_11_head_isolation_pair.sh', '--dry-run', '5090', '/data/imgr'],
                                check=True, capture_output=True, text=True, env=dict(os.environ, SEEDS='1993 1996 1997'))
        self.assertEqual(result.stdout.count('Starting '), 9)
        for seed in (1993, 1996, 1997):
            self.assertEqual(result.stdout.count(f'/task0_seed{seed}.pt'), 3)


if __name__ == '__main__':
    unittest.main()
