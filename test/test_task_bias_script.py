import json
import os
import shlex
import subprocess
import unittest


class TaskBiasScriptTests(unittest.TestCase):
    def test_pair_uses_one_task0_checkpoint_and_changes_only_bias_switch(self):
        result = subprocess.run(
            ['bash', 'scripts/9_11_task_bias_pair.sh', '--dry-run'],
            capture_output=True,
            text=True,
            env=dict(os.environ, SEEDS='1993'),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = []
        for line in result.stdout.splitlines():
            parts = shlex.split(line)
            if 'main.py' not in parts:
                continue
            args = {}
            for index, part in enumerate(parts):
                if part == '--set':
                    key, value = parts[index + 1].split('=', 1)
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                    args[key] = value
            runs.append(args)

        self.assertEqual(len(runs), 3)
        self.assertEqual([run['max_tasks'] for run in runs], [1, 3, 3])
        self.assertEqual([run['classification_training_mode'] for run in runs], ['task_local'] * 3)
        self.assertEqual([run['dual_mask_task_bias_calibration'] for run in runs], [False, False, True])
        self.assertEqual(runs[0]['task0_checkpoint_save'], runs[1]['task0_checkpoint_resume'])
        self.assertEqual(runs[1]['task0_checkpoint_resume'], runs[2]['task0_checkpoint_resume'])
        self.assertNotIn('data_path', runs[0])
        allowed = {'prefix', 'dual_mask_task_bias_calibration'}
        self.assertEqual({key: value for key, value in runs[1].items() if key not in allowed},
                         {key: value for key, value in runs[2].items() if key not in allowed})


if __name__ == '__main__':
    unittest.main()
