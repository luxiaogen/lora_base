import json
import os
import shlex
import subprocess
import unittest


class BoundaryRealCurrentScriptTests(unittest.TestCase):
    def test_pair_shares_task0_and_changes_only_candidate_switches(self):
        result = subprocess.run(
            ['bash', 'scripts/9_12_boundary_real_current_screen.sh', '--dry-run'],
            capture_output=True,
            text=True,
            env=dict(os.environ, SEED='1993'),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = []
        for line in result.stdout.splitlines():
            parts = shlex.split(line)
            if 'main.py' not in parts:
                continue
            settings = {}
            for index, part in enumerate(parts):
                if part != '--set':
                    continue
                key, value = parts[index + 1].split('=', 1)
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
                settings[key] = value
            runs.append(settings)

        self.assertEqual(len(runs), 3)
        self.assertEqual([run['max_tasks'] for run in runs], [1, 3, 3])
        self.assertEqual([run['dual_mask_boundary_calibration'] for run in runs],
                         [False, False, True])
        self.assertEqual([run['dual_mask_boundary_real_current'] for run in runs],
                         [False, False, True])
        self.assertEqual(runs[0]['task0_checkpoint_save'], runs[1]['task0_checkpoint_resume'])
        self.assertEqual(runs[1]['task0_checkpoint_resume'], runs[2]['task0_checkpoint_resume'])
        self.assertNotIn('data_path', runs[0])
        ignored = {
            'prefix',
            'wandb_tags',
            'dual_mask_boundary_calibration',
            'dual_mask_boundary_real_current',
        }
        baseline = {key: value for key, value in runs[1].items() if key not in ignored}
        candidate = {key: value for key, value in runs[2].items() if key not in ignored}
        self.assertEqual(baseline, candidate)


if __name__ == '__main__':
    unittest.main()
