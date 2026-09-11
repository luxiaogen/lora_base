import json
import os
import shlex
import subprocess
import unittest


class OvernightScriptTests(unittest.TestCase):
    scripts = (
        ('scripts/9_12_previous_function_5090_overnight.sh', 'dual_mask_previous_function_enabled'),
        ('scripts/9_12_boundary_calibration_3090_overnight.sh', 'dual_mask_boundary_calibration'),
    )

    @staticmethod
    def parse_runs(script):
        result = subprocess.run(
            ['bash', script, '--dry-run'],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHON_BIN='python'),
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr)
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
        return runs

    def test_each_script_runs_screen_then_three_full_seed_pairs(self):
        for script, switch in self.scripts:
            with self.subTest(script=script):
                runs = self.parse_runs(script)
                self.assertEqual(len(runs), 11)
                self.assertEqual([run['seed'][0] for run in runs],
                                 [1993] * 5 + [1996] * 3 + [1997] * 3)
                self.assertEqual([run['max_tasks'] for run in runs],
                                 [1, 3, 3, 10, 10, 1, 10, 10, 1, 10, 10])
                self.assertEqual([run[switch] for run in runs],
                                 [False, False, True, False, True,
                                  False, False, True, False, False, True])
                self.assertTrue(all('data_path' not in run for run in runs))

    def test_each_pair_shares_task0_and_changes_only_one_method_switch(self):
        ignored = {'prefix', 'wandb_tags'}
        for script, switch in self.scripts:
            with self.subTest(script=script):
                runs = self.parse_runs(script)
                for task0_index, pairs in ((0, ((1, 2), (3, 4))),
                                           (5, ((6, 7),)),
                                           (8, ((9, 10),))):
                    checkpoint = runs[task0_index]['task0_checkpoint_save']
                    for left, right in pairs:
                        self.assertEqual(runs[left]['task0_checkpoint_resume'], checkpoint)
                        self.assertEqual(runs[right]['task0_checkpoint_resume'], checkpoint)
                        baseline = {key: value for key, value in runs[left].items()
                                    if key not in ignored | {switch}}
                        candidate = {key: value for key, value in runs[right].items()
                                     if key not in ignored | {switch}}
                        self.assertEqual(baseline, candidate)


if __name__ == '__main__':
    unittest.main()
