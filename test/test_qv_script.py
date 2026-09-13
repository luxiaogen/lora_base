import os
import shlex
import subprocess
import unittest


class QVScriptTests(unittest.TestCase):
    def test_dry_run_pairs_same_checkpoint_and_changes_only_projection_switch(self):
        result = subprocess.run(['bash', 'scripts/9_13_qv_after_task0.sh', '--dry-run'],
                                env=dict(os.environ, PYTHON_BIN='python', SEED='1996'),
                                text=True, capture_output=True, check=True)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('python main.py ')]
        self.assertEqual(len(commands), 3)
        configs = []
        for command in commands:
            self.assertEqual(command[command.index('--config') + 1], 'exps/dlora/imgr10.json')
            values = dict(command[i + 1].split('=', 1) for i, item in enumerate(command) if item == '--set')
            self.assertNotIn('data_path', values)
            self.assertEqual(values['seed'], '[1996]')
            self.assertEqual(values['ca_epochs'], '5')
            self.assertEqual(values['total_sessions'], '10')
            configs.append(values)
        save, baseline, candidate = configs
        self.assertEqual(save['max_tasks'], '1')
        self.assertEqual(baseline['max_tasks'], '3')
        self.assertEqual(candidate['max_tasks'], '3')
        self.assertEqual(save['task0_checkpoint_save'], baseline['task0_checkpoint_resume'])
        self.assertEqual(baseline['task0_checkpoint_resume'], candidate['task0_checkpoint_resume'])
        self.assertEqual(candidate['dual_mask_qv_after_task0'], 'true')
        self.assertEqual(baseline['dual_mask_qv_after_task0'], 'false')
        changed = {key for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key)}
        self.assertEqual(changed, {'prefix', 'dual_mask_qv_after_task0'})

    def test_script_syntax(self):
        subprocess.run(['bash', '-n', 'scripts/9_13_qv_after_task0.sh'], check=True)


if __name__ == '__main__':
    unittest.main()
