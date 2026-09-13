import os
import shlex
import subprocess
import unittest


class QVScriptTests(unittest.TestCase):
    def test_dry_run_starts_both_runs_from_task0_and_changes_only_projection_switch(self):
        env = dict(os.environ, PYTHON_BIN='python', SEED='1996')
        env.pop('TASK0_CHECKPOINT', None)
        env.pop('MAX_TASKS', None)
        env.pop('TASK0_QK', None)
        result = subprocess.run(['bash', 'scripts/9_13_qv_after_task0.sh', '--dry-run'],
                                env=env,
                                text=True, capture_output=True, check=True)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('python main.py ')]
        self.assertEqual(len(commands), 2)
        configs = []
        for command in commands:
            self.assertEqual(command[command.index('--config') + 1], 'exps/dlora/imgr10.json')
            values = dict(command[i + 1].split('=', 1) for i, item in enumerate(command) if item == '--set')
            self.assertNotIn('data_path', values)
            self.assertNotIn('task0_checkpoint_save', values)
            self.assertNotIn('task0_checkpoint_resume', values)
            self.assertNotIn('dual_mask_task0_qk', values)
            self.assertEqual(values['seed'], '[1996]')
            self.assertEqual(values['ca_epochs'], '5')
            self.assertEqual(values['total_sessions'], '10')
            self.assertEqual(values['max_tasks'], '3')
            configs.append(values)
        baseline, candidate = configs
        self.assertEqual(candidate['dual_mask_qv_after_task0'], 'true')
        self.assertEqual(baseline['dual_mask_qv_after_task0'], 'false')
        changed = {key for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key)}
        self.assertEqual(changed, {'prefix', 'dual_mask_qv_after_task0'})

    def test_script_syntax(self):
        subprocess.run(['bash', '-n', 'scripts/9_13_qv_after_task0.sh'], check=True)
        subprocess.run(['bash', '-n', 'scripts/9_13_qv_after_task0_t10.sh'], check=True)

    def test_full_run_starts_two_independent_qkv_task0_runs(self):
        env = dict(os.environ, PYTHON_BIN='python', SEED='1993')
        env.pop('TASK0_CHECKPOINT', None)
        env.pop('TASK0_QK', None)
        result = subprocess.run(['bash', 'scripts/9_13_qv_after_task0_t10.sh', '--dry-run'],
                                env=env,
                                text=True, capture_output=True, check=True)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('python main.py ')]
        self.assertEqual(len(commands), 2)
        configs = [dict(command[i + 1].split('=', 1) for i, item in enumerate(command) if item == '--set')
                   for command in commands]
        baseline, candidate = configs
        for config in configs:
            self.assertNotIn('task0_checkpoint_save', config)
            self.assertNotIn('task0_checkpoint_resume', config)
            self.assertNotIn('dual_mask_task0_qk', config)
            self.assertEqual(config['max_tasks'], '10')
            self.assertEqual(config['ca_epochs'], '5')
            self.assertEqual(config['wandb_tags'], 'imgr10,qv_after_task0,ca5,full_t10')
        self.assertEqual(baseline['dual_mask_qv_after_task0'], 'false')
        self.assertEqual(candidate['dual_mask_qv_after_task0'], 'true')
        changed = {key for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key)}
        self.assertEqual(changed, {'prefix', 'dual_mask_qv_after_task0'})


if __name__ == '__main__':
    unittest.main()
