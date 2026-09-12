import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest


class POnlyScriptTests(unittest.TestCase):
    def test_three_stages_change_only_shared_branch_and_use_one_checkpoint(self):
        script = 'scripts/9_12_p_only_after_task0.sh'
        output = subprocess.check_output(['bash', script, '--dry-run'], text=True,
                                         env=dict(os.environ, PYTHON_BIN='python', DEVICE='0', WANDB_MODE='online'))
        commands = [shlex.split(line) for line in output.splitlines() if line.startswith('python main.py ')]
        self.assertEqual(len(commands), 3)
        settings = []
        for command in commands:
            self.assertEqual(command[command.index('--config') + 1], 'exps/dlora/imgr10.json')
            values = {}
            for idx, token in enumerate(command):
                if token == '--set':
                    key, value = command[idx + 1].split('=', 1)
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                    values[key] = value
            self.assertNotIn('data_path', values)
            self.assertEqual(values['ca_epochs'], 5)
            self.assertEqual(values['seed'], [1993])
            self.assertEqual(values['rank'], 64)
            self.assertEqual(values['dual_mask_private_rank'], 0)
            settings.append(values)
        task0, baseline, candidate = settings
        self.assertEqual(task0['max_tasks'], 1)
        self.assertEqual(baseline['max_tasks'], 3)
        self.assertEqual(task0['task0_checkpoint_save'], baseline['task0_checkpoint_resume'])
        self.assertEqual(baseline['task0_checkpoint_resume'], candidate['task0_checkpoint_resume'])
        differences = {key for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key)}
        self.assertEqual(differences, {'prefix', 'dual_mask_s_task0_only'})
        self.assertFalse(baseline['dual_mask_s_task0_only'])
        self.assertTrue(candidate['dual_mask_s_task0_only'])
        self.assertNotIn('\ncd ', Path(script).read_text())


if __name__ == '__main__':
    unittest.main()
