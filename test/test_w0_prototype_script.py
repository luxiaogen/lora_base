import shlex
import subprocess
import unittest
from pathlib import Path


class W0PrototypeScriptTests(unittest.TestCase):
    def test_dry_run_is_one_fresh_qkv_imagenet_r_seed1993_job(self):
        script = Path(__file__).resolve().parents[1] / (
            'scripts/9_17_imgr10_w0_prototype_consistency_seed1993.sh')
        self.assertTrue(script.exists(), 'W_pre prototype diagnostic script is missing')
        result = subprocess.run(
            ['bash', str(script)],
            env={'DRY_RUN': '1', 'PATH': '/usr/bin:/bin'},
            capture_output=True,
            text=True,
            check=True,
        )
        command = shlex.split(result.stdout)
        self.assertEqual(command[:4], [
            'python', 'main.py', '--config', 'exps/dlora/imgr10.json'])
        overrides = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == '--set'
        ]
        for expected in (
                'seed=[1993]', 'max_tasks=10', 'ca_epochs=5', 'rank=64',
                'dual_mask_p_conflict_diagnostics=true',
                'dual_mask_task0_qk=false', 'dual_mask_qv_after_task0=false',
                'dual_mask_qk_all_tasks=false', 'dual_mask_qv_all_tasks=false',
                'task0_checkpoint_resume=', 'task0_checkpoint_save='):
            self.assertIn(expected, overrides)
        self.assertFalse(any(value.startswith('data_path=') for value in overrides))
        self.assertEqual(command.count('main.py'), 1)


if __name__ == '__main__':
    unittest.main()
