import shlex
import subprocess
import unittest
from pathlib import Path


class Top2CounterfactualScriptTests(unittest.TestCase):
    def test_single_seed_qkv_t10_command_uses_config_dataset_path(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/9_16_imgr10_top2_counterfactual_3090.sh'
        result = subprocess.run(
            ['bash', str(script)],
            env={'DRY_RUN': '1', 'PATH': '/usr/bin:/bin'},
            capture_output=True,
            text=True,
            check=True,
        )
        command = shlex.split(result.stdout)
        self.assertEqual(command[:4], ['python', 'main.py', '--config', 'exps/dlora/imgr10.json'])
        overrides = [command[i + 1] for i, value in enumerate(command[:-1]) if value == '--set']
        self.assertIn('seed=[1993]', overrides)
        self.assertIn('max_tasks=10', overrides)
        self.assertIn('ca_epochs=5', overrides)
        self.assertIn('dual_mask_p_conflict_diagnostics=true', overrides)
        self.assertIn('dual_mask_qk_all_tasks=false', overrides)
        self.assertIn('dual_mask_qv_all_tasks=false', overrides)
        self.assertFalse(any(value.startswith('data_path=') for value in overrides))
        self.assertEqual(command.count('main.py'), 1)


if __name__ == '__main__':
    unittest.main()
