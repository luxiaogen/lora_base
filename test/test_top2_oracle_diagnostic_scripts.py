import shlex
import subprocess
import unittest
from pathlib import Path


class Top2OracleDiagnosticScriptTests(unittest.TestCase):
    def test_two_machine_commands_use_local_config_paths_and_expected_protocols(self):
        root = Path(__file__).resolve().parents[1]
        cases = [
            ('scripts/9_16_top2_oracle_diagnostic_3090.sh', 'exps/dlora/imgr10.json', 'max_tasks=10'),
            ('scripts/9_16_top2_oracle_diagnostic_5090.sh', 'exps/dlora/imgr20.json', 'max_tasks=20'),
        ]
        for relative, config, max_tasks in cases:
            result = subprocess.run(
                ['bash', str(root / relative)],
                cwd=root,
                env={'DRY_RUN': '1', 'PATH': '/usr/bin:/bin'},
                capture_output=True,
                text=True,
                check=True,
            )
            command = shlex.split(result.stdout)
            self.assertEqual(command[:4], ['python', 'main.py', '--config', config])
            overrides = [command[i + 1] for i, value in enumerate(command[:-1]) if value == '--set']
            self.assertIn('seed=[1993]', overrides)
            self.assertIn(max_tasks, overrides)
            self.assertIn('dual_mask_p_conflict_diagnostics=true', overrides)
            self.assertFalse(any(value.startswith('data_path=') for value in overrides))


if __name__ == '__main__':
    unittest.main()
