import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest


class LoRAInheritanceScriptTests(unittest.TestCase):
    def test_launchers_match_previous_protocol_and_split_branches(self):
        root = Path(__file__).resolve().parents[1]
        for machine, count in (('3090', 9), ('5090', 18)):
            with self.subTest(machine=machine):
                result = subprocess.run(
                    ['bash', f'scripts/9_14_lora_inherit_branches_{machine}.sh'],
                    cwd=root, env=dict(os.environ, DRY_RUN='1'), text=True, capture_output=True, check=True,
                )
                commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('python main.py ')]
                self.assertEqual(len(commands), count)
                spec = json.loads((root / f'scripts/sweeps/lora_inherit_{machine}.json').read_text())
                expected = dict(spec['common_overrides'], wandb_group='lora_inherit_branches_t10')
                for dataset in spec['datasets']:
                    matching = [cmd for cmd in commands if cmd[cmd.index('--config') + 1] == dataset['config']]
                    self.assertEqual(len(matching), 9)
                    for seed_index, seed in enumerate((1993, 1996, 1997)):
                        for i, (enabled, branches, name) in enumerate(((False, 'both', 'fresh'), (True, 's', 's_only'), (True, 'p', 'p_only'))):
                            cmd = matching[seed_index * 3 + i]
                            overrides = {}
                            for j, token in enumerate(cmd):
                                if token == '--set':
                                    key, value = cmd[j + 1].split('=', 1)
                                    self.assertNotIn(key, overrides)
                                    try:
                                        value = json.loads(value)
                                    except json.JSONDecodeError:
                                        pass
                                    overrides[key] = value
                            prefix = overrides.pop('prefix')
                            self.assertTrue(prefix.startswith(f"{dataset['name']}_{name}_seed{seed}_"))
                            self.assertEqual(overrides, dict(expected, **dataset['overrides'], seed=[seed],
                                dual_mask_lora_inherit=enabled, dual_mask_lora_inherit_branches=branches))

    def test_unknown_dataset_fails_before_training(self):
        result = subprocess.run(['bash', 'scripts/9_14_lora_inherit_branches.sh', 'typo'],
            cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, DRY_RUN='1'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Unknown dataset', result.stderr)


if __name__ == '__main__':
    unittest.main()
