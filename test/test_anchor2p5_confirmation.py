import json
from pathlib import Path
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNS = [('imgr10_anchor2p5_t10_3090', 10), ('imgr10_anchor2p5_t3_5090', 3)]


class AnchorConfirmationTests(unittest.TestCase):
    def test_recipes_match_frozen_baseline_and_single_candidate(self):
        baseline = json.loads((ROOT / 'docs/experiments/baselines/imgr10_B1_3090.json').read_text())
        for name, tasks in RUNS:
            spec = json.loads((ROOT / f'scripts/sweeps/{name}.json').read_text())
            self.assertEqual(spec['seeds'], [1993])
            self.assertEqual(spec['datasets'], [{'name': name, 'config': 'exps/dlora/imgr10.json'}])
            self.assertEqual(len(spec['variants']), 1)
            self.assertEqual(spec['variants'][0]['overrides'], {'dual_mask_anchor_reg_weight': 2.5})
            actual = {**spec['common_overrides'], **spec['variants'][0]['overrides'], 'seed': [1993]}
            actual.pop('wandb_group')
            self.assertEqual(actual, {**baseline, 'max_tasks': tasks})
            self.assertNotIn('data_path', actual)
            self.assertNotIn('device', actual)

    def test_scripts_from_outside_repo_formal_and_smoke(self):
        for name, tasks in RUNS:
            script = ROOT / f'scripts/9_28_{name}.sh'
            subprocess.run(['bash', '-n', str(script)], check=True)
            for smoke in (False, True):
                args = ['--smoke'] if smoke else []
                output = subprocess.check_output(['bash', str(script), '--dry-run', *args], cwd='/tmp', text=True)
                commands = [line for line in output.splitlines() if 'main.py --config' in line]
                self.assertEqual(len(commands), 1)
                tokens = shlex.split(commands[0])
                c = dict(tokens[i+1].split('=', 1) for i,t in enumerate(tokens) if t == '--set')
                self.assertEqual(c['seed'], '[1993]')
                self.assertEqual(c['max_tasks'], str(2 if smoke else tasks))
                self.assertEqual(c['total_sessions'], '10')
                self.assertEqual(c['dual_mask_anchor_reg_weight'], '2.5')
                self.assertEqual(c['slora_lr_multiplier'], '1')
                self.assertEqual(c['plora_lr_multiplier'], '1')
                self.assertEqual(c['init_epoch'], '1' if smoke else '20')
                self.assertEqual(c['epochs'], '1' if smoke else '20')
                self.assertEqual(c['ca_epochs'], '1' if smoke else '5')
                self.assertEqual(c['incremental_holdout'], 'false')
                self.assertEqual(c['task0_validation_enabled'], 'false')
                self.assertEqual(c['disable_fused_sdpa'], 'true')
                self.assertNotIn('data_path', c)
                self.assertNotIn('device', c)


if __name__ == '__main__':
    unittest.main()
