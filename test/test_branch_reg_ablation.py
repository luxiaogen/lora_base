import json
from pathlib import Path
import shlex
import subprocess
import unittest

import torch
from test import test_mask_reg_ablation as helpers

ROOT = Path(__file__).resolve().parents[1]


class BranchRegTests(unittest.TestCase):
    def test_fixed_denominator_and_branch_gradients(self):
        learner, module = helpers.MaskRegTests().make(.01)
        s, p = module.S_lora[1].B_weight, module.P_lora[1].B_weight
        # Distinct branch losses make accidental renormalization observable.
        module._joint_conflict_regularization = lambda unit, isolated: unit.B_weight.square().sum() * (3 if isolated else 2)
        for enabled_s, enabled_p in ((True, True), (True, False), (False, True), (False, False)):
            learner.args.update(dual_mask_s_reg_enabled=enabled_s, dual_mask_p_reg_enabled=enabled_p)
            actual = learner._extra_training_loss()
            expected = .01 * (2 * s.square().sum() * enabled_s + 3 * p.square().sum() * enabled_p) / 2
            self.assertTrue(torch.equal(actual, expected))
            grads = torch.autograd.grad(actual, (s, p))
            self.assertTrue(torch.allclose(grads[0], .02 * s * enabled_s))
            self.assertTrue(torch.allclose(grads[1], .03 * p * enabled_p))

    def test_default_matches_legacy_and_anchor_is_independent(self):
        learner, module = helpers.MaskRegTests().make(.01)
        expected = .01 * torch.stack([
            module._joint_conflict_regularization(module.S_lora[1], isolated=False),
            module._joint_conflict_regularization(module.P_lora[1], isolated=True),
        ]).mean()
        self.assertTrue(torch.equal(learner._extra_training_loss(), expected))
        learner, module = helpers.MaskRegTests().make(.01, task=0)
        learner.args.update(dual_mask_anchor_reg_enabled=True, dual_mask_anchor_reg_weight=10.,
                            dual_mask_anchor_reg_task0_only=True)
        module.anchor_regularization = lambda: module.S_lora[0].B_weight.square().mean()
        expected = learner._extra_training_loss()
        learner.args.update(dual_mask_s_reg_enabled=False, dual_mask_p_reg_enabled=False)
        self.assertTrue(torch.equal(learner._extra_training_loss(), expected))

    def test_switches_do_not_change_safe_delta(self):
        learner, module = helpers.MaskRegTests().make(.01)
        delta = module.P_lora[1].B_weight @ module.P_lora[1].A_weight
        expected = module._safe_delta(delta, isolated=True).detach().clone()
        learner.args.update(dual_mask_s_reg_enabled=False, dual_mask_p_reg_enabled=False)
        learner._extra_training_loss()
        self.assertTrue(torch.equal(module._safe_delta(delta, isolated=True), expected))

    def test_launcher_matches_spec_and_existing_recipe(self):
        spec = json.loads((ROOT / 'scripts/sweeps/imgr10_branch_reg_5090.json').read_text())
        previous = json.loads((ROOT / 'scripts/sweeps/imgr10_mask_reg_3090.json').read_text())
        for k, v in previous['common_overrides'].items():
            if k != 'wandb_group':
                self.assertEqual(spec['common_overrides'][k], v, k)
        self.assertEqual(spec['seeds'], [1993])
        self.assertEqual([v['overrides'] for v in spec['variants']], [
            {'dual_mask_s_reg_enabled': True, 'dual_mask_p_reg_enabled': True},
            {'dual_mask_s_reg_enabled': True, 'dual_mask_p_reg_enabled': False},
            {'dual_mask_s_reg_enabled': False, 'dual_mask_p_reg_enabled': True},
        ])
        script = ROOT / 'scripts/9_25_imgr10_branch_reg_5090.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        output = subprocess.check_output(['bash', str(script), '--dry-run'], cwd='/tmp', text=True)
        commands = output.replace('\\\n', '').splitlines()
        self.assertEqual(len(commands), 3)
        for command, variant in zip(commands, spec['variants']):
            tokens = shlex.split(command)
            settings = dict(tokens[i+1].split('=', 1) for i, t in enumerate(tokens) if t == '--set')
            self.assertNotIn('data_path', settings)
            self.assertIn('${TIMESTAMP}', settings['prefix'])
            self.assertIn('$@', tokens)
            for k, expected in {**spec['common_overrides'], **variant['overrides']}.items():
                try:
                    actual = json.loads(settings[k])
                except json.JSONDecodeError:
                    actual = settings[k]
                self.assertEqual(actual, expected, k)


if __name__ == '__main__':
    unittest.main()
