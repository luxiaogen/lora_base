import json
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import torch

from test.test_plora_lr import ROOT, grouping_method, legacy_groups, network


class SLoraLearningRateTests(unittest.TestCase):
    def groups(self, net, task=1, shared=.5, private=1.):
        old = legacy_groups(net)
        learner = SimpleNamespace(_network=net, _cur_task=task, args={
            'slora_lr_multiplier': shared, 'plora_lr_multiplier': private})
        return grouping_method()(learner, old[0]['params'], old[1]['params'], .02, 0.)

    def test_task0_and_default_preserve_groups_and_rng(self):
        for task, multiplier in ((0, .5), (1, 1.)):
            net = network(task=task)
            state = torch.get_rng_state().clone()
            actual = self.groups(net, task=task, shared=multiplier)
            expected = legacy_groups(net)
            self.assertEqual(len(actual), 2)
            for a, b in zip(actual, expected):
                self.assertEqual([id(p) for p in a['params']], [id(p) for p in b['params']])
                self.assertEqual(a['lr'], b['lr'])
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_only_current_shared_parameters_are_scaled(self):
        for wrapped in (False, True):
            for private in (1., 1.5):
                net = network(wrapped=wrapped)
                groups = self.groups(net, private=private)
                by_id = {id(p): g['lr'] for g in groups for p in g['params']}
                selected = [id(p) for g in groups for p in g['params']]
                self.assertEqual(len(selected), len(set(selected)))
                self.assertEqual(set(selected), {id(p) for p in net.parameters() if p.requires_grad})
                for name, p in net.named_parameters():
                    if p.requires_grad:
                        expected = .01 if 'S_lora' in name else (.02 * private if 'P_lora' in name else .02)
                        self.assertEqual(by_id[id(p)], expected)

    def test_disabled_shared_branch_keeps_legacy_groups(self):
        net = network()
        net.S_lora[1].weight.requires_grad_(False)
        self.assertEqual(len(self.groups(net)), 2)

    def test_sgd_step_and_cosine_scale_shared_only(self):
        net = network()
        opt = torch.optim.SGD(self.groups(net))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        before = {n: p.detach().clone() for n, p in net.named_parameters()}
        for p in net.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        opt.step()
        for name, p in net.named_parameters():
            expected = (.01 if 'S_lora' in name else .02) if p.requires_grad else 0.
            torch.testing.assert_close(before[name] - p, torch.full_like(p, expected))
        for _ in range(20):
            lrs = scheduler.get_last_lr()
            self.assertAlmostEqual(lrs[-1], .5 * lrs[0], places=12)
            self.assertEqual(lrs[0], lrs[1])
            opt.step()
            scheduler.step()


class SLoraQueueTests(unittest.TestCase):
    def test_single_fulltrain_candidate_and_dry_run(self):
        spec = json.loads((ROOT / 'scripts/sweeps/imgr10_slora_lr_t3_5090.json').read_text())
        self.assertEqual(spec['seeds'], [1993])
        self.assertEqual([v['overrides'] for v in spec['variants']], [{'slora_lr_multiplier': .5}])
        common = spec['common_overrides']
        reference = json.loads((ROOT / 'scripts/sweeps/imgr10_head_balance_5090.json').read_text())
        for key, value in reference['common_overrides'].items():
            if key not in ('max_tasks', 'wandb_group'):
                self.assertEqual(common[key], value, key)
        for key, value in dict(max_tasks=3, total_sessions=10, dual_mask_anchor_reg_weight=5,
                               init_epoch=20, epochs=20, ca_epochs=5, plora_lr_multiplier=1,
                               incremental_holdout=False, task0_validation_enabled=False,
                               disable_fused_sdpa=True).items():
            self.assertEqual(common[key], value, key)
        self.assertNotIn('data_path', common)
        script = ROOT / 'scripts/9_28_imgr10_slora_lr_t3_5090.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        output = subprocess.check_output(['bash', str(script), '--dry-run'], cwd='/tmp', text=True)
        commands = [line for line in output.splitlines() if 'main.py --config ' in line]
        self.assertEqual(len(commands), 1)
        tokens = shlex.split(commands[0])
        settings = dict(tokens[i + 1].split('=', 1) for i, t in enumerate(tokens) if t == '--set')
        for key, value in {**common, **spec['variants'][0]['overrides'], 'seed': [1993]}.items():
            actual = settings[key] if isinstance(value, str) else json.loads(settings[key])
            self.assertEqual(actual, value, key)


if __name__ == '__main__':
    unittest.main()
