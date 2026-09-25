import ast
import copy
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def grouping_method():
    # Run the actual method without importing pretrained ViT/data dependencies.
    tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
    learner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Learner')
    method = next(n for n in learner.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_lora_optimizer_groups')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<learner-groups>', 'exec'), namespace)
    return namespace['_lora_optimizer_groups']


def network(task=1, wrapped=False):
    net = torch.nn.Module()
    net.S_lora = torch.nn.ModuleList([torch.nn.Linear(4, 3, bias=False) for _ in range(2)])
    net.P_lora = torch.nn.ModuleList([torch.nn.Linear(4, 3, bias=False) for _ in range(2)])
    net.classifier = torch.nn.Linear(3, 2, bias=False)
    for p in net.parameters():
        p.requires_grad_(False)
    net.S_lora[task].weight.requires_grad_(True)
    if task > 0:
        net.P_lora[task].weight.requires_grad_(True)
    net.classifier.weight.requires_grad_(True)
    if wrapped:
        wrapper = torch.nn.Module()
        wrapper.module = net
        net = wrapper
    return net


def legacy_groups(net):
    lora, other = [], []
    for name, p in net.named_parameters():
        if p.requires_grad:
            (lora if 'lora' in name.lower() else other).append(p)
    return [dict(params=ps, lr=.02, momentum=.9, weight_decay=0.) for ps in (lora, other)]


class PLoraLearningRateTests(unittest.TestCase):
    def groups(self, net, task=1, multiplier=None):
        old = legacy_groups(net)
        args = {} if multiplier is None else {'plora_lr_multiplier': multiplier}
        learner = SimpleNamespace(_network=net, _cur_task=task, args=args)
        return grouping_method()(learner, old[0]['params'], old[1]['params'], .02, 0.)

    def test_default_and_explicit_one_preserve_legacy_groups_and_rng(self):
        net = network()
        for multiplier in (None, 1.):
            state = torch.get_rng_state().clone()
            actual, expected = self.groups(net, multiplier=multiplier), legacy_groups(net)
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assertEqual(a.keys(), b.keys())
                self.assertEqual([id(p) for p in a['params']], [id(p) for p in b['params']])
                for key in ('lr', 'momentum', 'weight_decay'):
                    self.assertEqual(a[key], b[key])
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_task0_ignores_multiplier(self):
        net = network(task=0)
        groups = self.groups(net, task=0, multiplier=1.5)
        self.assertEqual(len(groups), 2)
        self.assertEqual([g['lr'] for g in groups], [.02, .02])
        self.assertEqual([id(p) for p in groups[0]['params']],
                         [id(p) for p in legacy_groups(net)[0]['params']])

    def test_only_current_private_parameters_move_to_scaled_group(self):
        for wrapped in (False, True):
            net = network(wrapped=wrapped)
            groups = self.groups(net, multiplier=1.5)
            self.assertEqual([g['lr'] for g in groups], [.02, .02, .03])
            private_ids = {id(p) for n, p in net.named_parameters() if 'P_lora.1.' in n}
            self.assertEqual({id(p) for p in groups[2]['params']}, private_ids)
            selected = [id(p) for g in groups for p in g['params']]
            self.assertEqual(len(selected), len(set(selected)))
            self.assertEqual(set(selected), {id(p) for p in net.parameters() if p.requires_grad})

    def test_disabled_private_branch_keeps_two_groups(self):
        net = network()
        net.P_lora[1].weight.requires_grad_(False)
        self.assertEqual(len(self.groups(net, multiplier=1.5)), 2)

    def test_legacy_sgd_momentum_cosine_trajectory_is_bitwise_equal(self):
        old = network()
        new = copy.deepcopy(old)
        opt_old = torch.optim.SGD(legacy_groups(old))
        opt_new = torch.optim.SGD(self.groups(new))
        schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=20)
                      for o in (opt_old, opt_new)]
        for step in range(20):
            for net, opt, scheduler in zip((old, new), (opt_old, opt_new), schedulers):
                opt.zero_grad(set_to_none=True)
                sum((p.square().sum() * (step + 1)) for p in net.parameters() if p.requires_grad).backward()
                opt.step()
                scheduler.step()
            for a, b in zip(old.parameters(), new.parameters()):
                self.assertTrue(torch.equal(a, b))
            for a, b in zip(opt_old.state.values(), opt_new.state.values()):
                self.assertTrue(torch.equal(a['momentum_buffer'], b['momentum_buffer']))
            self.assertEqual(schedulers[0].get_last_lr(), schedulers[1].get_last_lr())

    def test_candidate_scales_private_step_and_cosine_only(self):
        net = network()
        opt = torch.optim.SGD(self.groups(net, multiplier=1.5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        before = {n: p.detach().clone() for n, p in net.named_parameters()}
        for p in net.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        opt.step()
        for n, p in net.named_parameters():
            expected = (.03 if 'P_lora' in n else .02) if p.requires_grad else 0.
            torch.testing.assert_close(before[n] - p, torch.full_like(p, expected))
        for _ in range(20):
            lrs = scheduler.get_last_lr()
            self.assertAlmostEqual(lrs[2], 1.5 * lrs[0], places=12)
            self.assertEqual(lrs[0], lrs[1])
            opt.step()
            scheduler.step()


class PLoraScriptTests(unittest.TestCase):
    def test_spec_changes_only_multiplier_and_preserves_a_recipe(self):
        old = json.loads((ROOT / 'scripts/sweeps/imgr10_gate_reg_overlap_3090.json').read_text())
        spec = json.loads((ROOT / 'scripts/sweeps/imgr10_plora_lr_3090.json').read_text())
        self.assertEqual(spec['seeds'], [1993])
        for key, value in old['common_overrides'].items():
            if key != 'wandb_group':
                self.assertEqual(spec['common_overrides'][key], value, key)
        self.assertNotIn('data_path', spec['common_overrides'])
        self.assertEqual([v['overrides'] for v in spec['variants']],
                         [{'plora_lr_multiplier': 1.}, {'plora_lr_multiplier': 1.5}])

    def test_shell_matches_spec_from_outside_repo(self):
        spec = json.loads((ROOT / 'scripts/sweeps/imgr10_plora_lr_3090.json').read_text())
        script = ROOT / 'scripts/9_25_imgr10_plora_lr_3090.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        output = subprocess.check_output(['bash', str(script), '--dry-run'], cwd='/tmp', text=True)
        commands = output.replace('\\\n', '').splitlines()
        self.assertEqual(len(commands), 2)
        for line, variant in zip(commands, spec['variants']):
            tokens = shlex.split(line)
            settings = dict(tokens[i + 1].split('=', 1) for i, t in enumerate(tokens) if t == '--set')
            self.assertEqual(json.loads(settings['seed']), [1993])
            self.assertNotIn('data_path', settings)
            self.assertIn('${TIMESTAMP}', settings['prefix'])
            self.assertIn('$@', tokens)
            for key, value in {**spec['common_overrides'], **variant['overrides']}.items():
                try:
                    actual = json.loads(settings[key])
                except json.JSONDecodeError:
                    actual = settings[key]
                self.assertEqual(actual, value, key)


if __name__ == '__main__':
    unittest.main()
