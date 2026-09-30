import ast
import copy
import json
import logging
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from test import test_p_conflict_freeze


ROOT = Path(__file__).resolve().parents[1]


def epoch_hook():
    tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_set_p_conflict_strength_epoch')
    namespace = {'logging': logging}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<epoch-hook>', 'exec'), namespace)
    return namespace[method.name]


class PConflictStrengthWarmupTests(unittest.TestCase):
    def fixture(self, enabled=True, task=1):
        attention = test_p_conflict_freeze.PConflictFreezeTests().make_attention(freeze=0, task=task)
        learner = SimpleNamespace(args={'p_conflict_strength_warmup': enabled}, _cur_task=task,
                                  _iter_lora_modules=lambda: iter([attention]))
        return learner, attention

    def test_epoch_schedule_and_one_based_real_loop_hook(self):
        learner, attention = self.fixture()
        for epoch, expected in enumerate([.5] * 5 + [.6, .7, .8, .9, 1.] + [1.] * 10, 1):
            epoch_hook()(learner, epoch)
            self.assertAlmostEqual(attention.p_conflict_strength_scale, expected)
        tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
        train = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                     and node.name == 'train_function')
        loop = next(node for node in train.body if isinstance(node, ast.For)
                    and any(isinstance(child, ast.Name) and child.id == 'epoch'
                            for child in ast.walk(node.target)))
        calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == '_set_p_conflict_strength_epoch']
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[0]), 'epoch + 1')
        batch_loop = next(node for node in loop.body if isinstance(node, ast.For))
        self.assertLess(calls[0].lineno, batch_loop.lineno)

    def test_only_p_training_forward_changes_not_masks_s_or_regularization(self):
        learner, attention = self.fixture()
        x = torch.randn(7, 4)
        p, s = attention.P_lora[1], attention.S_lora[1]
        raw = p.B_weight @ p.A_weight
        _, _, mask = attention._safe_delta(raw, True, return_details=True)
        original_s = attention._masked_unit_forward(x, s, False).detach().clone()
        reg = attention._joint_conflict_regularization(p, True).detach().clone()
        epoch_hook()(learner, 1)
        strength = attention._conflict_parameters()[1] * .5
        expected = F.linear(x, raw * (1 - attention.general_mask) * (1 - strength * mask))
        torch.testing.assert_close(attention._masked_unit_forward(x, p, True), expected)
        self.assertTrue(torch.equal(original_s, attention._masked_unit_forward(x, s, False)))
        self.assertTrue(torch.equal(mask, attention._safe_delta(raw, True, return_details=True)[2]))
        self.assertTrue(torch.equal(reg, attention._joint_conflict_regularization(p, True)))
        before = [unit.B_weight.detach().clone() for unit in (s, p)]
        optimizer = torch.optim.SGD([s.B_weight, p.B_weight], lr=.02)
        attention._contrib_from_units(x, 1).square().mean().backward()
        optimizer.step()
        for unit, initial in zip((s, p), before):
            self.assertIsNotNone(unit.B_weight.grad)
            self.assertFalse(torch.equal(initial, unit.B_weight))
            self.assertFalse(unit.A_weight.requires_grad)

    def test_disabled_task0_and_full_strength_preserve_weights_outputs_gradients_rng(self):
        for enabled, task, epoch in ((False, 1, 1), (True, 0, 1), (True, 1, 10), (True, 1, 20)):
            learner, attention = self.fixture(enabled, task)
            x = torch.randn(7, 4)
            before = attention._contrib_from_units(x, task)
            params = [p for p in attention.parameters() if p.requires_grad]
            grads = torch.autograd.grad(before.square().sum(), params, allow_unused=True)
            state = copy.deepcopy(attention.state_dict())
            rng = torch.get_rng_state().clone()
            epoch_hook()(learner, epoch)
            after = attention._contrib_from_units(x, task)
            other_grads = torch.autograd.grad(after.square().sum(), params, allow_unused=True)
            self.assertTrue(torch.equal(before, after))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(state[k], v) for k, v in attention.state_dict().items()))
            for a, b in zip(grads, other_grads):
                self.assertTrue(b is None if a is None else torch.equal(a, b))

    def test_eval_and_merge_always_use_original_strength(self):
        for epoch in (1, 6, 11, 20):
            learner, attention = self.fixture()
            reference = copy.deepcopy(attention)
            x = torch.randn(7, 4)
            epoch_hook()(learner, epoch)
            attention.eval()
            reference.eval()
            pre = F.linear(x, attention.qkv.weight, attention.qkv.bias) + attention._contrib_from_units(x, 1)
            expected = F.linear(x, reference.qkv.weight, reference.qkv.bias) + reference._contrib_from_units(x, 1)
            self.assertTrue(torch.equal(pre, expected))
            attention.train()
            attention.after_task(1)
            torch.testing.assert_close(pre, F.linear(x, attention.qkv.weight, attention.qkv.bias),
                                       atol=1e-6, rtol=1e-5)
            self.assertEqual(torch.count_nonzero(attention._contrib_from_units(x, 1)).item(), 0)
            attention.before_task(0)
            self.assertEqual(attention.p_conflict_strength_scale, 1.)

    def test_real_adaptive_rules_scale_effective_beta_not_mask_or_merge(self):
        learner, attention = self.fixture()
        attention.dual_mask_conflict_exact_topk = False
        attention.dual_mask_conflict_energy_adaptive = True
        attention.dual_mask_conflict_energy_ratio_floor = True
        attention.dual_mask_conflict_old_overlap_adaptive = True
        attention.set_pretrained_old_overlap_risk(.2)
        reference = copy.deepcopy(attention)
        unit = attention.P_lora[1]
        raw = unit.B_weight @ unit.A_weight
        _, _, mask = attention._safe_delta(raw, True, return_details=True)
        x = torch.randn(7, 4)
        epoch_hook()(learner, 1)
        self.assertAlmostEqual(attention._conflict_parameters()[1], .6)
        expected = F.linear(x, raw * (1 - attention.general_mask) * (1 - .3 * mask))
        torch.testing.assert_close(attention._masked_unit_forward(x, unit, True), expected)
        self.assertTrue(torch.equal(mask, attention._safe_delta(raw, True, return_details=True)[2]))
        attention.after_task(1)
        reference.after_task(1)
        self.assertTrue(torch.equal(attention.qkv.weight, reference.qkv.weight))

    def test_queue_is_one_matched_full_candidate_without_data_path_or_saves(self):
        spec = json.loads((ROOT / 'scripts/sweeps/imgr10_p_conflict_warmup_3090.json').read_text())
        baseline = json.loads((ROOT / 'scripts/sweeps/imgr10_granularity_budget_refresh_3090.json').read_text())
        expected = dict(baseline['common_overrides'], wandb_group=spec['name'])
        self.assertEqual(spec['common_overrides'], expected)
        self.assertEqual(spec['seeds'], [1993])
        self.assertEqual([v['overrides'] for v in spec['variants']], [{'p_conflict_strength_warmup': True}])
        output = subprocess.check_output(['bash', 'scripts/9_30_imgr10_p_conflict_warmup_3090.sh',
                                         '--dry-run'], cwd=ROOT, text=True)
        commands = [shlex.split(line[len('Command: '):]) for line in output.splitlines()
                    if line.startswith('Command: ')]
        self.assertEqual(len(commands), 1)
        command = commands[0]
        settings = dict(command[i + 1].split('=', 1) for i, arg in enumerate(command) if arg == '--set')
        for key, value in dict(expected, p_conflict_strength_warmup=True).items():
            encoded = str(value).lower() if isinstance(value, bool) else json.dumps(value, separators=(',', ':')) if isinstance(value, list) else str(value)
            self.assertEqual(settings[key], encoded, key)
        self.assertNotIn('data_path', settings)


if __name__ == '__main__':
    unittest.main()
