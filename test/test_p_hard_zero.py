"""P-only hard sparsity: exact plastic-space budget and identical forward/merge."""
import copy
import json
from pathlib import Path
import shlex
import subprocess
import unittest

import torch
from torch.nn import functional as F

from test import test_p_conflict_freeze


ROOT = Path(__file__).resolve().parents[1]


class PHardZeroTests(unittest.TestCase):
    def fixture(self, mode='conflict', ratio=.4, task=1):
        module = test_p_conflict_freeze.PConflictFreezeTests().make_attention(freeze=0, task=task)
        module.args.update(p_hard_zero_mode=mode, p_hard_zero_ratio=ratio, seed=1993)
        return module

    def test_exact_budget_only_in_plastic_region_for_all_scores_and_ties(self):
        for mode in ('random', 'magnitude', 'conflict'):
            for ratio in (0., .4, 1.):
                module = self.fixture(mode, ratio)
                for raw in (torch.ones(12, 4), torch.arange(48.).reshape(12, 4) + 1):
                    safe, gate, selected = module._safe_delta(raw, True, return_details=True)
                    plastic = (1 - module.general_mask).bool()
                    self.assertEqual(int(selected.sum()), int(plastic.sum() * ratio))
                    self.assertFalse(bool((selected.bool() & ~plastic).any()))
                    self.assertEqual(int((safe[plastic] == 0).sum()), int(selected.sum()))
                    self.assertTrue(torch.equal(safe, raw * plastic * (1 - selected)))
                    self.assertTrue(torch.equal(gate, plastic * (1 - selected)))

    def test_magnitude_and_conflict_rank_different_coordinates(self):
        module = self.fixture('magnitude')
        raw = torch.arange(48.).reshape(12, 4) + 1
        module.w0_importance.copy_(raw.flip(0).square())
        plastic = (1 - module.general_mask).bool()
        k = int(plastic.sum() * .4)
        indices = plastic.flatten().nonzero(as_tuple=True)[0]
        masks = []
        for mode in ('magnitude', 'conflict'):
            module.args['p_hard_zero_mode'] = mode
            _, _, selected = module._safe_delta(raw, True, return_details=True)
            score = raw.abs() if mode == 'magnitude' else module.w0_importance * raw.abs()
            expected = torch.zeros_like(raw).flatten()
            expected[indices[torch.topk(score.flatten()[indices], k, sorted=False).indices]] = 1
            self.assertTrue(torch.equal(selected, expected.reshape_as(raw)))
            masks.append(selected)
        self.assertFalse(torch.equal(*masks))

    def test_random_is_fixed_per_task_layer_reproducible_and_does_not_consume_rng(self):
        first, second = self.fixture('random'), self.fixture('random')
        raw = torch.arange(48.).reshape(12, 4) + 1
        rng = torch.get_rng_state().clone()
        a = first._safe_delta(raw, True, return_details=True)[2]
        b = second._safe_delta(raw, True, return_details=True)[2]
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, first._safe_delta(raw.flip(0), True, return_details=True)[2]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertNotIn('p_hard_zero_random_mask', first.state_dict())
        first.after_task(1)
        self.assertIsNone(first.p_hard_zero_random_mask)
        first.before_task(1)
        self.assertIsNone(first.p_hard_zero_random_mask)

    def test_s_task0_and_disabled_mode_are_unchanged(self):
        for task in (0, 1):
            module = self.fixture('off', task=task)
            reference = copy.deepcopy(module)
            raw = torch.arange(48.).reshape(12, 4) + 1
            baseline = module._safe_delta(raw, True)
            for mode in ('random', 'magnitude', 'conflict'):
                module.args['p_hard_zero_mode'] = mode
                self.assertTrue(torch.equal(module._safe_delta(raw, False), reference._safe_delta(raw, False)))
                self.assertTrue(torch.equal(module._joint_conflict_regularization(module.S_lora[task], False),
                                            reference._joint_conflict_regularization(reference.S_lora[task], False)))
                if task == 0:
                    self.assertTrue(torch.equal(module._safe_delta(raw, True), baseline))
                    self.assertIsNone(module.p_hard_zero_random_mask)
            module.args['p_hard_zero_mode'] = 'off'
            self.assertTrue(torch.equal(module._safe_delta(raw, True), baseline))

    def test_disabled_and_task0_preserve_full_outputs_gradients_and_rng(self):
        for task in (0, 1):
            module = self.fixture('off', task=task)
            x = torch.randn(7, 4)
            before = module._contrib_from_units(x, task)
            params = [p for p in module.parameters() if p.requires_grad]
            grads = torch.autograd.grad(before.square().sum(), params, allow_unused=True)
            state = copy.deepcopy(module.state_dict())
            rng = torch.get_rng_state().clone()
            if task == 0:
                module.args['p_hard_zero_mode'] = 'random'
            else:
                module.args.pop('p_hard_zero_mode')
            after = module._contrib_from_units(x, task)
            other_grads = torch.autograd.grad(after.square().sum(), params, allow_unused=True)
            self.assertTrue(torch.equal(before, after))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(state[k], v) for k, v in module.state_dict().items()))
            for a, b in zip(grads, other_grads):
                self.assertTrue(b is None if a is None else torch.equal(a, b))

    def test_gradients_pass_through_retained_coordinates_both_b_still_train(self):
        for mode in ('random', 'magnitude', 'conflict'):
            module = self.fixture(mode)
            raw = (torch.arange(48.).reshape(12, 4) + 1).requires_grad_()
            safe, gate, _ = module._safe_delta(raw, True, return_details=True)
            safe.sum().backward()
            self.assertTrue(torch.equal(raw.grad, gate))
            units = (module.S_lora[1], module.P_lora[1])
            initial = [unit.B_weight.detach().clone() for unit in units]
            opt = torch.optim.SGD([unit.B_weight for unit in units], lr=.02)
            module._contrib_from_units(torch.randn(7, 4), 1).square().mean().backward()
            opt.step()
            for unit, before in zip(units, initial):
                self.assertFalse(unit.A_weight.requires_grad)
                self.assertFalse(torch.equal(unit.B_weight, before))

    def test_forward_merge_diagnostics_and_disposal_match_actual_hard_gate(self):
        for mode in ('random', 'magnitude', 'conflict'):
            module = self.fixture(mode)
            module.dual_mask_applied_budget_log = True
            module.eval()
            x = torch.randn(7, 4)
            p = module.P_lora[1]
            raw = module.plora_gamma * (p.B_weight @ p.A_weight)
            safe = module._safe_delta(raw, True)
            diagnostic = module._private_merge_diagnostic(raw, safe, .1, .5)
            self.assertEqual(diagnostic['selected'], 14)
            self.assertEqual(diagnostic['plastic_overlap'], 1)
            self.assertEqual(diagnostic['applied_strength'], 1)
            self.assertEqual(diagnostic['merge_error'], 0)
            before = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
            with self.assertLogs(level='INFO') as logs:
                module.after_task(1)
            after = F.linear(x, module.qkv.weight, module.qkv.bias) + module._contrib_from_units(x, 1)
            torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)
            self.assertIsNone(module.P_lora[1])
            self.assertEqual(torch.count_nonzero(module._contrib_from_units(x, 1)).item(), 0)
            record = json.loads(next(line.split('PHardZero ', 1)[1] for line in logs.output if 'PHardZero ' in line))
            self.assertEqual(record['mode'], mode)
            self.assertEqual(record['plastic_coordinates'], 36)
            self.assertEqual(record['zeroed_coordinates'], 14)
            self.assertAlmostEqual(record['zeroed_fraction_of_plastic'], 14 / 36)
            self.assertEqual(record['merge_error'], 0)
            budget = json.loads(next(line.split('AppliedConflictBudget ', 1)[1] for line in logs.output
                                     if 'AppliedConflictBudget ' in line and '"branch": "P"' in line))
            self.assertEqual(budget['applied_k'], 14)
            self.assertEqual(budget['merge_error'], 0)

    def test_three_full_runs_only_vary_p_selection_no_saves_or_data_path(self):
        spec = json.loads((ROOT / 'scripts/sweeps/imgr10_p_hard_zero_3090.json').read_text())
        baseline = json.loads((ROOT / 'scripts/sweeps/imgr10_p_conflict_warmup_3090.json').read_text())
        expected = dict(baseline['common_overrides'], wandb_group=spec['name'],
                        p_conflict_strength_warmup=False, p_hard_zero_ratio=.4)
        self.assertEqual(spec['common_overrides'], expected)
        self.assertEqual([v['overrides'] for v in spec['variants']],
                         [{'p_hard_zero_mode': mode} for mode in ('random', 'magnitude', 'conflict')])
        output = subprocess.check_output(['bash', 'scripts/9_30_imgr10_p_hard_zero_3090.sh', '--dry-run'],
                                         cwd=ROOT, text=True)
        commands = [shlex.split(line[len('Command: '):]) for line in output.splitlines()
                    if line.startswith('Command: ')]
        self.assertEqual(len(commands), 3)
        for command, mode in zip(commands, ('random', 'magnitude', 'conflict')):
            settings = dict(command[i + 1].split('=', 1) for i, arg in enumerate(command) if arg == '--set')
            self.assertEqual(settings['p_hard_zero_mode'], mode)
            self.assertEqual(settings['save_task_weights'], 'false')
            self.assertEqual(settings['max_tasks'], '10')
            self.assertEqual(settings['dual_mask_anchor_reg_weight'], '2.5')
            self.assertEqual(settings['p_conflict_strength_warmup'], 'false')
            self.assertNotIn('data_path', settings)


if __name__ == '__main__':
    unittest.main()
