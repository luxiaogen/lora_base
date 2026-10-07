import copy
import json
from types import SimpleNamespace
import unittest

import torch

from test.test_compact_dualmask import CompactDualMaskTests


class TailUpdateTests(unittest.TestCase):
    def make(self, rule='soft_tail', task=1):
        module = CompactDualMaskTests().template()
        module.args.update(dual_mask_permission_mode='asymmetric', dual_mask_update_rule=rule)
        module.before_task(task)
        module.set_task_and_stage(task, 2)
        return module

    def test_formula_gradient_threshold_ties_and_zero(self):
        module = self.make()
        module.general_mask.zero_()
        value = torch.zeros(12, 4)
        value.flatten()[:5] = torch.tensor([2., -2., 1., -1., .99])
        value.requires_grad_()
        result = module._tail_update_state(value, False)
        self.assertEqual(float(result['threshold']), 1.)
        torch.testing.assert_close(result['safe'].flatten()[:5], torch.tensor([1.5, -1.5, 1., -1., .99]))
        result['safe'].sum().backward()
        torch.testing.assert_close(value.grad.flatten()[:5], torch.tensor([.5, .5, 1., 1., 1.]))
        self.assertFalse(result['threshold'].requires_grad)
        for value in (torch.zeros(12, 4), torch.ones(12, 4)):
            value.requires_grad_()
            result = module._tail_update_state(value, False)
            torch.testing.assert_close(result['safe'], value)
            result['safe'].sum().backward()
            torch.testing.assert_close(value.grad, torch.ones_like(value))
            self.assertEqual(int(result['applied'].sum()), 0)

    def test_budget_match_after_permissions_both_branches(self):
        value = torch.linspace(-3, 4, 48).reshape(12, 4)
        for isolated in (False, True):
            reference = self.make()
            state = reference._tail_update_state(value, isolated)
            for rule in ('step_tail_matched', 'uniform_tail_matched'):
                candidate = copy.deepcopy(reference)
                candidate.args['dual_mask_update_rule'] = rule
                actual = candidate._tail_update_state(value, isolated)
                torch.testing.assert_close((state['base'] - actual['safe']).norm(),
                                          (state['base'] - state['safe']).norm())
                self.assertLessEqual(float(actual['coefficient']), .5)
                self.assertFalse(actual['coefficient'].requires_grad)
                self.assertTrue(torch.all(actual['safe'].abs() <= state['base'].abs()))
                if isolated:
                    self.assertEqual(int((actual['safe'] * candidate.general_mask).count_nonzero()), 0)
            zero = reference._tail_update_state(torch.zeros_like(value), isolated)
            self.assertTrue(torch.isfinite(zero['gate']).all())

    def test_default_and_task0_rng_output_gradient_unchanged(self):
        reference = self.make('step', 0)
        for rule in ('soft_tail', 'step_tail_matched', 'uniform_tail_matched'):
            candidate = copy.deepcopy(reference)
            candidate.args['dual_mask_update_rule'] = rule
            state = torch.get_rng_state().clone()
            reference.before_task(0)
            final = torch.get_rng_state().clone()
            torch.set_rng_state(state)
            candidate.before_task(0)
            self.assertTrue(torch.equal(final, torch.get_rng_state()))
            x = torch.randn(3, 4)
            first, second = [m._contrib_from_units(x, 0) for m in (reference, candidate)]
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            first.sum().backward()
            second.sum().backward()
            torch.testing.assert_close(reference.S_lora[0].B_weight.grad, candidate.S_lora[0].B_weight.grad)
            reference.zero_grad()
        default = self.make('step')
        explicit = copy.deepcopy(default)
        default.args.pop('dual_mask_update_rule')
        value = torch.randn(12, 4, requires_grad=True)
        torch.testing.assert_close(default._safe_delta(value, True), explicit._safe_delta(value, True), atol=0, rtol=0)

    def test_zero_b_gradient_and_merge_once_for_each_rule(self):
        for rule in ('soft_tail', 'step_tail_matched', 'uniform_tail_matched'):
            module = self.make(rule).double()
            x = torch.randn(5, 4, dtype=torch.float64)
            module._contrib_from_units(x, 1).sum().backward()
            for unit in (module.S_lora[1], module.P_lora[1]):
                self.assertGreater(float(unit.B_weight.grad.norm()), 0)
                self.assertIsNone(unit.A_weight.grad)
                with torch.no_grad():
                    unit.B_weight.normal_()
            before = module.qkv(x) + module._contrib_from_units(x, 1)
            module.after_task(1)
            torch.testing.assert_close(before, module.qkv(x), atol=1e-9, rtol=1e-9)
            weights = module.qkv.weight.clone()
            module.after_task(1)
            torch.testing.assert_close(weights, module.qkv.weight, atol=0, rtol=0)

    def test_readonly_sampling_leaves_next_gradient_and_rng_unchanged(self):
        from utils.tail_update_audit import sample_updates
        module = self.make()
        with torch.no_grad():
            module.S_lora[1].B_weight.normal_()
            module.P_lora[1].B_weight.normal_()
        reference = copy.deepcopy(module)
        learner = SimpleNamespace(_cur_task=1, args=module.args,
            _device=torch.device('cpu'), _iter_lora_modules=lambda: [module])
        rng = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO') as logs:
            sample_updates(learner, 1, 0)
            sample_updates(learner, 1, 1)
        rows = [json.loads(line.split('TailUpdate ', 1)[1]) for line in logs.output if 'TailUpdate ' in line]
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row['gate_sample_change'] == 0 for row in rows if row['step'] == 1))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        x = torch.randn(3, 4)
        for current in (reference, module):
            current._contrib_from_units(x, 1).square().sum().backward()
        torch.testing.assert_close(reference.P_lora[1].B_weight.grad, module.P_lora[1].B_weight.grad)
        self.assertEqual(learner._tail_previous_gates, {})

    def test_zero_gamma_private_merge_diagnostic_stays_finite(self):
        module = self.make()
        module.plora_gamma = 0
        zero = torch.zeros(12, 4)
        row = module._private_merge_diagnostic(zero, zero, .1, .5)
        self.assertEqual(row['merge_error'], 0)


if __name__ == '__main__':
    unittest.main()
