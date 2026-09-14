import unittest

import torch

from test import test_private_rank as fixtures


class QKAllTasksTests(unittest.TestCase):
    def test_qk_only_updates_q_and_k_through_task_two(self):
        torch.manual_seed(1993)
        module = fixtures.PrivateRankTests().make_module(
            dual_mask_qk_all_tasks=True, dual_mask_p_conflict_diagnostics=True)
        original_v = module.qkv.weight[8:12].clone()
        for task in range(3):
            module.before_task(task)
            module.set_task_and_stage(task, 0)
            x = torch.randn(2, 3, 4)
            target = torch.randn(2, 3, 12)
            optimizer = torch.optim.SGD(
                [p for p in module.parameters() if p.requires_grad], lr=.1)
            optimizer.zero_grad()
            loss = (module._contrib_from_units(x, task) - target).square().mean()
            loss.backward()
            units = [module.S_lora[task]]
            if task > 0:
                units.append(module.P_lora[task])
            for unit in units:
                q_grad, k_grad, v_grad = unit.B.weight.grad.chunk(3)
                self.assertGreater(q_grad.norm().item(), 0)
                self.assertGreater(k_grad.norm().item(), 0)
                self.assertEqual(v_grad.count_nonzero().item(), 0)
            optimizer.step()
            module.eval()
            with torch.no_grad():
                before = module(x, task)
                module.after_task(task)
                torch.testing.assert_close(before, module(x, task), atol=1e-6, rtol=1e-5)
            self.assertTrue(torch.equal(module.qkv.weight[8:12], original_v))
            if task > 0:
                self.assertEqual(module.p_conflict_components[task].delta[8:12].count_nonzero().item(), 0)

    def test_disabled_flag_keeps_task_two_qkv(self):
        module = fixtures.PrivateRankTests().make_module(dual_mask_qk_all_tasks=False)
        module.before_task(2)
        delta = torch.ones(12, 4)
        self.assertTrue(torch.equal(module._projection_delta(delta), delta))


if __name__ == '__main__':
    unittest.main()
