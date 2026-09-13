"""Inheritance must start at zero effective delta and merge new changes once."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from test import test_private_rank as fixtures


class LoRAInheritanceTests(unittest.TestCase):
    def make_module(self, **kwargs):
        return fixtures.PrivateRankTests().make_module(**kwargs)

    def test_disabled_and_task0_preserve_baseline_weights_and_rng(self):
        outcomes = []
        for enabled in (None, False, True):
            torch.manual_seed(1993)
            kwargs = {} if enabled is None else dict(dual_mask_lora_inherit=enabled)
            module = self.make_module(**kwargs)
            module.before_task(0)
            module.set_task_and_stage(0, 0)
            x = torch.randn(2, 3, 4)
            module(x, 0).sum().backward()
            with torch.no_grad():
                module.S_lora[0].B.weight.add_(module.S_lora[0].B.weight.grad, alpha=-.01)
            module.after_task(0)
            state = {k: v for k, v in module.state_dict().items() if not k.startswith('inherit_')}
            outcomes.append((copy.deepcopy(state), torch.get_rng_state()))
        for state, rng in outcomes[1:]:
            self.assertTrue(torch.equal(rng, outcomes[0][1]))
            for key, value in state.items():
                self.assertTrue(torch.equal(value, outcomes[0][0][key]), key)

    def test_three_task_lifecycle_gradients_and_exactly_once_merge(self):
        torch.manual_seed(17)
        module = self.make_module(dual_mask_lora_inherit=True)
        anchor = module.pretrained_weight.clone()
        x = torch.randn(3, 4, 4)
        previous_s = previous_p = None
        for task in range(3):
            module.before_task(task)
            module.set_task_and_stage(task, 0)
            module.eval()
            self.assertEqual(module._contrib_from_units(x, task).count_nonzero().item(), 0)
            if task > 0:
                self.assertTrue(torch.equal(module.S_lora[task].A_weight, previous_s[0]))
                self.assertTrue(torch.equal(module.S_lora[task].B_weight, previous_s[1]))
                self.assertFalse(module.S_lora[task].A_weight.requires_grad)
            if task == 1:
                self.assertIsNone(module.P_lora[task].initial_delta)
                self.assertEqual(module.P_lora[task].B_weight.count_nonzero().item(), 0)
            if task == 2:
                self.assertTrue(torch.equal(module.P_lora[task].A_weight, previous_p[0]))
                self.assertTrue(torch.equal(module.P_lora[task].B_weight, previous_p[1]))
            optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.1)
            for _ in range(3):
                optimizer.zero_grad()
                loss = (module(x, task) - torch.ones_like(x)).square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertGreater(module.S_lora[task].B_weight.grad.norm().item(), 0)
                optimizer.step()
            previous_s = (module.S_lora[task].A_weight.clone(), module.S_lora[task].B_weight.clone())
            if task > 0:
                previous_p = (module.P_lora[task].A_weight.clone(), module.P_lora[task].B_weight.clone())
            with torch.no_grad():
                before = module(x, task)
                module.after_task(task)
                self.assertTrue(torch.allclose(before, module(x, task), atol=1e-6))
                self.assertTrue(torch.equal(module.pretrained_weight, anchor))
                self.assertIsNone(module.S_lora[task])
                self.assertIsNone(module.P_lora[task])

    def test_private_rank_resize_and_no_learning_does_not_remerge(self):
        module = self.make_module(dual_mask_lora_inherit=True)
        module.before_task(1)
        with torch.no_grad():
            module.P_lora[1].B_weight.fill_(.2)
        module.after_task(1)
        previous_a, previous_b = module.inherit_P_A.clone(), module.inherit_P_B.clone()
        base = module.qkv.weight.clone()
        for rank in (1, 4):
            module.dual_mask_private_rank = rank
            module.before_task(2)
            unit = module.P_lora[2]
            copied = min(rank, previous_a.shape[0])
            self.assertTrue(torch.equal(unit.A_weight[:copied], previous_a[:copied]))
            self.assertTrue(torch.equal(unit.B_weight[:, :copied], previous_b[:, :copied]))
            self.assertEqual(unit.delta_weight().count_nonzero().item(), 0)
            if rank > copied:
                self.assertEqual(unit.B_weight[:, copied:].count_nonzero().item(), 0)
            # All loss paths must see the centered update, not the cached BA.
            self.assertEqual(module._joint_conflict_regularization(unit, True).item(), 0)
            module.after_task(2)
            self.assertTrue(torch.equal(base, module.qkv.weight))
            module.inherit_P_A, module.inherit_P_B = previous_a.clone(), previous_b.clone()

    def test_real_learner_three_task_cpu_smoke(self):
        from methods.dlora import Learner
        from models.network import ViT

        torch.manual_seed(19)
        args = json.loads(Path('exps/dlora/imgr10.json').read_text())
        args.update(device=[torch.device('cpu')], embd_dim=8, init_cls=2, increment=2,
                    total_sessions=3, rank=2, num_heads=2, init_epoch=2, epochs=2, num_workers=0,
                    dual_mask_svd_rank=2, dual_mask_competence_adaptive=False,
                    dual_mask_reg_weight=.01, dual_mask_conflict_reg_enabled=False,
                    dual_mask_lora_inherit=True, dual_mask_qv_after_task0=False)
        encoder = ViT(img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2, rank=2, n_tasks=3)
        with patch('models.network._create_vision_transformer', return_value=encoder):
            learner = Learner(args)
        module = next(learner._iter_lora_modules())
        for task in range(3):
            learner._cur_task = task
            learner._known_classes, learner._total_classes = task * 2, (task + 1) * 2
            learner._network.numtask = task + 1
            data = TensorDataset(torch.arange(4), torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 0, 1]) + task * 2)
            loader = DataLoader(data, batch_size=4, num_workers=0)
            learner._train(loader, loader)
            self.assertIsNone(module.S_lora[task])
            self.assertIsNone(module.P_lora[task])
            self.assertTrue(torch.isfinite(module.qkv.weight).all())
            self.assertIsNotNone(module.inherit_S_B)
            if task > 0:
                self.assertIsNotNone(module.inherit_P_B)

    def test_sequential_initialization_does_not_overwrite_inherited_factors(self):
        module = self.make_module(dual_mask_lora_inherit=True, use_slora=False, use_plora=False)
        for task in range(2):
            module.before_task(task)
            module._init_lora_weight(task)
            module.set_task_and_stage(task, 0)
            unit = module.S_lora[task]
            self.assertEqual(unit.delta_weight().count_nonzero().item(), 0)
            if task == 1:
                self.assertTrue(torch.equal(unit.B_weight, module.inherit_S_B))
            with torch.no_grad():
                unit.B_weight.add_(.1)
            module.after_task(task)

    def test_weight_decay_applies_to_inherited_b_not_centered_coefficients(self):
        module = self.make_module(dual_mask_lora_inherit=True)
        module.before_task(0)
        with torch.no_grad():
            module.S_lora[0].B_weight.fill_(.2)
        module.after_task(0)
        module.before_task(1)
        module.set_task_and_stage(1, 0)
        unit = module.S_lora[1]
        start = unit.initial_delta.clone()
        optimizer = torch.optim.SGD([unit.B_weight], lr=.1, weight_decay=.01)
        unit.B_weight.grad = torch.zeros_like(unit.B_weight)
        optimizer.step()
        self.assertTrue(torch.allclose(unit.delta_weight(), -.001 * start, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
