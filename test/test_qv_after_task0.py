import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from test import test_private_rank as fixtures
from utils.task0_checkpoint import load_task0_checkpoint, save_task0_checkpoint


class SnapshotLearner:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class QVAfterTask0Tests(unittest.TestCase):
    def make_module(self, **kwargs):
        return fixtures.PrivateRankTests().make_module(**kwargs)

    def test_default_and_task0_are_bitwise_identical_including_rng(self):
        outcomes = []
        for switch in (None, False, True):
            torch.manual_seed(1993)
            args = {} if switch is None else {'dual_mask_qv_after_task0': switch}
            module = self.make_module(**args)
            module.before_task(0)
            module.set_task_and_stage(0, 0)
            x = torch.randn(2, 3, 4)
            optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.1)
            for _ in range(2):
                optimizer.zero_grad()
                loss = module(x, 0).square().mean() + module.anchor_regularization()
                loss.backward()
                optimizer.step()
            self.assertGreater(module.S_lora[0].B.weight[4:8].abs().sum().item(), 0)
            module.after_task(0)
            outcomes.append((copy.deepcopy(module.state_dict()), torch.get_rng_state()))
        for state, rng in outcomes[1:]:
            self.assertTrue(torch.equal(rng, outcomes[0][1]))
            for name, value in outcomes[0][0].items():
                self.assertTrue(torch.equal(value, state[name]), name)

    def test_k_gradient_zero_and_qv_nonzero_for_both_branches(self):
        torch.manual_seed(17)
        module = self.make_module(dual_mask_qv_after_task0=True)
        module.before_task(1)
        module.set_task_and_stage(1, 0)
        module.general_mask.zero_()
        x = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 12)
        loss = (module._contrib_from_units(x, 1) - target).square().mean()
        loss.backward()
        for unit in (module.S_lora[1], module.P_lora[1]):
            q, k, v = unit.B.weight.grad.chunk(3)
            self.assertEqual(k.count_nonzero().item(), 0)
            self.assertGreater(q.norm().item(), 0)
            self.assertGreater(v.norm().item(), 0)

    def test_projection_constraint_covers_gate_modes_regularization_and_merge(self):
        for gate in ('unmasked', 'protect_only', 'full'):
            for merge in ('suppress', 'none'):
                with self.subTest(gate=gate, merge=merge):
                    module = self.make_module(dual_mask_qv_after_task0=True,
                                              dual_mask_conflict_merge_mode=merge)
                    module.before_task(1)
                    for unit in (module.S_lora[1], module.P_lora[1]):
                        with torch.no_grad():
                            unit.B.weight.normal_()
                    with patch.object(module, '_effective_gate_mode', return_value=gate):
                        for isolated, unit in ((False, module.S_lora[1]), (True, module.P_lora[1])):
                            raw = unit.B.weight @ unit.A.weight
                            safe = module._safe_delta(raw, isolated)
                            merged = module._compose_merge_delta(raw, isolated, .1, .5)
                            self.assertEqual(safe[4:8].count_nonzero().item(), 0)
                            self.assertEqual(merged[4:8].count_nonzero().item(), 0)
                            unit.B.weight.grad = None
                            reg = module._joint_conflict_regularization(unit, isolated)
                            reg.backward()
                            self.assertEqual(unit.B.weight.grad[4:8].count_nonzero().item(), 0)
                        for unit in (module.S_lora[1], module.P_lora[1]):
                            unit.B.weight.grad = None
                        module.anchor_regularization().backward()
                        for unit in (module.S_lora[1], module.P_lora[1]):
                            self.assertEqual(unit.B.weight.grad[4:8].count_nonzero().item(), 0)

    def test_training_merge_equivalence_and_task0_k_preserved_through_task2(self):
        torch.manual_seed(19)
        module = self.make_module(dual_mask_qv_after_task0=True)
        anchor = module.pretrained_weight.clone()
        task0_k = None
        for task in range(3):
            module.before_task(task)
            module.set_task_and_stage(task, 0)
            module.train()
            inputs, targets = torch.randn(3, 4, 4), torch.randn(3, 4, 4)
            optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.1)
            for _ in range(3):
                optimizer.zero_grad()
                loss = (module(inputs, task) - targets).square().mean()
                loss.backward()
                optimizer.step()
            module.eval()
            with torch.no_grad():
                before = module(inputs, task)
                module.after_task(task)
                self.assertTrue(torch.allclose(before, module(inputs, task), atol=1e-6))
                if task == 0:
                    task0_k = module.qkv.weight[4:8].clone()
                    self.assertFalse(torch.equal(task0_k, anchor[4:8]))
                else:
                    self.assertTrue(torch.equal(task0_k, module.qkv.weight[4:8]))
                self.assertTrue(torch.equal(anchor, module.pretrained_weight))

    def test_checkpoint_restores_rng_and_switches_both_branches_to_qv(self):
        module = self.make_module(dual_mask_qv_after_task0=False)
        module.before_task(0)
        module.after_task(0)
        model = SnapshotLearner(args=module.args, _network=module, _cur_task=0,
                                _known_classes=2, _total_classes=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'task0.pt'
            save_task0_checkpoint(path, model, {'top1': [96.]}, 'data')
            expected = random.random(), np.random.rand(), torch.rand(3)
            restored, history = load_task0_checkpoint(path, dict(model.args, dual_mask_qv_after_task0=True), 'data')
            actual = random.random(), np.random.rand(), torch.rand(3)
            self.assertEqual(actual[:2], expected[:2])
            self.assertTrue(torch.equal(actual[2], expected[2]))
            self.assertEqual(history, {'top1': [96.]})
            self.assertIs(restored.args, restored._network.args)
            restored._network.before_task(1)
            delta = restored._network._projection_delta(torch.ones(12, 4))
            self.assertEqual(delta[4:8].count_nonzero().item(), 0)
            for args, signature in ((dict(model.args, rank=32), 'data'), (model.args, 'other')):
                with self.assertRaises(ValueError):
                    load_task0_checkpoint(path, args, signature)

    def test_real_learner_tiny_vit_cpu_smoke(self):
        from methods.dlora import Learner
        from models.network import ViT

        torch.manual_seed(19)
        args = json.loads(Path('exps/dlora/imgr10.json').read_text())
        args.update(device=[torch.device('cpu')], embd_dim=8, init_cls=2, increment=2,
                    total_sessions=3, rank=2, num_heads=2, init_epoch=2, epochs=2, num_workers=0,
                    dual_mask_svd_rank=2, dual_mask_competence_adaptive=False,
                    dual_mask_reg_weight=.01, dual_mask_conflict_reg_enabled=False,
                    dual_mask_qv_after_task0=True)
        encoder = ViT(img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2, rank=2, n_tasks=3)
        with patch('models.network._create_vision_transformer', return_value=encoder):
            learner = Learner(args)
        module = next(learner._iter_lora_modules())
        original_step = learner._backward_and_step
        gradient_checks = []

        def step(*values):
            loss = original_step(*values)
            if learner._cur_task > 0:
                for unit in (module.S_lora[learner._cur_task], module.P_lora[learner._cur_task]):
                    q, k, v = unit.B.weight.grad.chunk(3)
                    self.assertEqual(k.count_nonzero().item(), 0)
                    gradient_checks.append(q.norm().item() + v.norm().item())
            return loss

        learner._backward_and_step = step
        for task in range(3):
            learner._cur_task = task
            learner._known_classes, learner._total_classes = task * 2, (task + 1) * 2
            learner._network.numtask = task + 1
            data = TensorDataset(torch.arange(4), torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 0, 1]) + task * 2)
            loader = DataLoader(data, batch_size=4, num_workers=0)
            learner._train(loader, loader)
            self.assertIsNone(module.S_lora[task])
            self.assertIsNone(module.P_lora[task])
            if task == 0:
                task0_k = module.qkv.weight[8:16].detach().clone()
            else:
                self.assertTrue(torch.equal(task0_k, module.qkv.weight[8:16]))
        self.assertEqual(len(gradient_checks), 8)
        self.assertTrue(all(value > 0 for value in gradient_checks))


if __name__ == '__main__':
    unittest.main()
