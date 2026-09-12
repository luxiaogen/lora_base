import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from test import test_private_rank as fixtures
from utils.task0_checkpoint import save_task0_checkpoint, load_task0_checkpoint, dataset_signature


class SnapshotLearner:
    def __init__(self, **state):
        self.__dict__.update(state)


class POnlyTests(unittest.TestCase):
    def make_module(self, **overrides):
        return fixtures.PrivateRankTests().make_module(**overrides)

    def test_task0_and_default_behavior_are_unchanged(self):
        results = []
        for settings in ({}, {'dual_mask_s_task0_only': False}, {'dual_mask_s_task0_only': True}):
            torch.manual_seed(7)
            module = self.make_module(**settings)
            module.before_task(0)
            module.set_task_and_stage(0, 0)
            inputs, target = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
            optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.01)
            for _ in range(2):
                optimizer.zero_grad()
                loss = (module(inputs, task=0) - target).square().mean() + 10 * module.anchor_regularization()
                loss.backward()
                optimizer.step()
            module.after_task(0)
            results.append((copy.deepcopy(module.state_dict()), torch.get_rng_state()))
        for state, rng in results[1:]:
            self.assertTrue(torch.equal(rng, results[0][1]))
            for name, value in state.items():
                self.assertTrue(torch.equal(value, results[0][0][name]), name)

    def test_p_initialization_and_rng_match_baseline(self):
        results = []
        for flag in (False, True):
            torch.manual_seed(19)
            module = self.make_module(dual_mask_s_task0_only=flag)
            module.before_task(1)
            module.set_task_and_stage(1, 0)
            results.append((copy.deepcopy(module.state_dict()), torch.get_rng_state()))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        for name, value in results[0][0].items():
            self.assertTrue(torch.equal(value, results[1][0][name]), name)

    def test_p_only_gradients_training_and_exactly_once_merge(self):
        torch.manual_seed(1993)
        module = self.make_module(dual_mask_s_task0_only=True)
        anchor = module.pretrained_weight.clone()
        for task in (1, 2):
            module.before_task(task)
            module.set_task_and_stage(task, 0)
            s, p = module.S_lora[task], module.P_lora[task]
            self.assertFalse(any(v.requires_grad for v in s.parameters()))
            self.assertFalse(p.A.weight.requires_grad)
            self.assertTrue(p.B.weight.requires_grad)
            inputs, target = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
            optimizer = torch.optim.SGD([v for v in module.parameters() if v.requires_grad], lr=.1)
            for _ in range(3):
                optimizer.zero_grad()
                loss = (module(inputs, task=task) - target).square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNone(s.B.weight.grad)
                self.assertGreater(p.B.weight.grad.norm().item(), 0)
                optimizer.step()
            self.assertEqual(s.B.weight.count_nonzero().item(), 0)
            self.assertGreater(p.B.weight.norm().item(), 0)
            module.eval()
            with torch.no_grad():
                before = module(inputs, task=task)
                anchor_loss = module.anchor_regularization()
                # Even a nonzero inactive S must be excluded from forward, anchor and merge.
                s.B.weight.fill_(99)
                self.assertTrue(torch.equal(before, module(inputs, task=task)))
                self.assertTrue(torch.equal(anchor_loss, module.anchor_regularization()))
                module.after_task(task)
                self.assertTrue(torch.allclose(before, module(inputs, task=task), atol=1e-6))
                merged = module.qkv.weight.clone()
                module.after_task(task)
                self.assertTrue(torch.equal(merged, module.qkv.weight))
            self.assertIsNone(module.S_lora[task])
            self.assertIsNone(module.P_lora[task])
            self.assertTrue(torch.equal(anchor, module.pretrained_weight))

    def test_checkpoint_preserves_state_rng_and_updates_module_switch(self):
        module = self.make_module(dual_mask_s_task0_only=False)
        module.before_task(0)
        module.after_task(0)
        model = SnapshotLearner(args=module.args, _network=module, _cur_task=0,
                                _known_classes=2, _total_classes=2,
                                _class_means=torch.randn(2, 4), _class_covs=torch.eye(4).repeat(2, 1, 1),
                                train_loader=DataLoader(TensorDataset(torch.eye(4))))
        curves = {'cnn_curve': {'top1': [96.]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'task0.pt'
            save_task0_checkpoint(path, model, curves, 'data')
            expected = (random.random(), np.random.rand(), torch.rand(3))
            args = dict(model.args, dual_mask_s_task0_only=True, prefix='candidate')
            restored, history = load_task0_checkpoint(path, args, 'data')
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(actual[:2], expected[:2])
            self.assertTrue(torch.equal(actual[2], expected[2]))
            self.assertEqual(history, curves)
            self.assertIs(restored.args, restored._network.args)
            self.assertFalse(restored._network._shared_branch_active(1))
            self.assertFalse(hasattr(restored, 'train_loader'))
            for name in ('_class_means', '_class_covs'):
                self.assertTrue(torch.equal(getattr(restored, name), getattr(model, name)))
            for changed_args, signature in ((dict(args, rank=32), 'data'), (args, 'other-data')):
                with self.assertRaises(ValueError):
                    load_task0_checkpoint(path, changed_args, signature)
            with self.assertRaises(FileExistsError):
                save_task0_checkpoint(path, model, curves, 'data')

    def test_signature_covers_split_labels_and_order(self):
        dm = SimpleNamespace(_class_order=[1, 0], _train_data=['a', 'b'], _train_targets=[0, 1],
                             _test_data=['c'], _test_targets=[0])
        baseline = dataset_signature(dm)
        for key, value in (('_class_order', [0, 1]), ('_train_data', ['b', 'a']),
                           ('_test_targets', [1]), ('_test_data', ['a'])):
            changed = copy.deepcopy(dm)
            setattr(changed, key, value)
            self.assertNotEqual(baseline, dataset_signature(changed))

    def test_real_learner_tiny_vit_task0_to_task2_cpu_smoke(self):
        from methods.dlora import Learner
        from models.network import ViT

        torch.manual_seed(19)
        args = json.loads(Path('exps/dlora/imgr10.json').read_text())
        args.update(device=[torch.device('cpu')], embd_dim=8, init_cls=2, increment=2,
                    total_sessions=3, rank=2, num_heads=2, init_epoch=2, epochs=2, num_workers=0,
                    dual_mask_svd_rank=2, dual_mask_competence_adaptive=False,
                    dual_mask_reg_weight=.01, dual_mask_conflict_reg_enabled=False,
                    dual_mask_s_task0_only=True)
        encoder = ViT(img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2, rank=2, n_tasks=3)
        with patch('models.network._create_vision_transformer', return_value=encoder):
            learner = Learner(args)
        module = next(learner._iter_lora_modules())
        original_step = learner._backward_and_step
        gradient_checks = []

        def step(*values):
            loss = original_step(*values)
            task = learner._cur_task
            if task > 0:
                self.assertIsNone(module.S_lora[task].B.weight.grad)
                gradient_checks.append(module.P_lora[task].B.weight.grad.norm().item())
            return loss

        learner._backward_and_step = step
        for task in range(3):
            learner._cur_task = task
            learner._known_classes, learner._total_classes = task * 2, (task + 1) * 2
            learner._network.numtask = task + 1
            data = TensorDataset(torch.arange(4), torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 0, 1]) + task * 2)
            loader = DataLoader(data, batch_size=4, num_workers=0)
            previous_heads = [head.weight.detach().clone() for head in learner._network.classifier_pool[:task]]
            learner._train(loader, loader)
            self.assertIsNone(module.S_lora[task])
            self.assertIsNone(module.P_lora[task])
            self.assertTrue(torch.isfinite(module.qkv.weight).all())
            for before, head in zip(previous_heads, learner._network.classifier_pool[:task]):
                self.assertTrue(torch.equal(before, head.weight))
        self.assertEqual(len(gradient_checks), 4)
        self.assertTrue(all(value > 0 for value in gradient_checks))


if __name__ == '__main__':
    unittest.main()
