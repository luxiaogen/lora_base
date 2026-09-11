import copy
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from test import test_cross_task_training as training_tests
from utils.task0_checkpoint import save_task0_checkpoint, load_task0_checkpoint, dataset_signature


class Task0CheckpointTests(unittest.TestCase):
    def test_real_vit_merge_snapshot_and_next_task_initialization(self):
        fixture = training_tests.CrossTaskTrainingTests()
        model = fixture.make_vit_learner(task=0)
        model.args.update(device=[torch.device('cpu')], rank=2)
        model._known_classes = model._total_classes = 2
        attention = model._network.image_encoder.blocks[0].attn
        inputs = torch.randn(2, 3, 8, 8)
        optimizer = torch.optim.SGD([p for p in model._network.parameters() if p.requires_grad], lr=.01)
        fixture.criterion()(model._network(inputs)['logits'], torch.tensor([0, 1])).backward()
        optimizer.step()
        with torch.no_grad():
            attention.after_task(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'task0.pt'
            save_task0_checkpoint(path, model, {}, 'same-data')
            restored, _ = load_task0_checkpoint(path, model.args, 'same-data')
            self.assertTrue(torch.equal(model._network(inputs)['logits'], restored._network(inputs)['logits']))
            state = torch.get_rng_state()
            for learner in (model, restored):
                torch.set_rng_state(state)
                module = learner._network.image_encoder.blocks[0].attn
                module.before_task(1)
                module.set_task_and_stage(1, 0)
            for name, value in model._network.state_dict().items():
                self.assertTrue(torch.equal(value, restored._network.state_dict()[name]), name)

    def make_model(self):
        model = training_tests.CrossTaskTrainingTests().make_learner('task_local', task=0)
        model.args.update(device=[torch.device('cpu')], rank=64, ca_epochs=5, prefix='task0')
        model._known_classes = model._total_classes = 2
        model._class_means = torch.arange(8).reshape(2, 4).float()
        model._class_covs = torch.eye(4).repeat(2, 1, 1)
        model.acc_matrix = np.array([[96., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        model._network.image_encoder.plain_history = torch.tensor([2., 3.])
        model.train_loader = DataLoader(TensorDataset(torch.eye(4)))
        model.task_sizes = [2]
        return model

    def test_roundtrip_preserves_nonregistered_state_curves_and_rng(self):
        model = self.make_model()
        curves = {'cnn_curve': {'top1': [96.]}, 'w0_curve': [70.]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'task0.pt'
            save_task0_checkpoint(path, model, curves, 'same-data')
            expected = (random.random(), np.random.rand(), torch.rand(3))
            args = dict(model.args, prefix='candidate', classification_training_mode='task_local_head_replay')
            restored, history = load_task0_checkpoint(path, args, 'same-data')
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(actual[:2], expected[:2])
            self.assertTrue(torch.equal(actual[2], expected[2]))
            self.assertEqual(history, curves)
            self.assertEqual(restored._known_classes, 2)
            self.assertEqual(restored.task_sizes, [2])
            self.assertTrue(np.array_equal(restored.acc_matrix, model.acc_matrix))
            self.assertTrue(torch.equal(restored._class_means, model._class_means))
            self.assertTrue(torch.equal(restored._class_covs, model._class_covs))
            self.assertTrue(torch.equal(restored._network.image_encoder.plain_history, torch.tensor([2., 3.])))
            self.assertFalse(hasattr(restored, 'train_loader'))
            self.assertTrue(hasattr(model, 'train_loader'))
            self.assertEqual(restored.args['classification_training_mode'], 'task_local_head_replay')

    def test_changed_protocol_dataset_and_overwrite_are_rejected(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'task0.pt'
            save_task0_checkpoint(path, model, {}, 'same-data')
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_task0_checkpoint(path, model, {}, 'same-data')
            self.assertEqual(path.read_bytes(), original)
            for args, signature in [(dict(model.args, rank=32), 'same-data'),
                                    (dict(model.args, seed=1996), 'same-data'),
                                    (model.args, 'different-split')]:
                with self.assertRaises(ValueError):
                    load_task0_checkpoint(path, args, signature)

    def test_task_bias_candidate_can_resume_the_same_task0_checkpoint(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'task0.pt'
            save_task0_checkpoint(path, model, {}, 'same-data')
            args = dict(model.args, dual_mask_task_bias_calibration=True)
            try:
                restored, _ = load_task0_checkpoint(path, args, 'same-data')
            except ValueError as error:
                self.fail(str(error))
        self.assertTrue(restored.args['dual_mask_task_bias_calibration'])

    def test_restored_next_training_updates_match_including_optimizer_reset(self):
        model = self.make_model()
        model._network.numtask = 2
        targets = torch.tensor([0, 1, 0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'task0.pt'
            save_task0_checkpoint(path, model, {}, 'same-data')
            restored, _ = load_task0_checkpoint(path, model.args, 'same-data')
            for learner in (model, restored):
                learner._cur_task = 1
                learner._network.classifier_pool[0].requires_grad_(False)
                learner._network.classifier_pool[1].requires_grad_(True)
                optimizer = torch.optim.SGD([p for p in learner._network.parameters() if p.requires_grad], lr=.01)
                output = learner._network(torch.eye(4))
                loss, _ = learner._classification_training_loss(output, targets, training_tests.CrossTaskTrainingTests().criterion(), None)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            for name, value in model._network.state_dict().items():
                self.assertTrue(torch.equal(value, restored._network.state_dict()[name]), name)

    def test_signature_changes_with_order_targets_and_split(self):
        dm = SimpleNamespace(_class_order=[1, 0], _train_data=np.array(['/data/train/a.jpg', '/data/train/b.jpg']),
                             _train_targets=np.array([0, 1]), _test_data=np.array(['/data/test/c.jpg']),
                             _test_targets=np.array([0]))
        sig = dataset_signature(dm)
        changed = copy.deepcopy(dm)
        changed._train_data = changed._train_data[::-1]
        self.assertNotEqual(sig, dataset_signature(changed))
        changed = copy.deepcopy(dm)
        changed._test_targets[0] = 1
        self.assertNotEqual(sig, dataset_signature(changed))


if __name__ == '__main__':
    unittest.main()
