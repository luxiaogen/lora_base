import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from methods.dlora import Learner
from models.losses import AngularPenaltySMLoss
from models.network import MANet, ViT
from models.attention import Attention_LoRA


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, image, **kwargs):
        return self.proj(image).unsqueeze(1)


class CrossTaskTrainingTests(unittest.TestCase):
    def make_vit_learner(self, task=1):
        learner = self.make_learner(task=task)
        learner._network.image_encoder = ViT(img_size=8, patch_size=4, embed_dim=4, depth=1,
                                             num_heads=1, n_tasks=2, rank=2, num_classes=0)
        learner._network.image_encoder.requires_grad_(False)
        for module in learner._network.image_encoder.modules():
            if isinstance(module, Attention_LoRA):
                module._init_params(dict(use_slora=True, use_plora=True, lora_A_init='kaiming',
                                         dual_mask_task0_gate_mode='unmasked', dual_mask_private_conflict_mode='global'))
                module.before_task(task)
                module.set_task_and_stage(task, 0)
        return learner

    def test_real_vit_lora_gradient_is_local_only_with_trainable_new_head(self):
        torch.manual_seed(19)
        base = self.make_vit_learner()
        images, targets = torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 0, 1])
        output = base._network(images)
        self.criterion()(output['logits'], targets).backward()
        for mode in ('task_local_head', 'task_local_head_replay'):
            candidate = copy.deepcopy(base)
            candidate._network.zero_grad(set_to_none=True)
            candidate.args['classification_training_mode'] = mode
            output = candidate._network(images)
            loss, _ = candidate._classification_training_loss(output, targets, self.criterion(), candidate._prepare_training_replay())
            loss.backward()
            reference = dict(base._network.named_parameters())
            lora_norm = 0.
            for name, parameter in candidate._network.named_parameters():
                if 'lora' in name and parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.allclose(parameter.grad, reference[name].grad, atol=1e-7), name)
                    lora_norm += parameter.grad.norm().item()
            self.assertGreater(lora_norm, 0.)
            self.assertIsNone(candidate._network.classifier_pool[0].weight.grad)
            head_delta = candidate._network.classifier_pool[1].weight.grad - base._network.classifier_pool[1].weight.grad
            self.assertGreater(head_delta.norm().item(), 0.)

    def test_head_isolation_preserves_encoder_gradient_and_changes_current_head(self):
        for mode in ('task_local_head', 'task_local_head_replay'):
            base = self.make_learner('task_local')
            isolated = copy.deepcopy(base)
            isolated.args['classification_training_mode'] = mode
            targets = torch.tensor([0, 1, 0, 1])
            for learner in (base, isolated):
                output = learner._network(torch.eye(4))
                loss, metrics = learner._classification_training_loss(
                    output, targets, self.criterion(), learner._prepare_training_replay())
                loss.backward()
            self.assertTrue(torch.allclose(base._network.image_encoder.proj.weight.grad,
                                           isolated._network.image_encoder.proj.weight.grad))
            self.assertGreater(isolated._network.image_encoder.proj.weight.grad.norm().item(), 0)
            self.assertIsNone(isolated._network.classifier_pool[0].weight.grad)
            head_delta = isolated._network.classifier_pool[1].weight.grad - base._network.classifier_pool[1].weight.grad
            self.assertGreater(head_delta.norm().item(), 0)
            self.assertIn('classification_local', metrics)
            self.assertIn('classification_head', metrics)

    def test_extra_head_loss_has_zero_feature_gradient(self):
        for mode in ('task_local_head', 'task_local_head_replay'):
            learner = self.make_learner(mode)
            output = learner._network(torch.eye(4))
            targets = torch.tensor([0, 1, 0, 1])
            loss, _ = learner._classification_training_loss(
                output, targets, self.criterion(), learner._prepare_training_replay())
            extra = loss - self.criterion()(output['logits'], targets)
            feature_grad, head_grad = torch.autograd.grad(
                extra, (output['features'], learner._network.classifier_pool[1].weight))
            self.assertTrue(torch.allclose(feature_grad, torch.zeros_like(feature_grad), atol=1e-7))
            self.assertGreater(head_grad.norm().item(), 0)

    def make_learner(self, mode='task_local', task=1):
        learner = Learner.__new__(Learner)
        learner.args = {'classification_training_mode': mode, 'seed': 1993}
        net = MANet.__new__(MANet)
        nn.Module.__init__(net)
        net.image_encoder = TinyEncoder()
        net.classifier_pool = nn.ModuleList([nn.Linear(4, 2, bias=False) for _ in range(2)])
        net.numtask, net.use_RP = task + 1, False
        for i, head in enumerate(net.classifier_pool):
            head.requires_grad_(i == task)
        learner._network = net
        learner._cur_task, learner._known_classes = task, task * 2
        learner._device = torch.device('cpu')
        learner._class_means = torch.eye(4)[:task * 2].clone()
        learner._class_covs = torch.eye(4).repeat(task * 2, 1, 1) * 0.1
        learner.scale, learner.margin, learner.run_epoch = 5.0, 0.1, 1
        return learner

    def criterion(self):
        return AngularPenaltySMLoss(s=5.0, m=0.1)

    def test_default_and_task0_exactly_keep_local_loss_and_rng(self):
        for mode, task in [('task_local', 1), ('all_seen', 0), ('all_seen_replay', 0),
                           ('task_local_head', 0), ('task_local_head_replay', 0)]:
            learner = self.make_learner(mode, task)
            if mode == 'task_local':
                learner.args.pop('classification_training_mode')
            state = torch.get_rng_state().clone()
            replay = learner._prepare_training_replay()
            self.assertIsNone(replay)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            output = learner._network(torch.eye(4))
            targets = torch.tensor([0, 1, 0, 1])
            loss, metrics = learner._classification_training_loss(output, targets, self.criterion(), replay)
            self.assertTrue(torch.equal(loss, self.criterion()(output['logits'], targets)))
            self.assertEqual(metrics, {})

    def test_all_seen_uses_global_labels_and_preserves_feature_gradient(self):
        learner = self.make_learner('all_seen')
        output = learner._network(torch.eye(4))
        targets = torch.tensor([0, 1, 0, 1])
        loss, metrics = learner._classification_training_loss(output, targets, self.criterion(), None)
        logits = learner._network(output['features'], fc_only=True)
        self.assertTrue(torch.equal(loss, self.criterion()(logits, targets + 2)))
        loss.backward()
        self.assertGreater(learner._network.image_encoder.proj.weight.grad.norm().item(), 0)
        self.assertIsNone(learner._network.classifier_pool[0].weight.grad)
        self.assertIsNotNone(learner._network.classifier_pool[1].weight.grad)
        self.assertIn('train_global_acc', metrics)

    def test_replay_is_balanced_detached_repeatable_and_rng_isolated(self):
        learner = self.make_learner('all_seen_replay')
        # Even accidentally attached statistics must never retain a training graph.
        learner._class_means.requires_grad_()
        state = torch.get_rng_state().clone()
        bank, labels, generator = learner._prepare_training_replay()
        other_bank, other_labels, _ = learner._prepare_training_replay()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(bank, other_bank))
        self.assertTrue(torch.equal(labels, other_labels))
        self.assertEqual(labels.bincount().tolist(), [256, 256])
        self.assertFalse(bank.requires_grad)
        self.assertTrue(torch.isfinite(bank).all())
        loss = self.criterion()(learner._network(bank[:8], fc_only=True), labels[:8])
        loss.backward()
        self.assertIsNone(learner._network.image_encoder.proj.weight.grad)
        self.assertIsNone(learner._network.classifier_pool[0].weight.grad)
        self.assertGreater(learner._network.classifier_pool[1].weight.grad.norm().item(), 0)

    def test_replay_adds_unit_weight_loss_without_changing_real_gradient(self):
        base = self.make_learner('all_seen')
        replay_learner = copy.deepcopy(base)
        replay_learner.args['classification_training_mode'] = 'all_seen_replay'
        targets = torch.tensor([0, 1, 0, 1])
        state = torch.get_rng_state().clone()
        for learner in (base, replay_learner):
            replay = learner._prepare_training_replay()
            output = learner._network(torch.eye(4))
            loss, metrics = learner._classification_training_loss(output, targets, self.criterion(), replay)
            loss.backward()
            if replay is not None:
                self.assertTrue(torch.allclose(loss.detach(), metrics['classification_real'] + metrics['classification_replay']))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.allclose(base._network.image_encoder.proj.weight.grad,
                                      replay_learner._network.image_encoder.proj.weight.grad))

    def test_real_training_loop_smoke_all_modes(self):
        for mode in ('task_local', 'all_seen', 'all_seen_replay', 'task_local_head', 'task_local_head_replay'):
            learner = self.make_learner(mode)
            data = TensorDataset(torch.arange(4), torch.eye(4), torch.tensor([2, 3, 2, 3]))
            loader = DataLoader(data, batch_size=2)
            optimizer = torch.optim.SGD([p for p in learner._network.parameters() if p.requires_grad], lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            old_head = learner._network.classifier_pool[0].weight.detach().clone()
            new_head = learner._network.classifier_pool[1].weight.detach().clone()
            with patch.object(learner, '_extra_training_context', return_value=None), \
                 patch.object(learner, '_extra_training_loss', return_value=None), \
                 patch.object(learner, '_compute_accuracy', return_value=50.0):
                learner.train_function(loader, loader, optimizer, scheduler)
            self.assertTrue(torch.equal(old_head, learner._network.classifier_pool[0].weight))
            self.assertFalse(torch.equal(new_head, learner._network.classifier_pool[1].weight))
            for value in learner._network.parameters():
                self.assertTrue(torch.isfinite(value).all())

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_learner('all_sean')._classification_mode()

    def test_task0_update_matches_legacy_loop_for_every_mode(self):
        initial = self.make_learner(task=0)
        data = TensorDataset(torch.arange(4), torch.eye(4), torch.tensor([0, 1, 0, 1]))
        loader = DataLoader(data, batch_size=2)
        results = []
        for mode in ('legacy', 'task_local', 'all_seen', 'all_seen_replay', 'task_local_head', 'task_local_head_replay'):
            learner = copy.deepcopy(initial)
            learner.args['classification_training_mode'] = 'task_local' if mode == 'legacy' else mode
            optimizer = torch.optim.SGD([p for p in learner._network.parameters() if p.requires_grad], lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            torch.manual_seed(123)
            if mode == 'legacy':
                learner._network.train()
                for _, inputs, targets in loader:
                    loss = self.criterion()(learner._network(inputs)['logits'], targets)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            else:
                with patch.object(learner, '_extra_training_context', return_value=None), \
                     patch.object(learner, '_extra_training_loss', return_value=None), \
                     patch.object(learner, '_compute_accuracy', return_value=50.0):
                    learner.train_function(loader, loader, optimizer, scheduler)
            results.append((copy.deepcopy(learner._network.state_dict()), torch.get_rng_state()))
        for params, rng in results[1:]:
            self.assertTrue(torch.equal(rng, results[0][1]))
            for name, value in params.items():
                self.assertTrue(torch.equal(value, results[0][0][name]))


if __name__ == '__main__':
    unittest.main()
