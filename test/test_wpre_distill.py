"""Current-train teacher selection and S-only auxiliary optimizer updates."""
import ast
import copy
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from test.test_pair_separation import backward_hook


class TeacherLossTests(unittest.TestCase):
    def test_only_teacher_correct_samples_and_global_labels_are_used(self):
        from utils.wpre_distill import selective_feature_loss
        teacher = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], requires_grad=True)
        student = torch.tensor([[0., 1.], [0., 1.], [-1., 0.]], requires_grad=True)
        weight = torch.tensor([[-2., -2., 1., 0.], [-2., -2., 0., 1.]], requires_grad=True)
        loss, metrics = selective_feature_loss(student, teacher, weight, torch.tensor([2, 3, 3]))
        self.assertAlmostEqual(loss.item(), .5)
        self.assertAlmostEqual(metrics['wpre_selected_ratio'].item(), 2 / 3, places=6)
        loss.backward()
        self.assertTrue(torch.equal(student.grad[2], torch.zeros(2)))
        self.assertIsNone(teacher.grad)
        self.assertIsNone(weight.grad)

    def test_no_teacher_correct_samples_has_differentiable_zero(self):
        from utils.wpre_distill import selective_feature_loss
        student = torch.randn(3, 2, requires_grad=True)
        loss, metrics = selective_feature_loss(student, torch.ones(3, 2), torch.eye(2),
                                                torch.ones(3, dtype=torch.long))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(metrics['wpre_selected_ratio'].item(), 0.)
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

    def test_teacher_restores_mixed_modes_anchor_and_all_rng_states(self):
        from utils.wpre_distill import teacher_features
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout()
                self.anchor = False

            def forward(self, x):
                random.random()
                np.random.rand()
                torch.rand(1)
                self.assert_anchor = self.anchor
                return {'features': self.dropout(x)}

        net = Network().train()
        net.dropout.eval()
        @contextmanager
        def anchor():
            net.anchor = True
            try:
                yield
            finally:
                net.anchor = False

        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        result = teacher_features(net, torch.ones(3, 2, requires_grad=True), anchor())
        self.assertFalse(result.requires_grad)
        self.assertTrue(net.assert_anchor)
        self.assertFalse(net.anchor)
        self.assertTrue(net.training)
        self.assertFalse(net.dropout.training)
        self.assertEqual(before[0], random.getstate())
        after_np = np.random.get_state()
        self.assertEqual(before[1][0], after_np[0])
        np.testing.assert_array_equal(before[1][1], after_np[1])
        self.assertEqual(before[1][2:], after_np[2:])
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))


class SOnlyUpdateTests(unittest.TestCase):
    def run_step(self, weight=1., task=1, diagnostic=False, reference=False, select=True):
        torch.manual_seed(29)
        class Unit(nn.Module):
            def __init__(self):
                super().__init__()
                self.A_weight = nn.Parameter(torch.randn(3, 4), requires_grad=False)
                self.B_weight = nn.Parameter(torch.randn(4, 3) * .2)

        s, p = Unit(), Unit()
        head = nn.Linear(4, 2, bias=False)
        parameters = list(s.parameters()) + list(p.parameters()) + list(head.parameters())
        optimizer = torch.optim.SGD(parameters, lr=.02, momentum=.9)
        module = SimpleNamespace(S_lora=[s, s], P_lora=[None, p])
        # All six known current training labels are global class 2.
        ridge = torch.zeros(4, 4)
        ridge[:, 2 if select else 0] = 1.
        learner = SimpleNamespace(args={'wpre_distill_weight': weight}, _cur_task=task,
                                  _known_classes=2, _wpre_ridge_weight=ridge,
                                  _iter_lora_modules=lambda: [module],
                                  _wpre_distill_diagnostic=(0, 0) if diagnostic else None)
        x, labels = torch.randn(6, 4), torch.zeros(6, dtype=torch.long)
        features = x + x @ (s.B_weight @ s.A_weight).t() + x @ (p.B_weight @ p.A_weight).t()
        teacher = torch.ones_like(features)
        task_loss = F.cross_entropy(head(features), labels)
        extra = .01 * (s.B_weight.square().mean() + p.B_weight.square().mean())
        rng = torch.get_rng_state().clone()
        if reference:
            loss = task_loss + extra
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            loss = backward_hook()(learner, task_loss, extra, optimizer,
                                   {'features': features, 'wpre_teacher_features': teacher}, labels)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        return [v.detach().clone() for v in parameters], copy.deepcopy(optimizer.state_dict()), loss, learner

    def test_auxiliary_changes_only_s_b_in_actual_training_hook(self):
        base, _, _, _ = self.run_step(weight=0.)
        candidate, _, _, learner = self.run_step()
        self.assertFalse(torch.equal(base[1], candidate[1]))
        for index in (0, 2, 3, 4):
            self.assertTrue(torch.equal(base[index], candidate[index]), index)
        self.assertEqual(learner._last_wpre_distill_metrics['wpre_selected_ratio'].item(), 1.)

    def test_off_task0_and_no_selected_match_original_sgd_and_momentum(self):
        reference = self.run_step(reference=True)
        for kwargs in ({'weight': 0.}, {'task': 0}, {'select': False}):
            candidate = self.run_step(**kwargs)
            for a, b in zip(reference[0], candidate[0]):
                self.assertTrue(torch.equal(a, b))
            for index, state in reference[1]['state'].items():
                self.assertTrue(torch.equal(state['momentum_buffer'],
                                            candidate[1]['state'][index]['momentum_buffer']))
            self.assertEqual(reference[2].item(), candidate[2].item())

    def test_gradient_logging_does_not_change_updates(self):
        plain = self.run_step()[0]
        with self.assertLogs(level='INFO') as logs:
            measured = self.run_step(diagnostic=True)[0]
        self.assertTrue(any('WpreDistillGrad' in line for line in logs.output))
        for a, b in zip(plain, measured):
            self.assertTrue(torch.equal(a, b))

    def test_real_gates_teacher_restoration_and_scoped_gradient(self):
        from test.test_global_conflict_budget import GlobalBudgetSelectionTests
        from utils.wpre_distill import teacher_features
        torch.manual_seed(72)
        attention = GlobalBudgetSelectionTests._make_attention('layer')
        attention.before_task(1)
        attention.set_task_and_stage(1, 0)
        for unit in (attention.S_lora[1], attention.P_lora[1]):
            unit.A_weight.requires_grad_(False)
            with torch.no_grad():
                unit.B_weight.normal_(std=.03)
        attention.qkv.weight.requires_grad_(False)
        with torch.no_grad():
            attention.qkv.weight.zero_()
            attention.capture_pretrained_anchor(force=True)
            attention.qkv.weight.fill_(.01)

        class Network(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer

            def forward(self, x):
                features = x.mean(1) + self.layer(x, task=1).mean(1)
                return {'features': features}

        network, head = Network(attention), nn.Linear(4, 2)
        initial, head_state = copy.deepcopy(attention.state_dict()), copy.deepcopy(head.state_dict())
        x, labels = torch.rand(6, 3, 4) + 1., torch.zeros(6, dtype=torch.long)
        ridge = torch.zeros(4, 4)
        ridge[:, 2] = 1.
        finals = []
        for weight in (0., 1.):
            attention.load_state_dict(initial)
            head.load_state_dict(head_state)
            optimizer = torch.optim.SGD(list(attention.parameters()) + list(head.parameters()), lr=.02)
            teacher = teacher_features(network, x, attention.use_pretrained_anchor())
            self.assertTrue(torch.equal(attention.qkv.weight, initial['qkv.weight']))
            self.assertFalse(attention.pretrained_anchor_mode)
            output = network(x)
            output['wpre_teacher_features'] = teacher
            learner = SimpleNamespace(args={'wpre_distill_weight': weight}, _cur_task=1,
                                      _known_classes=2, _wpre_ridge_weight=ridge,
                                      _iter_lora_modules=lambda: [attention])
            backward_hook()(learner, F.cross_entropy(head(output['features']), labels),
                            None, optimizer, output, labels)
            finals.append((attention.S_lora[1].B_weight.detach().clone(),
                           attention.P_lora[1].B_weight.detach().clone(), head.weight.detach().clone()))
        self.assertFalse(torch.equal(finals[0][0], finals[1][0]))
        self.assertTrue(torch.equal(finals[0][1], finals[1][1]))
        self.assertTrue(torch.equal(finals[0][2], finals[1][2]))


class DistillWiringTests(unittest.TestCase):
    def test_teacher_context_before_student_and_task0_bypass(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_extra_training_context')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<teacher-hook>', 'exec'), namespace)
        learner = SimpleNamespace(args={'wpre_distill_weight': 1.}, _cur_task=0)
        self.assertEqual(namespace['_extra_training_context'](learner, None, None, 0), {})
        source = Path('methods/dlora.py').read_text()
        self.assertTrue("output.update(batch_context)" in source)
        self.assertTrue("batch_training_metrics.update(getattr(self, '_last_wpre_distill_metrics', {}))" in source)

    def test_fixed_teacher_uses_additive_current_training_stats(self):
        from utils.frozen_readout import FrozenReadout
        stats = FrozenReadout(3, 4)
        stats.update(torch.eye(3), torch.tensor([0, 1, 2]))
        stats.update(torch.ones(2, 3), torch.tensor([3, 3]))
        weight, _ = stats.ridge_weight(1., 4)
        self.assertEqual(stats.samples, 5)
        self.assertEqual(weight.shape, (3, 4))
        self.assertEqual(stats.storage_bytes, (3 * 3 + 3 * 4) * 4)
        source = Path('methods/dlora.py').read_text()
        self.assertTrue('self._wpre_readout.update(features, targets)' in source)
        self.assertTrue('self._wpre_readout.ridge_weight(1., self._total_classes)' in source)

    def test_two_machine_recipes_fulltrain_no_fusion_no_checkpoints(self):
        import sys
        sys.path.insert(0, str(Path('scripts').resolve()))
        try:
            spec = importlib.util.spec_from_file_location('distill_runner', 'scripts/run_wpre_distill.py')
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            for machine in ('3090', '5090'):
                full = runner.settings_for(machine, 't10', Path('/tmp/unit'))
                for name, value in dict(seed=[1993], max_tasks=10, init_epoch=20, epochs=20,
                        ca_epochs=5, disable_fused_sdpa=True, dual_mask_anchor_reg_weight=2.5,
                        wpre_distill_weight=1., two_expert_calibration_holdout_mod=0,
                        ridge_fusion_enabled=False, save_task_weights=False,
                        incremental_holdout=False, task0_validation_enabled=False).items():
                    self.assertEqual(full[name], value, name)
                self.assertNotIn('data_path', full)
                smoke = runner.settings_for(machine, 'smoke', Path('/tmp/unit'))
                self.assertEqual((smoke['max_tasks'], smoke['epochs'], smoke['ca_epochs']), (2, 1, 1))
        finally:
            sys.path.pop(0)


if __name__ == '__main__':
    unittest.main()
