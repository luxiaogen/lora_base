import ast
import contextlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import tempfile
import types
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class TwoScoreNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor_forwards = 0

    def interface(self, inputs):
        torch.rand(1)
        random.random()
        np.random.rand()
        return inputs[:, :3]

    def extract_vector(self, inputs):
        self.anchor_forwards += 1
        return inputs[:, 3:]


class RidgeFusionTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('utils.ridge_fusion')

    def test_zero_weight_is_exact_base_and_nonzero_changes_real_prediction(self):
        base = torch.tensor([[.51, .50, .10], [.52, .50, .10]])
        ridge = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
        torch.testing.assert_close(self.api.fuse_scores(base, ridge, 0.), base)
        self.assertEqual(self.api.fuse_scores(base, ridge, .1).argmax(1).tolist(), [1, 0])

    def test_each_expert_scale_and_offset_do_not_change_fusion_prediction(self):
        base = torch.tensor([[.51, .50, .10], [.52, .50, .10]])
        ridge = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
        first = self.api.fuse_scores(base, ridge, .1).argmax(1)
        second = self.api.fuse_scores(base * 20 + 10, ridge * 100 - 3, .1).argmax(1)
        self.assertEqual(first.tolist(), second.tolist())

    def test_current_holdout_can_select_fusion_and_ties_keep_base(self):
        base = torch.tensor([[.51, .50, .10], [.52, .50, .10]])
        ridge = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
        coefficient, rows = self.api.select_coefficient(base, ridge, torch.tensor([1, 0]))
        self.assertGreater(coefficient, 0.)
        self.assertEqual(rows[0]['utility'], 0)
        coefficient, _ = self.api.select_coefficient(base, ridge, torch.tensor([0, 0]))
        self.assertEqual(coefficient, 0.)

    def test_potential_harm_is_penalized_not_hidden_by_rescue_count(self):
        base = torch.tensor([[.51, .50, .10]] * 4)
        ridge = torch.tensor([[0., 1., 0.]] * 4)
        coefficient, rows = self.api.select_coefficient(base, ridge, torch.tensor([1, 1, 0, 2]))
        self.assertEqual(coefficient, 0.)  # Two rescues do not offset one cost-3 harm.
        changed = next(row for row in rows if row['rescued'])
        self.assertEqual(changed['rescued'], 2)
        self.assertEqual(changed['harmed'], 1)
        self.assertEqual(changed['utility'], -1)

    def test_constant_scores_remain_finite(self):
        scores = self.api.fuse_scores(torch.ones(2, 4), torch.ones(2, 4), .1)
        self.assertTrue(torch.isfinite(scores).all())

    def test_primary_prediction_api_has_no_label_or_task_input(self):
        import inspect
        self.assertEqual(list(inspect.signature(self.api.fuse_scores).parameters),
                         ['base', 'ridge', 'coefficient'])

    def test_pair_report_has_real_fusion_metrics_and_separate_base_matrix(self):
        state = self.api.RidgeFusion(3, 4, 2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            report = state.report(0, np.array([0, 1, 0]), np.array([0, 1, 1]),
                                  np.array([0, 1, 1]), tmp)
            self.assertAlmostEqual(report['base']['total'], 200 / 3)
            self.assertEqual(report['candidate']['total'], 100.)
            self.assertEqual(report['rescued'], 1)
            report = state.report(1, np.array([0, 0, 2, 3]), np.array([0, 1, 2, 2]),
                                  np.array([0, 1, 2, 3]), tmp)
            self.assertEqual(report['candidate']['old'], 100.)
            self.assertEqual(report['candidate']['new'], 50.)
            self.assertEqual(report['candidate_forgetting'], 0.)
            self.assertAlmostEqual(report['base_forgetting'], 50 / 3)
            self.assertFalse(any(p.suffix in ('.pt', '.npy', '.npz') for p in Path(tmp).rglob('*')))

    def test_hooks_prepare_policy_after_ca_and_use_it_in_formal_inference(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Learner')
        functions = {node.name: ast.get_source_segment(Path('methods/dlora.py').read_text(), node)
                     for node in cls.body if isinstance(node, ast.FunctionDef)}
        train = functions['incremental_train']
        self.assertLess(train.rfind('self._stage2_compact_classifier('),
                        train.rfind('self._prepare_ridge_fusion('))
        self.assertIn('self._ridge_fusion_ready = False', train)
        self.assertIn('self._ridge_fusion.update(features, targets)', functions['_prepare_w0_prototypes'])
        self.assertIn('fuse_scores(', functions['_eval_cnn'])
        self.assertIn('self._ridge_fusion.report(', functions['eval_task'])

    def test_real_formal_eval_uses_fusion_without_reading_target_for_prediction(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Learner')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_eval_cnn')
        namespace = dict(torch=torch, np=np)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual_eval_cnn>', 'exec'), namespace)
        network = TwoScoreNetwork()
        learner = types.SimpleNamespace(_network=network, _device='cpu', class_num=3, topk=1,
                                        _ridge_fusion_ready=True,
                                        _ridge_fusion=types.SimpleNamespace(coefficient=.1),
                                        _ridge_fusion_weight=torch.eye(3),
                                        _pretrained_anchor_context=contextlib.nullcontext)
        inputs = torch.tensor([[.51, .50, .10, 0., 1., 0.], [.52, .50, .10, 1., 0., 0.]])
        first = namespace['_eval_cnn'](learner, [(torch.arange(2), inputs, torch.tensor([1, 0]))])[0]
        second = namespace['_eval_cnn'](learner, [(torch.arange(2), inputs, torch.tensor([2, 2]))])[0]
        self.assertEqual(first.tolist(), [1, 0])
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(learner._ridge_fusion_eval_base.tolist(), [0, 0])
        learner._ridge_fusion_ready = False
        calls = network.anchor_forwards
        original = namespace['_eval_cnn'](learner, [(torch.arange(2), inputs, torch.tensor([1, 0]))])[0]
        self.assertEqual(original.tolist(), [0, 0])
        self.assertEqual(network.anchor_forwards, calls)

    def test_calibration_preserves_all_rng_and_model_modes(self):
        network = TwoScoreNetwork().train()
        inputs = torch.tensor([[.51, .50, .10, 0., 1., 0.], [.52, .50, .10, 1., 0., 0.]])
        loader = DataLoader(TensorDataset(torch.arange(2), inputs, torch.tensor([1, 0])),
                            batch_size=1, generator=torch.Generator().manual_seed(11))
        before = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        base, ridge, labels = self.api.collect_calibration(network, loader, 'cpu',
                                                           contextlib.nullcontext, torch.eye(3))
        torch.testing.assert_close(base, inputs[:, :3])
        torch.testing.assert_close(ridge, inputs[:, 3:])
        self.assertEqual(labels.tolist(), [1, 0])
        self.assertTrue(network.training)
        torch.testing.assert_close(torch.get_rng_state(), before[0])
        self.assertEqual(random.getstate(), before[1])
        np.testing.assert_equal(np.random.get_state(), before[2])

    def test_wrappers_really_train_t10_not_a_readout_cache_only_job(self):
        for machine, mod in (('3090', 20), ('5090', 10)):
            result = subprocess.run(['bash', f'scripts/10_01_imgr10_ridge_fusion_{machine}.sh', '--dry-run'],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            commands = [line for line in lines if line.startswith('Command:')]
            self.assertEqual(len(commands), 2)
            self.assertIn(' main.py ', commands[1])
            self.assertIn('--set max_tasks=10 ', commands[1])
            self.assertIn('--set epochs=20 ', commands[1])
            self.assertIn('--set ca_epochs=5 ', commands[1])
            self.assertIn('--set ridge_fusion_enabled=true ', commands[1])
            self.assertIn('--set save_task_weights=false ', commands[1])
            self.assertIn(f'--set two_expert_calibration_holdout_mod={mod} ', commands[1])

    def test_sealed_policy_and_report_do_not_refit_using_test_labels(self):
        state = self.api.RidgeFusion(3, 4, 2, 2)
        state.coefficient = .1
        with tempfile.TemporaryDirectory() as tmp:
            state.seal(1, tmp, [dict(coefficient=.1, utility=1)], 5, 1.)
            before = (Path(tmp) / 'task_01_policy.json').read_bytes()
            state.report(1, np.array([0, 2]), np.array([1, 2]), np.array([1, 2]), tmp)
            self.assertEqual((Path(tmp) / 'task_01_policy.json').read_bytes(), before)
            self.assertEqual(state.coefficient, .1)
            self.assertFalse(json.loads(before)['test_labels_used_for_selection'])


if __name__ == '__main__':
    unittest.main()
