"""Read-only P-conflict interventions: no training changes or label leakage."""
import copy
import json
from pathlib import Path
import random
import shlex
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from test import test_private_rank as fixtures


def make_module(**kwargs):
    return fixtures.PrivateRankTests().make_module(dual_mask_p_conflict_diagnostics=True, **kwargs)


def merged_module():
    module = make_module(dual_mask_competence_adaptive=False,
                         dual_mask_conflict_energy_adaptive=False,
                         dual_mask_conflict_ratio=.5, plora_gamma=.75)
    module.before_task(1)
    # BA is larger in columns 0/1. Only first half of columns are high-scoring;
    # column 0 is protected, so only column 1 retains conflict contribution.
    with torch.no_grad():
        module.S_lora[1].B_weight.zero_()
        module.P_lora[1].A_weight.fill_(1)
        module.P_lora[1].A_weight[:, 2:] = .25
        module.P_lora[1].B_weight.fill_(1)
        module.general_mask.zero_()
        module.general_mask[:, 0] = 1
        module.w0_importance.fill_(.1)
        module.w0_importance[:, :2] = 1
    module.eval()
    module.after_task(1)
    return module


class PConflictTests(unittest.TestCase):
    def require_feature(self, module):
        self.assertTrue(hasattr(module, 'p_conflict_components'), 'P-conflict bank not implemented')

    def test_store_only_suppressed_plastic_conflict_and_snapshot_mask(self):
        module = merged_module()
        self.require_feature(module)
        expected = torch.zeros(12, 4)
        expected[:, 1] = 1.5  # 4 * gamma .75 * remaining strength .5
        self.assertTrue(torch.equal(module.p_conflict_components[1].delta, expected))
        self.assertFalse(module.p_conflict_components[1].delta.requires_grad)
        self.assertEqual(module.p_conflict_components[0].delta.numel(), 0)
        module.general_mask.fill_(1)
        module.w0_importance.zero_()
        self.assertTrue(torch.equal(module.p_conflict_components[1].delta, expected))

    def test_sample_weights_equal_explicit_dense_subtraction_not_double_add(self):
        module = merged_module()
        self.require_feature(module)
        from utils.p_conflict_diagnostics import conflict_weights
        x = torch.randn(3, 5, 4)
        weights = torch.tensor([[0., 1., 0.], [0., .25, .75], [1., 0., 0.]])
        baseline = module(x, 1)
        with conflict_weights(module, weights):
            actual = module(x, 1)
        for i, factor in enumerate((0., -.75, -1.)):
            explicit = copy.deepcopy(module)
            with torch.no_grad():
                explicit.qkv.weight.add_(explicit.p_conflict_components[1].delta, alpha=factor)
            torch.testing.assert_close(actual[i:i+1], explicit(x[i:i+1], 1), atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(actual[0], baseline[0]))
        self.assertFalse(torch.allclose(actual[1:], baseline[1:]))
        self.assertTrue(torch.equal(module(x, 1), baseline))

    def test_all_ones_and_exception_restore_context(self):
        module = merged_module()
        self.require_feature(module)
        from utils.p_conflict_diagnostics import conflict_weights
        x = torch.randn(2, 3, 4)
        baseline = module(x, 1)
        with conflict_weights(module, torch.ones(2, 3)):
            self.assertTrue(torch.equal(baseline, module(x, 1)))
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with conflict_weights(module, torch.zeros(2, 3)):
                    raise RuntimeError('test failure')
            self.assertTrue(torch.equal(baseline, module(x, 1)))
        self.assertIsNone(module._p_conflict_weights)

    def test_enabled_capture_preserves_three_task_training_parameters_and_rng(self):
        outcomes = []
        for enabled in (False, True):
            torch.manual_seed(1993)
            module = fixtures.PrivateRankTests().make_module(dual_mask_p_conflict_diagnostics=enabled)
            anchor = module.pretrained_weight.clone()
            for task in range(3):
                module.before_task(task)
                module.set_task_and_stage(task, 0)
                x = torch.randn(3, 4, 4)
                optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.1)
                for _ in range(2):
                    optimizer.zero_grad()
                    loss = (module(x, task) - 1).square().mean()
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    optimizer.step()
                module.eval()
                before = module(x, task).detach()
                module.after_task(task)
                torch.testing.assert_close(before, module(x, task), atol=1e-6, rtol=1e-5)
                self.assertTrue(torch.equal(anchor, module.pretrained_weight))
            state = {k: v.clone() for k, v in module.state_dict().items() if not k.startswith('p_conflict_components.')}
            outcomes.append((state, torch.get_rng_state()))
            if enabled:
                self.require_feature(module)
                self.assertGreater(module.p_conflict_components[2].delta.numel(), 0)
        self.assertTrue(torch.equal(outcomes[0][1], outcomes[1][1]))
        self.assertEqual(outcomes[0][0].keys(), outcomes[1][0].keys())
        for key in outcomes[0][0]:
            self.assertTrue(torch.equal(outcomes[0][0][key], outcomes[1][0][key]), key)

    def test_saved_bank_roundtrip_and_dtype_conversion(self):
        module = merged_module()
        self.require_feature(module)
        restored = make_module()
        restored.load_state_dict(module.state_dict())
        self.assertTrue(torch.equal(restored.p_conflict_components[1].delta, module.p_conflict_components[1].delta))
        self.assertEqual(restored.double().p_conflict_components[1].delta.dtype, torch.float64)

    def test_zero_conflict_and_disabled_private_gate_store_zero(self):
        for kwargs in ({'dual_mask_conflict_ratio': 0.}, {'dual_mask_private_conflict_mode': 'none'},
                       {'dual_mask_conflict_strength': 1.}):
            module = make_module(**kwargs)
            self.require_feature(module)
            module.before_task(1)
            with torch.no_grad():
                module.P_lora[1].B_weight.fill_(1)
            module.after_task(1)
            self.assertEqual(module.p_conflict_components[1].delta.count_nonzero().item(), 0)


def make_network():
    from models.network import MANet, ViT
    net = MANet.__new__(MANet)
    nn.Module.__init__(net)
    net.numtask, net.class_num = 3, 2
    net.image_encoder = ViT(img_size=8, patch_size=4, embed_dim=4, depth=1, num_heads=1, rank=4, n_tasks=3)
    net.image_encoder.blocks[0].attn = merged_module()
    net.classifier_pool = nn.ModuleList([nn.Linear(4, 2, bias=False) for _ in range(3)])
    return net.eval()


class PConflictNetworkTests(unittest.TestCase):
    def require_feature(self):
        self.assertTrue(hasattr(make_module(), 'p_conflict_components'), 'P-conflict diagnostics not implemented')

    def test_twelve_modes_all_classes_soft_weights_and_no_label_leakage(self):
        self.require_feature()
        from utils.p_conflict_diagnostics import diagnostic_logits, conflict_weights
        torch.manual_seed(29)
        net = make_network()
        x = torch.randn(3, 3, 8, 8)
        baseline = net.interface(x)
        outputs, weights = diagnostic_logits(net, x, 20., torch.tensor([0, 1, 2]))
        self.assertTrue(torch.equal(outputs['ones'], baseline))
        self.assertEqual(set(outputs), {'ones', 'uniform', 'soft', 'conservative', 'predicted_onehot',
                                        'conditional_onehot', 'conditional_blend', 'conditional_oracle',
                                        'high_confidence_oracle', 'top2_counterfactual',
                                        'union_counterfactual', 'union_delta_margin', 'union_own_gain',
                                        'union_top_class_gain',
                                        'top2_task_oracle', 'union_task_oracle', 'oracle'})
        expected = (20 * baseline).softmax(1).reshape(3, 3, 2).sum(2)
        torch.testing.assert_close(weights['soft'], expected)
        torch.testing.assert_close(weights['uniform'], torch.full((3, 3), 1/3))
        self.assertTrue(torch.all(weights['conservative'] >= weights['soft']))
        self.assertTrue(torch.all(weights['conservative'] <= 1))
        self.assertTrue(torch.equal(weights['oracle'], torch.eye(3)))
        for mode, logits in outputs.items():
            self.assertEqual(logits.shape, (3, 6))
            with conflict_weights(net, weights[mode]):
                torch.testing.assert_close(logits, net.interface(x))
        alternate, _ = diagnostic_logits(net, x, 20., torch.tensor([2, 0, 1]))
        for mode in ('ones', 'uniform', 'soft', 'conservative', 'predicted_onehot',
                     'conditional_onehot', 'conditional_blend', 'top2_counterfactual',
                     'union_counterfactual', 'union_delta_margin', 'union_own_gain',
                     'union_top_class_gain'):
            self.assertTrue(torch.equal(outputs[mode], alternate[mode]))
        self.assertFalse(torch.equal(outputs['oracle'], alternate['oracle']))
        self.assertTrue(torch.equal(net.interface(x), baseline))

    def test_conservative_weights_use_entropy_without_a_tuned_threshold(self):
        from utils.p_conflict_diagnostics import conservative_weights
        flat = torch.full((2, 4), .25)
        torch.testing.assert_close(conservative_weights(flat), torch.ones_like(flat))
        sharp = torch.eye(4)
        torch.testing.assert_close(conservative_weights(sharp), sharp)
        probs = torch.tensor([[.7, .2, .1]])
        weights = conservative_weights(probs)
        self.assertTrue(torch.all(weights > probs))
        self.assertTrue(torch.all(weights < 1))

    def test_task0_twelve_modes_identical(self):
        self.require_feature()
        from utils.p_conflict_diagnostics import diagnostic_logits
        net = make_network()
        net.numtask = 1
        net.image_encoder.blocks[0].attn = make_module().eval()
        x = torch.randn(2, 3, 8, 8)
        outputs, _ = diagnostic_logits(net, x, 20., torch.zeros(2, dtype=torch.long))
        for logits in outputs.values():
            self.assertTrue(torch.equal(logits, net.interface(x)))

    def test_evaluation_restores_rng_modes_and_reports_corrected_broken(self):
        self.require_feature()
        from utils.p_conflict_diagnostics import evaluate_p_conflict
        torch.manual_seed(19)
        net = make_network()
        net.train()
        net.classifier_pool[0].eval()
        data = TensorDataset(torch.arange(6), torch.randn(6, 3, 8, 8), torch.arange(6))
        loader = DataLoader(data, batch_size=2)
        modes = [m.training for m in net.modules()]
        state = copy.deepcopy(net.state_dict())
        rng = torch.get_rng_state().clone()
        py_rng, np_rng = random.getstate(), np.random.get_state()
        report = evaluate_p_conflict(net, loader, torch.device('cpu'), 20.)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(py_rng, random.getstate())
        self.assertTrue(np.array_equal(np_rng[1], np.random.get_state()[1]))
        self.assertEqual(modes, [m.training for m in net.modules()])
        for key, value in state.items():
            self.assertTrue(torch.equal(value, net.state_dict()[key]), key)
        self.assertEqual(report['ones']['corrected'], 0)
        self.assertEqual(report['ones']['broken'], 0)
        for mode in report:
            self.assertEqual(sum(map(sum, report[mode]['task_confusion_counts'])), 6)
            self.assertAlmostEqual(report[mode]['total'] - report['ones']['total'],
                                   (report[mode]['corrected'] - report[mode]['broken']) * 100 / 6)
        json.dumps(report)

    def test_learner_logs_post_ca_diagnostics_without_changing_normal_result(self):
        from methods.base import BaseLearner
        from methods.dlora import Learner
        learner = Learner.__new__(Learner)
        learner._network = make_network()
        learner._device = torch.device('cpu')
        learner._cur_task = 2
        learner.args = {'dual_mask_p_conflict_diagnostics': True}
        learner.scale = 20.
        learner.test_loader = DataLoader(TensorDataset(
            torch.arange(6), torch.randn(6, 3, 8, 8), torch.arange(6)), batch_size=2)
        result = ({'grouped': {'total': 75.}, 'top1': 75.}, None, None, None)
        with patch.object(BaseLearner, 'eval_task', return_value=result), self.assertLogs(level='INFO') as logs:
            self.assertIs(learner.eval_task(), result)
        self.assertIn('P-conflict diagnostic Task 2', '\n'.join(logs.output))
        self.assertIn('"oracle_only": true', '\n'.join(logs.output))

    def test_summary_uses_same_average_and_old_task_forgetting_definition(self):
        from utils.p_conflict_diagnostics import summarize_p_conflict
        reports = [
            {'ones': {'total': 96., 'per_task': [96.]}},
            {'ones': {'total': 90., 'per_task': [92., 88.]}},
            {'ones': {'total': 86., 'per_task': [83., 80., 95.]}},
        ]
        summary = summarize_p_conflict(reports)
        self.assertEqual(summary['ones']['average'], (96. + 90. + 86.) / 3)
        self.assertEqual(summary['ones']['forgetting'], ((96. - 83.) + (88. - 80.)) / 2)
        self.assertIsNone(summarize_p_conflict(reports[-1:])['ones']['forgetting'])

    def test_real_learner_trains_three_tasks_and_reports_twelve_modes(self):
        from methods.dlora import Learner
        from models.network import ViT
        from utils.p_conflict_diagnostics import evaluate_p_conflict
        args = json.loads(Path('exps/dlora/imgr10.json').read_text())
        args.update(device=[torch.device('cpu')], embd_dim=8, init_cls=2, increment=2,
                    total_sessions=3, rank=2, num_heads=2, init_epoch=2, epochs=2,
                    num_workers=0, dual_mask_svd_rank=2, dual_mask_competence_adaptive=False,
                    dual_mask_reg_weight=.01, dual_mask_conflict_reg_enabled=False,
                    dual_mask_lora_inherit=False, dual_mask_p_conflict_diagnostics=True)
        encoder = ViT(img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2, rank=2, n_tasks=3)
        with patch('models.network._create_vision_transformer', return_value=encoder):
            learner = Learner(args)
        module = next(learner._iter_lora_modules())
        for task in range(3):
            learner._cur_task = task
            learner._known_classes, learner._total_classes = task * 2, (task + 1) * 2
            learner._network.numtask = task + 1
            data = TensorDataset(torch.arange(4), torch.randn(4, 3, 8, 8),
                                 torch.tensor([0, 1, 0, 1]) + task * 2)
            loader = DataLoader(data, batch_size=4, num_workers=0)
            learner._train(loader, loader)
            self.assertIsNone(module.P_lora[task])
            self.assertTrue(torch.isfinite(module.qkv.weight).all())
        self.assertGreater(module.p_conflict_components[2].delta.numel(), 0)
        test = TensorDataset(torch.arange(6), torch.randn(6, 3, 8, 8), torch.arange(6))
        report = evaluate_p_conflict(learner._network, DataLoader(test, batch_size=3),
                                     torch.device('cpu'), learner.scale)
        self.assertEqual(set(report), {'ones', 'uniform', 'soft', 'conservative', 'predicted_onehot',
                                       'conditional_onehot', 'conditional_blend', 'conditional_oracle',
                                       'high_confidence_oracle', 'top2_counterfactual',
                                       'union_counterfactual', 'union_delta_margin', 'union_own_gain',
                                       'union_top_class_gain',
                                       'top2_task_oracle', 'union_task_oracle', 'oracle'})
        self.assertEqual(sum(map(sum, report['ones']['task_confusion_counts'])), 6)


class FirstRoundScriptTests(unittest.TestCase):
    def test_dry_run_is_one_from_scratch_task0_to2_command_without_data_override(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/9_14_p_conflict_first_round.sh'
        result = subprocess.run(['bash', str(script)], env={'DRY_RUN': '1', 'PATH': '/usr/bin:/bin'},
                                capture_output=True, text=True, check=True)
        command = shlex.split(result.stdout)
        self.assertEqual(command[:4], ['python', 'main.py', '--config', 'exps/dlora/imgr10.json'])
        overrides = [command[i + 1] for i, value in enumerate(command[:-1]) if value == '--set']
        self.assertIn('max_tasks=3', overrides)
        self.assertIn('ca_epochs=5', overrides)
        self.assertIn('dual_mask_p_conflict_diagnostics=true', overrides)
        self.assertIn('dual_mask_lora_inherit=false', overrides)
        self.assertIn('seed=[1993]', overrides)
        self.assertIn('task0_checkpoint_resume=', overrides)
        self.assertIn('task0_checkpoint_save=', overrides)
        self.assertFalse(any(value.startswith('data_path=') for value in overrides))
        self.assertEqual(command.count('main.py'), 1)


if __name__ == '__main__':
    unittest.main()
