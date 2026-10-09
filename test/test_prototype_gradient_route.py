"""原型分工只替换当前 B 的分类梯度，不改变前向和分类头。"""
import copy
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.losses import AngularPenaltySMLoss
from test.test_pair_separation import backward_hook
from test import test_tail_update as tail_tests


class PrototypeAssignmentTests(unittest.TestCase):
    def test_all_seen_classes_global_ids_and_detachment(self):
        from utils.prototype_gradient_route import assign_samples
        features = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], requires_grad=True)
        prototypes = torch.eye(2, requires_grad=True)
        route, correct = assign_samples(features, prototypes, torch.tensor([2, 7]),
                                        torch.tensor([2, 7, 7]), 'prototype')
        self.assertEqual(route.tolist(), [True, True, False])
        self.assertTrue(torch.equal(route, correct))
        self.assertFalse(route.requires_grad)
        self.assertIsNone(features.grad)
        self.assertIsNone(prototypes.grad)

    def test_random_matched_is_private_reproducible_and_not_balanced(self):
        from utils.prototype_gradient_route import assign_samples
        features, prototypes = torch.eye(2).repeat(4, 1), torch.eye(2)
        labels = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1])
        rng = torch.get_rng_state().clone()
        results = [assign_samples(features, prototypes, torch.arange(2), labels,
                   'random_matched', torch.Generator().manual_seed(123)) for _ in range(2)]
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertEqual(int(results[0][0].sum()), int(results[0][1].sum()))
        self.assertFalse(torch.equal(results[0][0], results[0][1]))
        for labels in (torch.tensor([0, 1] * 4), torch.tensor([1, 0] * 4)):
            route, correct = assign_samples(features, prototypes, torch.arange(2), labels,
                'random_matched', torch.Generator().manual_seed(4))
            self.assertTrue(torch.equal(route, correct))

    def test_probe_restores_modes_rng_and_uses_local_label_offset(self):
        from utils.prototype_gradient_route import prepare_batch

        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout()
                self.weight = nn.Parameter(torch.eye(2))

            def forward(self, x):
                # 即便探针内部用随机数，也不应改变下一次学生前向。
                torch.rand(3)
                return {'features': self.dropout(x) @ self.weight}

        network = Probe().train()
        network.dropout.eval()
        learner = SimpleNamespace(args={'seed': 1993, 'dual_mask_gradient_route': 'prototype'},
            _cur_task=1, _known_classes=2, _total_classes=4, _network=network,
            _w0_class_means={0: torch.tensor([-1., 0.]), 1: torch.tensor([0., -1.]),
                            2: torch.tensor([1., 0.]), 3: torch.tensor([0., 1.])},
            _pretrained_anchor_context=nullcontext)
        modes = [m.training for m in network.modules()]
        rng = torch.get_rng_state().clone()
        context = prepare_batch(learner, torch.eye(2), torch.tensor([0, 1]))
        self.assertEqual(context['gradient_route_mask'].tolist(), [True, True])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in network.modules()])
        self.assertIsNone(network.weight.grad)
        learner._cur_task = 0
        self.assertEqual(prepare_batch(learner, torch.eye(2), torch.tensor([0, 1])), {})
        learner._cur_task = 1
        learner.args['dual_mask_gradient_route'] = 'off'
        self.assertEqual(prepare_batch(learner, torch.eye(2), torch.tensor([0, 1])), {})


class PrototypeGradientTests(unittest.TestCase):
    def exercise(self, mask, mode='prototype', task=1, zero_b=False):
        torch.manual_seed(9)
        module = tail_tests.TailUpdateTests().make('step', task).double()
        with torch.no_grad():
            for unit in (module.S_lora[task], module.P_lora[task]):
                if unit is not None and not zero_b:
                    unit.B_weight.normal_(0, .1)
        head = nn.Linear(12, 3, bias=False).double()
        reference, reference_head = copy.deepcopy(module), copy.deepcopy(head)
        x = torch.randn(4, 4, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0])
        criterion = AngularPenaltySMLoss(loss_type='cosface', s=5., m=.1)
        logits = head(module.qkv(x) + module._contrib_from_units(x, task))
        ref_x = x.detach().clone().requires_grad_()
        ref_logits = reference_head(reference.qkv(ref_x) + reference._contrib_from_units(ref_x, task))
        torch.testing.assert_close(logits, ref_logits, atol=0, rtol=0)
        samples = criterion(logits, labels, return_type='per_sample')
        full = criterion(logits, labels)
        ref_samples = criterion(ref_logits, labels, return_type='per_sample')
        named = [(b, u.B_weight) for b, u in (('S', reference.S_lora[task]),
                  ('P', reference.P_lora[task])) if u is not None and u.B_weight.requires_grad]
        expected = []
        for branch, param in named:
            # 明确逐样本参考；分母始终是完整批次4，而非组内人数。
            grad = torch.zeros_like(param)
            for i in range(4):
                selected = mode in ('off', 'all') or task == 0 or bool(mask[i]) == (branch == 'S')
                if selected:
                    grad += torch.autograd.grad(ref_samples[i] / 4, param, retain_graph=True)[0]
            expected.append(grad)
        criterion(ref_logits, labels).backward()
        output = {'gradient_route_mask': mask, 'gradient_route_correct': mask,
                  'gradient_route_per_sample': samples, 'gradient_route_probe_ms': 0.,
                  'gradient_route_location': (1, 1)} if mode != 'off' and task else {}
        learner = SimpleNamespace(args={'dual_mask_gradient_route': mode}, _cur_task=task,
                                  _iter_lora_modules=lambda: [module])
        parameters = list(module.parameters()) + list(head.parameters())
        optimizer = torch.optim.SGD(parameters, lr=.01, momentum=.9)
        rng = torch.get_rng_state().clone()
        backward_hook()(learner, full, None, optimizer, output, labels)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(head.weight.grad, reference_head.weight.grad, atol=0, rtol=0)
        torch.testing.assert_close(x.grad, ref_x.grad, atol=0, rtol=0)
        for (_, expected_param), expected_grad in zip(named, expected):
            branch = next(b for b, p in named if p is expected_param)
            actual = getattr(module, branch + '_lora')[task]
            torch.testing.assert_close(actual.B_weight.grad, expected_grad, atol=1e-10, rtol=1e-10)
            if task > 0:
                self.assertIsNone(actual.A_weight.grad)
        # 分配不能改变门控和交付函数，且只能合并一次。
        query = torch.randn(3, 4, dtype=torch.double)
        before = module.qkv(query) + module._contrib_from_units(query, task)
        module.after_task(task)
        torch.testing.assert_close(before, module.qkv(query), atol=1e-9, rtol=1e-9)
        weight = module.qkv.weight.detach().clone()
        module.after_task(task)
        torch.testing.assert_close(weight, module.qkv.weight, atol=0, rtol=0)

    def test_mixed_all_correct_all_wrong_and_zero_initialization(self):
        for mask in (torch.tensor([1, 0, 1, 0], dtype=torch.bool),
                     torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)):
            for zero in (False, True):
                self.exercise(mask, zero_b=zero)

    def test_off_all_and_task0_preserve_ordinary_gradients(self):
        for mode, task in (('off', 1), ('all', 1), ('prototype', 0), ('random_matched', 0)):
            self.exercise(torch.tensor([1, 0, 1, 0], dtype=torch.bool), mode, task)


class PrototypeQueueTests(unittest.TestCase):
    def test_three_recipes_only_differ_in_route(self):
        import sys
        sys.path.insert(0, 'scripts')
        import run_tail_update as runner
        previous = runner.SPEC
        try:
            runner.SPEC = runner.ROOT / 'scripts/sweeps/imgr10_prototype_gradient_route_3090.json'
            runner.validate_settings('3090')
            self.assertEqual(runner.modes('3090'), ['A_seed1993', 'B_seed1993', 'C_seed1993'])
            configs = [runner.settings_for('3090', name) for name in runner.modes('3090')]
            self.assertEqual([c.pop('dual_mask_gradient_route') for c in configs],
                             ['all', 'prototype', 'random_matched'])
            self.assertEqual(configs[0], configs[1])
            self.assertEqual(configs[0], configs[2])
            self.assertEqual(configs[0]['dual_mask_fixed_coverage'], .9)
            self.assertEqual(configs[0]['dual_mask_private_rank'], 64)
            old = json.loads(Path('scripts/sweeps/imgr10_m_cov90_rank64_3090.json').read_text())
            for key, value in dict(old['common_overrides'], **old['compact_overrides']).items():
                self.assertEqual(configs[0][key], value)
        finally:
            runner.SPEC = previous

    def test_entry_rejects_unknown_or_single_route(self):
        from scripts.dualmask_config import normalize_dualmask_config
        for values in ({'dual_mask_gradient_route': 'unknown'},
                       {'dual_mask_gradient_route': 'prototype', 'dual_mask_branch_layout': 'single'},
                       {'dual_mask_gradient_route': 'all', 'use_plora': False},
                       {'dual_mask_gradient_route': 'all', 'sp_staged_s_epochs': 1},
                       {'dual_mask_gradient_route': 'prototype', 'dual_mask_reg_weight': .01}):
            with self.assertRaises(ValueError):
                normalize_dualmask_config(values)
        config = {'dual_mask_gradient_route': 'off'}
        self.assertEqual(normalize_dualmask_config(config)['dual_mask_gradient_route'], 'off')

    def test_partial_summary_has_no_unavailable_comparison(self):
        from scripts.analyze_prototype_gradient_route import paired_results
        records = [{'mode': 'A_seed1993', 'valid_performance': True, 'Average': 87.},
                   {'mode': 'B_seed1993', 'valid_performance': False, 'Average': 90.}]
        self.assertEqual(paired_results(records, []), [])
        records[1]['valid_performance'] = True
        for row in records:
            row.update(Last=82., Old=81., New=88., StageOld=86., StageNew=87., Forgetting=6.)
        self.assertEqual([r['comparison'] for r in paired_results(records, [])], ['B_minus_A'])
        self.assertEqual(paired_results(records, [{'field': 'hardware'}]), [])

    def test_analysis_failure_does_not_abort_training_queue(self):
        import sys
        from unittest.mock import patch
        sys.path.insert(0, 'scripts')
        import run_tail_update as runner
        import analyze_prototype_gradient_route as analyzer
        previous = runner.SPEC
        try:
            runner.SPEC = runner.ROOT / 'scripts/sweeps/imgr10_prototype_gradient_route_3090.json'
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                with patch.object(analyzer, 'summarize_saved', side_effect=ValueError('analysis only')):
                    runner.summarize_safely(directory, '3090', [])
                self.assertIn('analysis only', (directory / 'analysis_errors.jsonl').read_text())
        finally:
            runner.SPEC = previous


if __name__ == '__main__':
    unittest.main()
