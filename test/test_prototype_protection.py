"""原型位置评分、状态恢复和四位置有效更新的行为测试。"""
import copy
import json
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from test.test_tail_update import TailUpdateTests


class TinyData:
    def __init__(self):
        self.labels = np.array([0, 1, 2, 3])
        self.inputs = torch.tensor([[1., 2., -1., .5], [-2., 1., .5, 1.],
                                    [1., -.5, 2., 1.], [.2, 1., -2., .7]], dtype=torch.double)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, self.inputs[index], int(self.labels[index])


class TinyEncoder(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.attention = module
        self.dropout = nn.Dropout(.3)

    def extract_vector(self, inputs):
        torch.rand(1)
        np.random.rand()
        random.random()
        return self.dropout(self.attention.qkv(inputs) +
                            self.attention._contrib_from_units(inputs, self.attention.cur_task))

    def interface(self, inputs):
        return self.extract_vector(inputs)[:, :4]


class PrototypeProtectionTests(unittest.TestCase):
    def test_partition_round_robin_and_private_derangement(self):
        from utils.prototype_protection import probe_indices, shuffled_prototypes
        labels = np.array([2] * 6 + [3] * 6)
        fit, holdout = probe_indices(labels, fit_limit=6, holdout_limit=4)
        self.assertEqual(fit, [1, 7, 2, 8, 3, 9])
        self.assertEqual(holdout, [0, 6, 5, 11])
        before = torch.get_rng_state().clone()
        prototypes = torch.eye(6)
        first, permutation = shuffled_prototypes(prototypes, 1993, 1)
        second, other = shuffled_prototypes(prototypes, 1993, 1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(permutation, other))
        self.assertTrue(torch.all(permutation != torch.arange(6)))
        self.assertEqual(sorted(permutation.tolist()), list(range(6)))

    def test_per_sample_squared_gradients_and_full_state_restore(self):
        from utils.prototype_protection import estimate_scores
        torch.manual_seed(5)
        module = TailUpdateTests().make('step').double()
        module.qkv.weight.requires_grad_(False)
        module.qkv.weight.grad = torch.ones_like(module.qkv.weight)
        with torch.no_grad():
            module.qkv.weight.add_(.3)
        network = TinyEncoder(module).train()
        network.dropout.eval()
        learner = SimpleNamespace(_network=network, _device=torch.device('cpu'),
            _pretrained_anchor_context=module.use_pretrained_anchor)
        dataset = TinyData()
        prototypes = torch.eye(12, dtype=torch.double)[:4].requires_grad_()
        shuffled = prototypes[[1, 2, 3, 0]]
        expected, average_grad = [], []
        for bank in (prototypes.detach(), shuffled.detach()):
            gradients = []
            weight = module.pretrained_weight.detach().clone().requires_grad_()
            for index in range(4):
                features = F.linear(dataset.inputs[index:index + 1], weight, module.qkv.bias)
                logits = F.normalize(features, dim=1) @ bank.t()
                competitor = logits.detach().clone()
                competitor[0, index] = -torch.inf
                negative = int(competitor.argmax(1))
                margin = logits[0, index] - logits[0, negative]
                gradients.append(torch.autograd.grad(margin, weight)[0])
            expected.append(torch.stack(gradients).square().mean(0))
            average_grad.append(torch.stack(gradients).mean(0).square())
        state = {k: v.clone() for k, v in network.state_dict().items()}
        flags = [p.requires_grad for p in network.parameters()]
        modes = [m.training for m in network.modules()]
        torch_state = torch.get_rng_state().clone()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        scores, diagnostics = estimate_scores(learner, [module], dataset, list(range(4)),
                                              prototypes, shuffled)
        for actual, reference in zip((scores[0][0], scores[1][0]), expected):
            torch.testing.assert_close(actual, reference, atol=1e-8, rtol=1e-6)
            self.assertFalse(actual.requires_grad)
        self.assertFalse(torch.allclose(expected[0], average_grad[0]))
        self.assertEqual(diagnostics['samples'], 4)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
        self.assertEqual(modes, [m.training for m in network.modules()])
        self.assertEqual(flags, [p.requires_grad for p in network.parameters()])
        self.assertTrue(torch.equal(module.qkv.weight.grad, torch.ones_like(module.qkv.weight)))
        self.assertIsNone(prototypes.grad)
        for name, value in state.items():
            torch.testing.assert_close(value, network.state_dict()[name], atol=0, rtol=0)

    def test_exact_counts_ties_low_is_not_complement(self):
        from utils.prototype_protection import ranked_mask
        reference = torch.tensor([[1., 1.], [1., 0.]] * 3)
        score = torch.tensor([[4., 3.], [2., 1.]] * 3)
        high = ranked_mask(score, reference)
        low = ranked_mask(score, reference, lowest=True)
        self.assertEqual(high.tolist(), [[True, True], [True, False]] * 3)
        self.assertEqual(low.tolist(), [[False, True], [True, True]] * 3)
        self.assertFalse(torch.equal(low, ~high))
        for zeros in (torch.zeros_like(score), torch.ones_like(score)):
            mask = ranked_mask(zeros, reference)
            self.assertTrue(torch.equal(mask, reference.bool()))
        for source, chosen in zip(reference.chunk(3), low.chunk(3)):
            self.assertEqual(int(source.sum()), int(chosen.sum()))

    def make_bank(self, task=1):
        module = TailUpdateTests().make('step', task).double()
        reference = module.general_mask.bool()
        masks = torch.stack([reference, reference.flip(0), reference.flip(1),
                             reference.flip((0, 1)), reference.roll(1, 1)])
        module.prototype_position_masks = masks
        module.args.update(dual_mask_prototype_position_probe=True,
                           dual_mask_position_norm_match='prototype_min')
        return module

    def test_four_same_state_norms_zero_grad_and_permissions(self):
        module = self.make_bank()
        value = torch.linspace(-2, 3, 48, dtype=torch.double).reshape(12, 4).requires_grad_()
        for isolated in (False, True):
            norms = []
            for position, index in (('wpre', 0), ('prototype_high', 1),
                                    ('prototype_shuffled', 2), ('permuted', 4)):
                module.args['dual_mask_protect_position'] = position
                safe = module._safe_delta(value, isolated)
                norms.append(safe.reshape(3, -1).norm(dim=1))
                self.assertTrue(torch.all(safe.abs() <= value.abs() + 1e-12))
                if isolated:
                    self.assertEqual(int(safe[module.prototype_position_masks[index]].count_nonzero()), 0)
            for actual in norms[1:]:
                torch.testing.assert_close(actual, norms[0])
        zero = torch.zeros_like(value, requires_grad=True)
        module.args['dual_mask_protect_position'] = 'wpre'
        module._safe_delta(zero, False).sum().backward()
        self.assertGreater(float(zero.grad.norm()), 0)
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_all_positions_forward_merge_once_and_task0(self):
        for position in ('wpre', 'prototype_high', 'prototype_shuffled', 'permuted'):
            module = self.make_bank()
            module.args['dual_mask_protect_position'] = position
            with torch.no_grad():
                for unit in (module.S_lora[1], module.P_lora[1]):
                    unit.B_weight.normal_(0, .1)
            x = torch.randn(3, 4, dtype=torch.double)
            before = module.qkv(x) + module._contrib_from_units(x, 1)
            module.after_task(1)
            torch.testing.assert_close(before, module.qkv(x), atol=1e-9, rtol=1e-9)
            weight = module.qkv.weight.clone()
            module.after_task(1)
            torch.testing.assert_close(weight, module.qkv.weight, atol=0, rtol=0)
        reference = TailUpdateTests().make('step', 0).double()
        changed = copy.deepcopy(reference)
        changed.args.update(dual_mask_position_norm_match='prototype_min',
                            dual_mask_protect_position='prototype_high')
        value = torch.randn_like(reference.qkv.weight)
        torch.testing.assert_close(reference._safe_delta(value, False),
                                   changed._safe_delta(value, False), atol=0, rtol=0)

    def test_merge_records_do_not_mislabel_norm_control_as_conflict_only(self):
        module = self.make_bank()
        module.dual_mask_applied_budget_log = True
        with self.assertLogs(level='INFO') as logs:
            module.after_task(1)
        rows = [json.loads(line.split('AppliedConflictBudget ', 1)[1])
                for line in logs.output if 'AppliedConflictBudget ' in line]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['removed_norm_definition'] == 'post_permission_conflict_and_norm_control' for r in rows))

    def test_install_uses_original_spectrum_even_for_permuted_and_preserves_next_step(self):
        from utils.prototype_protection import prepare_positions
        module = self.make_bank()
        module.args.update(seed=1993, dual_mask_protect_position='permuted')
        module.core_reference_protect = module.general_mask.clone()
        module.general_mask.copy_(module.general_mask.flip(1))
        dataset = TinyData()
        dataset.labels = np.array([0, 0, 1, 1])
        network = TinyEncoder(module).train()
        network.dropout.eval()
        learner = SimpleNamespace(_network=network, _device=torch.device('cpu'), _cur_task=1,
            _total_classes=2, args=module.args, _iter_lora_modules=lambda: [module],
            _pretrained_anchor_context=module.use_pretrained_anchor,
            w0_loader=SimpleNamespace(dataset=dataset),
            _w0_class_means={0: torch.eye(12)[0], 1: torch.eye(12)[1]})
        state = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO'):
            prepare_positions(learner)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(module.prototype_position_masks[0], module.core_reference_protect.bool()))
        self.assertTrue(torch.equal(module.general_mask.bool(), module.prototype_position_masks[4]))
        self.assertTrue(torch.equal(module.isolated_mask, 1 - module.general_mask))
        self.assertIsNone(module.qkv.weight.grad)
        reference = copy.deepcopy(network)
        x = torch.randn(3, 4, dtype=torch.double)
        rng = torch.get_rng_state().clone()
        first = network.extract_vector(x)
        torch.set_rng_state(rng)
        second = reference.extract_vector(x)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        first.sum().backward()
        second.sum().backward()
        torch.testing.assert_close(module.S_lora[1].B_weight.grad, reference.attention.S_lora[1].B_weight.grad)

    def test_equal_norm_diagnostic_restores_next_training_step(self):
        from utils.prototype_protection import diagnose_positions
        module = self.make_bank()
        with torch.no_grad():
            module.S_lora[1].B_weight.normal_(0, .1)
            module.P_lora[1].B_weight.normal_(0, .1)
        network = TinyEncoder(module).train()
        reference = copy.deepcopy(network)
        learner = SimpleNamespace(_network=network, _device=torch.device('cpu'), _cur_task=1,
            _known_classes=2, _total_classes=4, batch_size=2, _iter_lora_modules=lambda: [module])
        state = torch.get_rng_state().clone()
        with self.assertLogs(level='INFO') as logs:
            diagnose_positions(learner, TinyData(), 20)
        self.assertEqual(sum('PrototypePositionDiagnostic ' in line for line in logs.output), 8)
        self.assertEqual(sum('PrototypePositionNorm ' in line for line in logs.output), 24)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual([m.training for m in network.modules()], [m.training for m in reference.modules()])
        self.assertIsNone(module._prototype_audit_position)
        x = torch.randn(3, 4, dtype=torch.double)
        rng = torch.get_rng_state().clone()
        first = network.extract_vector(x)
        torch.set_rng_state(rng)
        second = reference.extract_vector(x)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        first.sum().backward()
        second.sum().backward()
        torch.testing.assert_close(module.P_lora[1].B_weight.grad, reference.attention.P_lora[1].B_weight.grad)


class PrototypePositionQueueTests(unittest.TestCase):
    def test_missing_invalid_or_duplicate_position_evidence_is_incomplete(self):
        import sys
        sys.path.insert(0, 'scripts')
        from analyze_prototype_position import position_evidence_complete
        positions = ('wpre', 'prototype_high', 'prototype_shuffled', 'permuted')
        telemetry = dict(
            norms=[dict(task=t, layer=l, branch=b, projection=p, position=pos,
                        raw_norm=2., effective_norm=1., removed_norm=1., same_state_norm_residual=0.)
                   for t in (1, 5, 9) for l in range(12) for b in ('S', 'P')
                   for p in ('Q', 'K', 'V') for pos in positions],
            holdout=[dict(task=t, layer=l, position=pos, samples=64,
                          true_sensitive_mass=.5, shuffled_sensitive_mass=.4)
                     for t in range(1, 10) for l in range(12)
                     for pos in positions + ('prototype_low',)],
            diagnostics=[dict(task=t, partition=part, position=pos, count=128,
                              mean_margin=.1, norm_control='same_state_four_position_min')
                         for t in (1, 5, 9) for part in ('old', 'new') for pos in positions])
        self.assertTrue(position_evidence_complete(telemetry))
        for section in ('norms', 'holdout', 'diagnostics'):
            missing = copy.deepcopy(telemetry)
            missing[section].pop()
            self.assertFalse(position_evidence_complete(missing))
            duplicate = copy.deepcopy(telemetry)
            duplicate[section][-1] = duplicate[section][0]
            self.assertFalse(position_evidence_complete(duplicate))
        for section, field, value in (('norms', 'same_state_norm_residual', .01),
                                      ('holdout', 'true_sensitive_mass', float('nan')),
                                      ('diagnostics', 'mean_margin', float('inf')),
                                      ('diagnostics', 'count', 0)):
            invalid = copy.deepcopy(telemetry)
            invalid[section][0][field] = value
            self.assertFalse(position_evidence_complete(invalid))

    def test_fixed_queues_and_recipe_validation(self):
        import sys
        sys.path.insert(0, 'scripts')
        import run_tail_update as runner
        previous = runner.SPEC
        try:
            runner.SPEC = runner.ROOT / 'scripts/sweeps/imgr10_prototype_position.json'
            for machine, prefix in (('3090', 'R'), ('5090', 'S')):
                runner.validate_settings(machine)
                names = runner.modes(machine)
                self.assertEqual(names, [prefix + str(i) + '_seed1993' for i in range(4)])
                configs = [runner.settings_for(machine, name) for name in names]
                for config in configs:
                    self.assertEqual(config['dual_mask_private_rank'], 64)
                    self.assertEqual(config['dual_mask_gradient_route'], 'off')
                    self.assertTrue(config['dual_mask_prototype_position_probe'])
                    self.assertEqual(config['dual_mask_fixed_coverage'], .9)
                    self.assertEqual(config['dual_mask_position_norm_match'], 'off' if machine == '3090' else 'prototype_min')
                for config in configs:
                    config.pop('dual_mask_protect_position')
                self.assertTrue(all(config == configs[0] for config in configs))
        finally:
            runner.SPEC = previous

    def test_unsupported_combinations_rejected_before_training(self):
        from scripts.dualmask_config import normalize_dualmask_config
        good = dict(dual_mask_prototype_position_probe=True, dual_mask_protect_position='prototype_high',
                    dual_mask_gradient_route='off', dual_mask_position_norm_match='prototype_min',
                    dual_mask_conflict_score_mode='magnitude', dual_mask_conflict_exact_topk=True,
                    dual_mask_update_rule='step', dual_mask_competence_holdout_mod=5)
        normalize_dualmask_config(good.copy())
        for changed in (dict(dual_mask_prototype_position_probe=False), dict(dual_mask_gradient_route='prototype'),
                        dict(dual_mask_position_norm_match='paired_min'), dict(dual_mask_update_rule='soft_tail'),
                        dict(dual_mask_competence_holdout_mod=3)):
            with self.assertRaises(ValueError):
                normalize_dualmask_config(dict(good, **changed))

    def test_missing_or_mismatched_results_never_produce_pairs(self):
        import sys
        sys.path.insert(0, 'scripts')
        from analyze_prototype_position import paired_results
        from analyze_core_evidence import METRICS
        rows = [dict(mode='R0_seed1993', valid_performance=True, **{k: 80. for k in METRICS}),
                dict(mode='R1_seed1993', valid_performance=False, **{k: None for k in METRICS})]
        self.assertEqual(paired_results(rows, [], '3090'), [])
        rows[1].update(valid_performance=True, **{k: 80.1 for k in METRICS})
        self.assertAlmostEqual(paired_results(rows, [], '3090')[0]['Average'], .1)
        self.assertEqual(paired_results(rows, ['source_mismatch'], '3090'), [])


if __name__ == '__main__':
    unittest.main()
