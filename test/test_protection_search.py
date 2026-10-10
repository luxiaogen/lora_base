"""保护强度规则的实际梯度、默认回归与同提交配对。"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_tail_update as runner
import analyze_protection_search as analyzer
import analyze_prototype_position as position
from dualmask_config import normalize_dualmask_config
from test import test_ncm_direct_strengths as fixtures
from utils.protection_strength import RULES, protection_update


class ProtectionSearchTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, runner, 'SPEC', runner.SPEC)
        runner.SPEC = ROOT / 'scripts/sweeps/imgr10_protection_search_3090.json'

    def settings(self, name):
        runner.SPEC = ROOT / 'scripts/sweeps/imgr10_protection_search_3090.json'
        return runner.settings_for('3090', name + '_seed1993')

    def module(self, name='REF', task=1):
        module = fixtures.NCMDirectStrengthTests().module()
        module.args.update(self.settings(name), seed=1993)
        module.set_pretrained_competence(.72, 0.)
        module.set_pretrained_old_overlap_risk(.08)
        module.before_task(task)
        module.general_mask.zero_()
        module.general_mask[::2] = 1.
        return module

    def test_fixed_single_seed_order_only_declared_factor_changes(self):
        expected = ['REF', 'A0', 'A1', 'A010', 'A090', 'ALL', 'RISK', 'UP', 'DOWN', 'E1', 'E05']
        self.assertEqual(runner.modes('3090'), [name + '_seed1993' for name in expected])
        reference = self.settings('REF')
        for name in expected:
            config = self.settings(name)
            difference = {key for key in config.keys() | reference.keys()
                          if config.get(key) != reference.get(key)}
            self.assertLessEqual(difference, {'dual_mask_protection_rule', 'dual_mask_fixed_protect_strength'})
            self.assertEqual(config['seed'], [1993])
        runner.validate_settings('3090')
        bad = dict(reference, dual_mask_protection_rule='unknown')
        with self.assertRaises(ValueError):
            normalize_dualmask_config(bad)
        for field, value in (('dual_mask_reg_weight', .01), ('dual_mask_update_rule', 'soft_tail'),
                             ('dual_mask_permission_mode', 'symmetric_soft')):
            with self.assertRaises(ValueError):
                normalize_dualmask_config(dict(self.settings('E1'), **{field: value}))

    def test_task0_all_rules_and_rng_match_static(self):
        reference = self.module('REF', task=0)
        x = torch.randn(3, 4)
        for rule in RULES:
            candidate = copy.deepcopy(reference)
            candidate.args['dual_mask_protection_rule'] = rule
            state = torch.get_rng_state().clone()
            torch.testing.assert_close(candidate._contrib_from_units(x, 0),
                                       reference._contrib_from_units(x, 0), rtol=0, atol=0)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_default_matches_previous_source_forward_loss_gradient_step_merge(self):
        source = subprocess.check_output(['git', 'show',
            '4eb069560ddd5b46235dbaf2fe405122c919e1b5:models/attention.py'], cwd=ROOT, text=True)
        namespace = {'__name__': 'previous_attention'}
        exec(compile(source, '<previous_attention>', 'exec'), namespace)
        current = self.module()
        with torch.no_grad():
            current.S_lora[1].B_weight.normal_(std=.01)
            current.P_lora[1].B_weight.normal_(std=.01)
        previous = copy.deepcopy(current)
        previous.__class__ = namespace[current.__class__.__name__]
        x = torch.randn(3, 4)
        optimizers = [torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.02)
                      for m in (previous, current)]
        values = [m._contrib_from_units(x, 1) for m in (previous, current)]
        torch.testing.assert_close(*values, rtol=0, atol=0)
        for value in values:
            value.square().mean().backward()
        for first, second in zip(previous.parameters(), current.parameters()):
            if first.grad is not None:
                torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
        for optimizer in optimizers:
            optimizer.step()
        for m in (previous, current):
            m.after_task(1)
        torch.testing.assert_close(previous.qkv.weight, current.qkv.weight, rtol=0, atol=0)

    def test_signal_and_schedule_values_and_p_unchanged(self):
        reference = self.module()
        raw = torch.arange(1, reference.qkv.weight.numel() + 1,
                           dtype=torch.float32).reshape_as(reference.qkv.weight)
        expected_p = reference._safe_delta(raw, True)
        expected = {'ALL': .64, 'RISK': .08, 'UP': .125, 'DOWN': .875}
        for name, alpha in expected.items():
            module = copy.deepcopy(reference)
            module.args.update(self.settings(name))
            module.protection_progress = .25
            state = protection_update(module, raw)
            self.assertAlmostEqual(float(state['alpha']), alpha)
            torch.testing.assert_close(module._safe_delta(raw, False), state['safe'])
            torch.testing.assert_close(module._safe_delta(raw, True), expected_p, rtol=0, atol=0)
            base, _ = module._merge_base_and_conflict(raw, False, .1, compute_conflict=False)
            torch.testing.assert_close(base, state['base'], rtol=0, atol=0)
        for name in ('UP', 'DOWN'):
            module = self.module(name)
            module.protection_progress = 1.
            torch.testing.assert_close(module._safe_delta(raw, False), reference._safe_delta(raw, False))

    def test_energy_rules_detach_alpha_and_enforce_post_conflict_budget(self):
        for name, ratio in (('E1', 1.), ('E05', .5)):
            module = self.module(name)
            raw = torch.full_like(module.qkv.weight, 1., requires_grad=True)
            with torch.no_grad():
                raw[module.general_mask.bool()] = 10.
            state = protection_update(module, raw)
            self.assertFalse(state['alpha'].requires_grad)
            protected = (state['safe'] * module.general_mask).norm()
            outside = (state['safe'] * (1 - module.general_mask)).norm()
            self.assertLessEqual(float(protected.detach()), ratio * float(outside.detach()) + 1e-5)
            self.assertTrue(torch.all(state['gate'] >= 0))
            self.assertTrue(torch.all(state['gate'] <= 1))
            state['safe'].sum().backward()
            torch.testing.assert_close(raw.grad, state['gate'], rtol=0, atol=0)
            for mask_value, raw_value, expected_alpha in ((0., 1., 0.), (1., 1., 1.), (1., 0., .5)):
                module.general_mask.fill_(mask_value)
                state = protection_update(module, torch.full_like(raw, raw_value))
                self.assertAlmostEqual(float(state['alpha']), expected_alpha)

    def test_all_rules_zero_b_gradient_and_forward_merge_once(self):
        for name in ('ALL', 'RISK', 'UP', 'DOWN', 'E1', 'E05'):
            module = self.module(name)
            x = torch.randn(3, 4)
            module._contrib_from_units(x, 1).sum().backward()
            for unit in (module.S_lora[1], module.P_lora[1]):
                self.assertTrue(torch.isfinite(unit.B_weight.grad).all())
                self.assertGreater(float(unit.B_weight.grad.norm()), 0.)
            with torch.no_grad():
                module.S_lora[1].B_weight.normal_(std=.01)
                module.P_lora[1].B_weight.normal_(std=.01)
            module.protection_progress = 1.
            expected = module._contrib_from_units(x, 1)
            original = module.qkv.weight.detach().clone()
            module.after_task(1)
            torch.testing.assert_close(torch.nn.functional.linear(x, module.qkv.weight - original),
                                       expected, rtol=1e-4, atol=1e-7)
            merged = module.qkv.weight.detach().clone()
            module.after_task(1)
            torch.testing.assert_close(module.qkv.weight, merged, rtol=0, atol=0)

    def test_counterfactual_position_diagnostics_keep_actual_rule(self):
        for name in ('ALL', 'UP', 'E1'):
            module = self.module(name)
            module.prototype_position_masks = torch.stack([torch.roll(module.general_mask.bool(), i, 0)
                                                          for i in range(5)])
            raw = torch.arange(1, module.qkv.weight.numel() + 1,
                               dtype=torch.float32).reshape_as(module.qkv.weight)
            values = []
            for position_name in ('wpre', 'prototype_high', 'prototype_shuffled', 'permuted'):
                module._prototype_audit_position = position_name
                values.append(module._safe_delta(raw, False).reshape(3, -1).norm(dim=1))
            norms = torch.stack(values)
            torch.testing.assert_close(norms.max(0).values, norms.min(0).values, rtol=1e-5, atol=1e-5)
            module._prototype_audit_position = None

    def test_incomplete_or_confounded_results_do_not_make_pairs(self):
        row = dict(valid_performance=True, **{key: 1. for key in analyzer.METRICS})
        reference = dict(row, mode='REF_seed1993')
        candidate = dict(row, mode='ALL_seed1993')
        self.assertEqual(len(analyzer.compare_complete([reference, candidate], [])), 1)
        self.assertEqual(analyzer.compare_complete([candidate], []), [])
        self.assertEqual(analyzer.compare_complete([reference, candidate], [{'field': 'source_sha256'}]), [])

    def test_summary_accepts_only_alpha_rule_changes(self):
        for unexpected in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / 'manifest.json').write_text(json.dumps(dict(machine='3090',
                    sweep_spec='scripts/sweeps/imgr10_protection_search_3090.json')))
                records = [dict(mode=name + '_seed1993', status='completed') for name in ('REF', 'E1')]
                (directory / 'queue.json').write_text(json.dumps(records))
                for record in records:
                    run = directory / record['mode']
                    run.mkdir()
                    (run / 'training.log').write_text('\n'.join('LoRA learning rates: task={}, epoch={}'.format(t, e)
                        for t in range(10) for e in range(1, 21)))
                def read_run(_, record):
                    variant = record['mode'].split('_seed')[0]
                    config = dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()), **self.settings(variant))
                    if unexpected and variant == 'E1':
                        config['slora_gamma'] = .4
                    snapshot = dict(machine='3090', phase='formal', mode=record['mode'],
                        effective_config=config, code_revision='same', source_sha256={}, software={}, hardware={})
                    return dict(record, valid_performance=True, Task0=97.1,
                                **{key: 1. for key in analyzer.METRICS}), snapshot, {key: [] for key in ('tasks', 'epochs', 'costs', 'storage')}
                with patch.object(position, 'read_run', side_effect=read_run), patch.object(position, 'draw'):
                    position.summarize_saved(directory)
                issues = json.loads((directory / 'matching_issues.json').read_text())
                self.assertEqual(bool(issues), unexpected)


if __name__ == '__main__':
    unittest.main()
