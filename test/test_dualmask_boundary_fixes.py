"""边界修复及合法基线路径的确定性回归。"""
import argparse
import ast
import copy
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import torch
from torch.nn import functional as F

from test import test_protect_position
from scripts.dualmask_config import normalize_dualmask_config


ROOT = Path(__file__).resolve().parents[1]
BASE = '8c92912d6a6e44fcb45e7795d72387713ede835e'


def function_from_source(source, name):
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(torch=torch, logging=logging, argparse=argparse, json=json,
                     math=math, warnings=warnings)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<real-function>', 'exec'), namespace)
    return namespace[name]


def run_entry(config, overrides=()):
    """只替换昂贵的训练边界，执行真实JSON加载、覆盖和启动逻辑。"""
    namespace = dict(argparse=argparse, json=json, math=math, warnings=warnings,
                     normalize_dualmask_config=normalize_dualmask_config)
    nodes = [n for n in ast.parse((ROOT / 'main.py').read_text()).body
             if isinstance(n, ast.FunctionDef)]
    results = []
    namespace['train'] = lambda args: results.append(copy.deepcopy(args))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<entrypoint>', 'exec'), namespace)
    with tempfile.TemporaryDirectory() as temp:
        config_path = Path(temp) / 'config.json'
        config_path.write_text(json.dumps(config))
        argv = ['main.py', '--config', str(config_path)]
        for override in overrides:
            argv += ['--set', override]
        with patch.object(sys, 'argv', argv):
            namespace['main']()
    return results[0]


def average_parameters(learner):
    """执行真实训练函数的平均参数收集块，省略模型训练与数据加载。"""
    tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
    train = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'train_function')
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i,n in enumerate(train.body) if assigns(n, 'average_epochs'))
    end = next(i for i,n in enumerate(train.body) if assigns(n, 'repro'))
    wrapper = ast.parse('def collect(self):\n    pass\n').body[0]
    wrapper.body = train.body[start:end] + [ast.Return(value=ast.Name(id='averaged_params', ctx=ast.Load()))]
    ast.fix_missing_locations(wrapper)
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<average-collection>', 'exec'), namespace)
    return namespace['collect'](learner)


class BoundaryFixTests(unittest.TestCase):
    def test_entry_clips_only_out_of_range_after_cli_overrides(self):
        for raw, want in ((-.2, 0.), (1.2, 1.), (0., 0.), (.5, .5), (1., 1.)):
            with self.subTest(raw=raw), warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter('always')
                result = run_entry(dict(dual_mask_fixed_protect_strength=.5),
                                   ['dual_mask_fixed_protect_strength=' + str(raw)])
                self.assertEqual(result['dual_mask_fixed_protect_strength'], want)
                self.assertEqual(len(emitted), int(raw != want))
        with warnings.catch_warnings(record=True) as emitted:
            result = run_entry(dict(dual_mask_fixed_protect_strength=None))
        self.assertIsNone(result['dual_mask_fixed_protect_strength'])
        self.assertFalse(emitted)

    def test_entry_rejects_nonfinite_before_training(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite|有限'):
                run_entry(dict(dual_mask_fixed_protect_strength=value))

    def test_entry_overlap_missing_false_and_explicit_true(self):
        self.assertFalse(run_entry({})['dual_mask_conflict_old_overlap_adaptive'])
        self.assertFalse(run_entry(dict(dual_mask_conflict_old_overlap_adaptive=False))[
            'dual_mask_conflict_old_overlap_adaptive'])
        self.assertTrue(run_entry(dict(dual_mask_conflict_old_overlap_adaptive=True))[
            'dual_mask_conflict_old_overlap_adaptive'])

    def test_entry_rejects_single_staged_only_when_incremental(self):
        config = dict(dual_mask_branch_layout='single', sp_staged_s_epochs=5)
        for limit in (None, 2, 10):
            case = dict(config)
            if limit is not None:
                case['max_tasks'] = limit
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'single|单分支'):
                run_entry(case)
        self.assertEqual(run_entry(dict(config, max_tasks=1))['max_tasks'], 1)
        self.assertEqual(run_entry(dict(config, dual_mask_branch_layout='dual'))['sp_staged_s_epochs'], 5)
        self.assertEqual(run_entry(dict(config, sp_staged_s_epochs=0))['dual_mask_branch_layout'], 'single')

    def test_permutation_four_combinations_use_one_reference(self):
        template = test_protect_position.ProtectPositionTests().make()
        template.rebuild_dual_masks()
        original = template.general_mask.clone()
        once = copy.deepcopy(template)
        once.args['dual_mask_protect_position'] = 'permuted'
        once.rebuild_dual_masks()
        permuted = once.general_mask.clone()
        self.assertFalse(torch.equal(original, permuted))
        rng = torch.get_rng_state().clone()
        for common, private in (('wpre','wpre'), ('wpre','permuted'),
                                ('permuted','wpre'), ('permuted','permuted')):
            module = copy.deepcopy(template)
            module.args.update(dual_mask_protect_position=common, p_permission_position=private)
            module.rebuild_dual_masks()
            expected_s = original if common == 'wpre' else permuted
            expected_p = permuted if private == 'permuted' else expected_s
            torch.testing.assert_close(module.general_mask, expected_s, atol=0, rtol=0)
            torch.testing.assert_close(module._p_protect_mask(), expected_p, atol=0, rtol=0)
            torch.testing.assert_close(module.isolated_mask, 1 - expected_s, atol=0, rtol=0)
            for raw, actual in zip(original.chunk(3), module._p_protect_mask().chunk(3)):
                self.assertEqual(int(raw.sum()), int(actual.sum()))
            delta = torch.ones_like(original)
            safe = module._safe_delta(delta, True)
            self.assertEqual(int(safe[expected_p.bool()].count_nonzero()), 0)
            module.cur_task = 2
            module.rebuild_dual_masks()
            torch.testing.assert_close(module._p_protect_mask(), expected_p, atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_model_assignment_bounds_tail_and_release_paths(self):
        for raw, want in ((-.2, 0.), (1.2, 1.), (.5, .5)):
            for mode in ('soft_tail', 'release'):
                module = test_protect_position.ProtectPositionTests().make()
                module.args.update(dual_mask_fixed_protect_strength=raw)
                if mode == 'soft_tail':
                    module.args['dual_mask_update_rule'] = mode
                else:
                    module.args.update(p_permission_release='benefit', p_permission_norm_match=False)
                module.before_task(1)
                self.assertEqual(module.effective_protect_strength, want)
                module.p_permission_release_mask = module.general_mask.bool().clone()
                delta = torch.ones_like(module.qkv.weight, requires_grad=True)
                safe, gate, _ = module._safe_delta(delta, mode == 'release', return_details=True)
                self.assertTrue(torch.all((gate >= 0) & (gate <= 1)))
                self.assertTrue(torch.all(safe.abs() <= delta.abs()))
                empty = torch.zeros_like(delta, requires_grad=True)
                module._safe_delta(empty, mode == 'release').sum().backward()
                self.assertGreater(float(empty.grad.norm()), 0.)

    def prototype_fixture(self, overlap=None):
        args = dict(dual_mask_plasticity_adaptive=False, dual_mask_competence_holdout_mod=2)
        if overlap is not None:
            args['dual_mask_conflict_old_overlap_adaptive'] = overlap
        module = test_protect_position.ProtectPositionTests().make()
        args = dict(module.args, **args)
        module._init_params(args)
        module.cur_task = 1
        features = torch.tensor([[1., 0.]] * 4)
        learner = SimpleNamespace(args=args, _cur_task=1, _known_classes=1, _total_classes=2,
            scale=20., _w0_class_means={0:torch.tensor([1.,0.])},
            _collect_anchor_features=lambda loader:(torch.arange(4), features, torch.ones(4, dtype=torch.long)),
            _iter_lora_modules=lambda:iter([module]))
        return learner, module

    def test_overlap_preparation_defaults_off_and_true_keeps_adaptation(self):
        prepare = function_from_source((ROOT / 'methods/dlora.py').read_text(), '_prepare_w0_prototypes')
        for flag in (None, False, True):
            learner, module = self.prototype_fixture(flag)
            prepare(learner, None)
            self.assertEqual(learner._w0_competence_new, 1.)
            if flag is True:
                self.assertEqual(learner._w0_competence_all_seen, 0.)
                self.assertEqual(learner._w0_old_overlap_risk, 1.)
                self.assertEqual(module._conflict_parameters()[1], 1.)
            else:
                self.assertIsNone(learner._w0_competence_all_seen)
                self.assertIsNone(learner._w0_old_overlap_risk)
                self.assertEqual(module._conflict_parameters()[1], .5)

    def test_single_late_average_collects_joint_b_once_and_merge_once(self):
        apply_average = function_from_source((ROOT / 'methods/dlora.py').read_text(), '_apply_epoch_average')
        for layout in ('dual', 'single'):
            module = test_protect_position.ProtectPositionTests().make()
            module.args['dual_mask_branch_layout'] = layout
            module.before_task(1)
            module.set_task_and_stage(1, 2)
            network = torch.nn.Module()
            network.classifier_pool = torch.nn.ModuleList([torch.nn.Linear(12,2), torch.nn.Linear(12,2)])
            learner = SimpleNamespace(args={'late_weight_average_epochs':5}, _cur_task=1, _network=network,
                _iter_lora_modules=lambda:iter([module]))
            params = average_parameters(learner)
            expected = [module.S_lora[1].B_weight]
            if layout == 'dual':
                expected.append(module.P_lora[1].B_weight)
            expected += list(network.classifier_pool[1].parameters())
            self.assertEqual([id(p) for p in params], [id(p) for p in expected])
            self.assertEqual(len(params), len(set(id(p) for p in params)))
            old_head = copy.deepcopy(network.classifier_pool[0].state_dict())
            original_a = module.S_lora[1].A_weight.detach().clone()
            sums = [torch.zeros_like(p) for p in params]
            with torch.no_grad():
                for epoch in range(16,21):
                    for p, total in zip(params, sums):
                        p.fill_(epoch / 100.)
                        total.add_(p)
            apply_average(params, sums, 5)
            for p in params:
                torch.testing.assert_close(p, torch.full_like(p,.18))
            torch.testing.assert_close(module.S_lora[1].A_weight, original_a, atol=0, rtol=0)
            for key,value in old_head.items():
                torch.testing.assert_close(network.classifier_pool[0].state_dict()[key], value, atol=0, rtol=0)
            x = torch.ones(3,4)
            before = F.linear(x,module.qkv.weight,module.qkv.bias) + module._contrib_from_units(x,1)
            module.after_task(1)
            torch.testing.assert_close(before, F.linear(x,module.qkv.weight,module.qkv.bias), atol=2e-6, rtol=2e-5)
            self.assertEqual(int(module._contrib_from_units(x,1).count_nonzero()),0)
            merged = module.qkv.weight.detach().clone()
            module.after_task(1)
            torch.testing.assert_close(module.qkv.weight, merged, atol=0, rtol=0)

    def test_legal_baseline_matches_original_rng_forward_loss_grad_step_merge(self):
        from models.attention import Attention_LoRA
        source = subprocess.check_output(['git','show',BASE + ':models/attention.py'], text=True)
        namespace = {'__name__':'boundary_reference'}
        exec(compile(source,'boundary_reference.py','exec'),namespace)
        config = json.loads((ROOT / 'docs/experiments/baselines/imgr10_B1_3090.json').read_text())
        config.update(seed=1993, dual_mask_svd_rank=2)
        initial_rng = torch.get_rng_state().clone()
        reference = namespace['Attention_LoRA'](dim=4,num_heads=1,r=2,n_tasks=2)
        torch.set_rng_state(initial_rng)
        current = Attention_LoRA(dim=4,num_heads=1,r=2,n_tasks=2)
        for model in (reference,current):
            model._init_params(copy.deepcopy(config))
        current.load_state_dict(reference.state_dict())
        for task in (0,1):
            for model in (reference,current):
                model.set_pretrained_competence(.8,.2)
            state = torch.get_rng_state().clone()
            reference.before_task(task)
            final_rng = torch.get_rng_state().clone()
            torch.set_rng_state(state)
            current.before_task(task)
            self.assertTrue(torch.equal(final_rng,torch.get_rng_state()))
            for model in (reference,current):
                model.set_task_and_stage(task,2)
            x = torch.randn(3,4)
            target = torch.randn(3,12)
            records = []
            for model in (reference,current):
                optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.02)
                output = F.linear(x,model.qkv.weight,model.qkv.bias) + model._contrib_from_units(x,task)
                loss = F.mse_loss(output,target)
                for unit,isolated in ((model.S_lora[task],False),(model.P_lora[task],True)):
                    loss = loss + .01 * model._joint_conflict_regularization(unit,isolated)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grads = {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
                optimizer.step()
                records.append((output.detach(),loss.detach(),grads,copy.deepcopy(model.state_dict())))
            for a,b in zip(records[0][:2],records[1][:2]):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            self.assertEqual(set(records[0][2]),set(records[1][2]))
            for i in (2,3):
                for name,value in records[0][i].items():
                    torch.testing.assert_close(value,records[1][i][name],atol=0,rtol=0)
            for model in (reference,current):
                model.after_task(task)
            torch.testing.assert_close(reference.qkv.weight,current.qkv.weight,atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
