import ast
import copy
import json
import logging
from pathlib import Path
import random
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import torch

from test.test_global_conflict_budget import GlobalBudgetSelectionTests


ROOT = Path(__file__).resolve().parents[1]


def learner_class():
    tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Learner')
    names = {'_extra_training_loss', '_log_mask_reg_gradients', '_backward_and_step'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    source = ast.Module(body=[ast.ClassDef(name='TestLearner', bases=[], keywords=[],
                                         body=methods, decorator_list=[])], type_ignores=[])
    namespace = {'torch': torch, 'logging': logging}
    exec(compile(ast.fix_missing_locations(source), '<learner>', 'exec'), namespace)
    return namespace['TestLearner']


class MaskRegTests(unittest.TestCase):
    def make(self, weight, task=1):
        module = GlobalBudgetSelectionTests._make_attention()
        module.before_task(task)
        module.cur_task = task
        module.dual_mask_task0_gate_mode = 'unmasked'
        with torch.no_grad():
            module.S_lora[task].B.weight.fill_(.1)
            if task:
                module.P_lora[task].B.weight.fill_(.2)
        learner = learner_class()()
        learner._cur_task = task
        learner.args = {'dual_mask_reg_weight': weight, 'use_slora': True, 'use_plora': True,
                        'dual_mask_anchor_reg_enabled': False}
        learner._iter_lora_modules = lambda: iter([module])
        return learner, module

    def test_diagnostic_preserves_gradients_weights_momentum_and_rng(self):
        for weight in (.01, 0.):
            learner, module = self.make(weight)
            state = copy.deepcopy(module.state_dict())
            results = []
            for diagnostic in (False, True):
                module.load_state_dict(state)
                params = [module.S_lora[1].B.weight, module.P_lora[1].B.weight]
                opt = torch.optim.SGD(params, lr=.02, momentum=.9)
                learner._sample_mask_reg_grad = diagnostic
                for step in range(3):
                    task_loss = sum((p - .5).square().sum() for p in params)
                    extra_loss = learner._extra_training_loss()
                    if diagnostic:
                        before_grads = [None if p.grad is None else p.grad.clone() for p in params]
                        rng, py_rng = torch.get_rng_state().clone(), random.getstate()
                        with self.assertLogs(level='INFO') as captured:
                            learner._log_mask_reg_gradients(task_loss, 0, 0)
                        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                        self.assertEqual(py_rng, random.getstate())
                        for p, before in zip(params, before_grads):
                            self.assertTrue(p.grad is None if before is None else torch.equal(p.grad, before))
                        records = [ast.literal_eval(line.split('MaskRegGrad ')[1]) for line in captured.output]
                        self.assertEqual([r['branch'] for r in records], ['S', 'P'])
                        if weight == 0:
                            self.assertTrue(all(r['weighted_reg_grad_norm'] == 0 and r['cosine'] is None for r in records))
                        else:
                            self.assertTrue(all(r['weighted_reg_grad_norm'] > 0 for r in records))
                    learner._backward_and_step(task_loss, extra_loss, opt, None, None)
                results.append(([p.detach().clone() for p in params],
                                [p.grad.clone() for p in params],
                                [opt.state[p]['momentum_buffer'].clone() for p in params]))
            for left, right in zip(*results):
                for a, b in zip(left, right):
                    self.assertTrue(torch.equal(a, b))

    def test_zero_removes_both_penalties_not_gates(self):
        learner, module = self.make(.01)
        learner._sample_mask_reg_grad = True
        delta = module.P_lora[1].B_weight @ module.P_lora[1].A_weight
        before = module._safe_delta(delta, isolated=True).detach().clone()
        self.assertGreater(float(learner._extra_training_loss().detach()), 0.)
        learner.args['dual_mask_reg_weight'] = 0.
        self.assertIsNone(learner._extra_training_loss())
        self.assertIsNone(learner._sampled_weighted_mask_reg)
        self.assertTrue(torch.equal(before, module._safe_delta(delta, isolated=True)))

    def test_task0_anchor_is_retained_and_unchanged(self):
        learner, module = self.make(.01, task=0)
        learner.args.update(dual_mask_anchor_reg_enabled=True, dual_mask_anchor_reg_weight=10.,
                            dual_mask_anchor_reg_task0_only=True)
        module.anchor_regularization = lambda: module.S_lora[0].B_weight.square().mean()
        on = learner._extra_training_loss()
        learner.args['dual_mask_reg_weight'] = 0.
        off = learner._extra_training_loss()
        self.assertGreater(float(off.detach()), 0.)
        self.assertTrue(torch.equal(on, off))

    def test_cosine_uses_weighted_regularizer_not_anchor(self):
        learner, module = self.make(.01)
        s, p = module.S_lora[1].B_weight, module.P_lora[1].B_weight
        task_loss = s.square().sum() + p.square().sum()
        learner._sampled_weighted_mask_reg = -.01 * task_loss
        with self.assertLogs(level='INFO') as captured:
            learner._log_mask_reg_gradients(task_loss, 9, 0)
        for line in captured.output:
            record = ast.literal_eval(line.split('MaskRegGrad ')[1])
            self.assertAlmostEqual(record['cosine'], -1., places=5)
            self.assertAlmostEqual(record['reg_to_task_ratio'], .01, places=6)

    def test_sampling_only_incremental_first_batch_epochs_1_10_20(self):
        tree = ast.parse((ROOT / 'methods/dlora.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and t.attr == '_sample_mask_reg_grad' for t in n.targets))
        expr = compile(ast.Expression(node.value), '<sampling>', 'eval')
        for enabled in (False, True):
            for task in (0, 1, 9):
                learner = SimpleNamespace(args={'dual_mask_reg_grad_diagnostic': enabled}, _cur_task=task)
                selected = [(e, b) for e in range(20) for b in range(2)
                            if eval(expr, {'self': learner, 'epoch': e, 'i': b})]
                self.assertEqual(selected, [(0, 0), (9, 0), (19, 0)] if enabled and task > 0 else [])


class ScriptTests(unittest.TestCase):
    def test_pair_only_changes_reg_weight(self):
        old = json.loads((ROOT / 'scripts/sweeps/imgr10_plora_lr_3090.json').read_text())
        spec = json.loads((ROOT / 'scripts/sweeps/imgr10_mask_reg_3090.json').read_text())
        self.assertEqual(spec['seeds'], [1993])
        for key, value in old['common_overrides'].items():
            if key != 'wandb_group':
                self.assertEqual(spec['common_overrides'][key], value, key)
        self.assertEqual(spec['common_overrides']['plora_lr_multiplier'], 1.)
        self.assertTrue(spec['common_overrides']['dual_mask_reg_grad_diagnostic'])
        self.assertEqual([v['overrides'] for v in spec['variants']],
                         [{'dual_mask_reg_weight': .01}, {'dual_mask_reg_weight': 0.}])
        script = ROOT / 'scripts/9_25_imgr10_mask_reg_3090.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        output = subprocess.check_output(['bash', str(script), '--dry-run'], cwd='/tmp', text=True)
        commands = output.replace('\\\n', '').splitlines()
        self.assertEqual(len(commands), 2)
        for command, variant in zip(commands, spec['variants']):
            tokens = shlex.split(command)
            settings = dict(tokens[i + 1].split('=', 1) for i, t in enumerate(tokens) if t == '--set')
            self.assertNotIn('data_path', settings)
            self.assertIn('${TIMESTAMP}', settings['prefix'])
            self.assertIn('$@', tokens)
            for key, expected in {**spec['common_overrides'], **variant['overrides']}.items():
                try:
                    actual = json.loads(settings[key])
                except json.JSONDecodeError:
                    actual = settings[key]
                self.assertEqual(actual, expected, key)


if __name__ == '__main__':
    unittest.main()
