import ast
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from utils.old_competition import old_competition_loss


class OldReferenceTests(unittest.TestCase):
    def test_same_forward_but_only_positive_side_gradient(self):
        features = torch.tensor([[1., .3]], requires_grad=True)
        old = torch.nn.Parameter(torch.tensor([[1., 0.]]))
        new = torch.nn.Parameter(torch.tensor([[0., 1.], [-1., 0.]]))
        logits = F.linear(F.normalize(features, dim=1), F.normalize(new, dim=1))
        targets = torch.tensor([0])
        legacy, active = old_competition_loss(features, logits, targets, old, 20.)
        detached, active_detached = old_competition_loss(
            features, logits, targets, old, 20., detach_old=True)
        torch.testing.assert_close(legacy, detached, rtol=0, atol=0)
        torch.testing.assert_close(active, active_detached, rtol=0, atol=0)
        actual = torch.autograd.grad(detached, (features, new, old), retain_graph=True, allow_unused=True)
        expected = torch.autograd.grad(-20 * logits[0, 0], (features, new), retain_graph=True)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        self.assertIsNone(actual[2])
        self.assertGreater(actual[0].norm().item(), 0)
        legacy_grad = torch.autograd.grad(legacy, features)[0]
        self.assertFalse(torch.allclose(legacy_grad, actual[0]))

    def test_inactive_hinge_has_zero_gradient(self):
        features = torch.tensor([[0., 1.]], requires_grad=True)
        logits = torch.tensor([[.8, .2]], requires_grad=True)
        loss, active = old_competition_loss(features, logits, torch.tensor([0]),
                                           torch.tensor([[1., 0.]]), 20., detach_old=True)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(active.item(), 0)
        self.assertIsNone(features.grad)
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_training_hook_passes_flag_and_skips_task0(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_old_competition_term')
        ns = {'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<hook>', 'exec'), ns)
        learner = SimpleNamespace(_cur_task=1, args={'old_competition_weight': .1,
                                  'old_competition_detach_old': True, 'scale': 20},
                                  _network=SimpleNamespace(classifier_pool=[torch.nn.Linear(2, 1, bias=False)]))
        with torch.no_grad():
            learner._network.classifier_pool[0].weight.copy_(torch.tensor([[1., 0.]]))
        features = torch.tensor([[1., .3]], requires_grad=True)
        logits = torch.tensor([[.2, .1]], requires_grad=True)
        loss, _ = ns['_old_competition_term'](learner, {'features': features, 'logits': logits}, torch.tensor([0]))
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertAlmostEqual(logits.grad[0, 0].item(), -2.)
        for task, weight in ((0, .1), (1, 0)):
            learner._cur_task = task
            learner.args['old_competition_weight'] = weight
            self.assertEqual(ns['_old_competition_term'](learner, {}, None), (None, {}))

    def test_specs_match_previous_machine_recipe(self):
        for gpu in ('3090', '5090'):
            old = json.loads(Path(f'scripts/sweeps/imgr10_old_competition_{gpu}.json').read_text())
            spec = json.loads(Path(f'scripts/sweeps/imgr10_old_reference_{gpu}.json').read_text())
            expected = dict(old['common_overrides'])
            expected['wandb_group'] = f'imgr10_old_reference_{gpu}'
            self.assertEqual(spec['common_overrides'], expected)
            self.assertEqual(spec['seeds'], [1993])
            self.assertEqual([v['overrides'] for v in spec['variants']],
                             [{'old_competition_weight': .1, 'old_competition_detach_old': True}])
            script = f'scripts/9_26_imgr10_old_reference_{gpu}.sh'
            subprocess.run(['bash', '-n', script], check=True)
            commands = subprocess.check_output(['bash', script, '--dry-run'], text=True).split('    python main.py')[1:]
            self.assertEqual(len(commands), 1)
            tokens = shlex.split(commands[0].replace('\\\n', ' '))
            settings = dict(tokens[i+1].split('=', 1) for i, token in enumerate(tokens) if token == '--set')
            self.assertEqual(settings['old_competition_detach_old'], 'true')
            self.assertEqual(settings['old_competition_weight'], '0.1')
            self.assertEqual(settings['max_tasks'], '10')
            self.assertNotIn('data_path', settings)
            for key, value in {**spec['common_overrides'], **spec['variants'][0]['overrides']}.items():
                encoded = value if isinstance(value, str) else json.dumps(value, separators=(',', ':'))
                self.assertEqual(settings[key], encoded, key)


if __name__ == '__main__':
    unittest.main()
