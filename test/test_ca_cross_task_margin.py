import ast
import copy
import logging
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from utils.ca_cross_task_margin import cross_task_margin


class CrossTaskMarginTests(unittest.TestCase):
    def test_3090_recipe_changes_only_ca_objective(self):
        root = Path(__file__).resolve().parents[1]
        base = json.loads((root / 'scripts/sweeps/imgr10_anchor2p5_save_t10_3090.json').read_text())
        new = json.loads((root / 'scripts/sweeps/imgr10_ca_cross_margin_3090.json').read_text())
        old_args = {**base['common_overrides'], **base['variants'][0]['overrides']}
        new_args = {**new['common_overrides'], **new['variants'][0]['overrides']}
        old_args.pop('wandb_group'); new_args.pop('wandb_group')
        self.assertFalse(new_args.pop('ca_stats_transport'))
        self.assertEqual(new_args.pop('ca_cross_task_margin_weight'), 1)
        self.assertEqual(new_args.pop('ca_cross_task_margin'), .05)
        self.assertEqual(old_args, new_args)
        script = root / 'scripts/9_28_imgr10_ca_cross_margin_3090.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        for mode, tasks in (([], '10'), (['--smoke'], '2')):
            output = subprocess.check_output(['bash', str(script), '--dry-run', *mode], cwd='/tmp', text=True)
            commands = [line for line in output.splitlines() if 'main.py --config' in line]
            self.assertEqual(len(commands), 1)
            tokens = shlex.split(commands[0])
            settings = dict(tokens[i + 1].split('=', 1) for i, token in enumerate(tokens) if token == '--set')
            self.assertEqual(settings['max_tasks'], tasks)
            self.assertEqual(settings['seed'], '[1993]')
            self.assertEqual(settings['save_task_weights'], 'true')
            self.assertNotIn('data_path', settings)

    def test_only_other_tasks_are_negative_candidates(self):
        logits = torch.tensor([[.5, .99, .52, .1, .3]], requires_grad=True)
        # Unequal task sizes: classes 0/1 vs 2/3/4.
        task_ids = torch.tensor([0, 0, 1, 1, 1])
        loss, active = cross_task_margin(logits, torch.tensor([0]), task_ids, .05)
        torch.testing.assert_close(loss, torch.tensor(.07))
        self.assertEqual(active.item(), 1.)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[-1., 0., 1., 0., 0.]]))

    def test_old_old_pairs_and_reverse_direction_both_receive_gradients(self):
        task_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        logits = torch.tensor([[.3, .1, .4, .1, -.2, -.3],
                               [.4, .1, .3, .1, -.2, -.3]], requires_grad=True)
        loss, _ = cross_task_margin(logits, torch.tensor([0, 2]), task_ids, .05)
        loss.backward()
        torch.testing.assert_close(logits.grad[0], torch.tensor([-.5, 0., .5, 0., 0., 0.]))
        torch.testing.assert_close(logits.grad[1], torch.tensor([.5, 0., -.5, 0., 0., 0.]))

    def test_separated_and_single_task_examples_have_zero_loss(self):
        for ids in ([0, 1], [0, 0]):
            logits = torch.tensor([[.8, .2]], requires_grad=True)
            loss, active = cross_task_margin(logits, torch.tensor([0]), torch.tensor(ids), .05)
            self.assertEqual(loss.item(), 0.)
            self.assertEqual(active.item(), 0.)
            loss.backward()
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_actual_ca_default_unchanged_and_candidate_updates_only_heads(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Learner')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_stage2_compact_classifier')
        namespace = dict(torch=torch, optim=torch.optim, F=F, logging=logging,
                         MultivariateNormal=torch.distributions.MultivariateNormal)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-ca>', 'exec'), namespace)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 3)
                self.classifier_pool = nn.ModuleList([nn.Linear(3, 2, bias=False) for _ in range(3)])

            def forward(self, x, fc_only=False):
                return torch.cat([F.linear(F.normalize(x, dim=1), F.normalize(h.weight, dim=1))
                                  for h in self.classifier_pool], dim=1)

        torch.manual_seed(7)
        net = Net()
        initial = copy.deepcopy(net.state_dict())
        learner = SimpleNamespace(args=dict(ca_epochs=1, ca_lrate=.01, scale=20),
                                  _network=net, _cur_task=2, _total_classes=6, _known_classes=4,
                                  _device=torch.device('cpu'), logit_norm=.1, task_sizes=[2, 2, 2],
                                  _class_means=torch.randn(6, 3), _class_covs=torch.eye(3).repeat(6, 1, 1))
        results = []
        for weight in (None, 0., 1.):
            net.load_state_dict(initial)
            if weight is not None:
                learner.args['ca_cross_task_margin_weight'] = weight
            learner.args['ca_cross_task_margin'] = .05
            torch.manual_seed(1993)
            with self.assertLogs(level='INFO') as captured:
                namespace['_stage2_compact_classifier'](learner, 2)
            results.append((copy.deepcopy(net.state_dict()), torch.get_rng_state().clone()))
            self.assertEqual(any('CACrossTaskMargin' in line for line in captured.output), weight == 1.)
            torch.testing.assert_close(net.backbone.weight, initial['backbone.weight'], rtol=0, atol=0)
            self.assertIsNone(net.backbone.weight.grad)
        for key in initial:
            self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertTrue(torch.equal(results[0][1], results[2][1]))
        self.assertFalse(torch.equal(results[0][0]['classifier_pool.0.weight'],
                                     results[2][0]['classifier_pool.0.weight']))


if __name__ == '__main__':
    unittest.main()
