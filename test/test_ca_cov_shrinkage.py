import ast
import copy
import json
import logging
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F


class CovarianceShrinkageTests(unittest.TestCase):
    def test_actual_ca_covariances_default_rng_and_head_only_update(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Learner')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_stage2_compact_classifier')
        captured = []

        def distribution(mean, covariance):
            captured.append((mean.clone(), covariance.clone()))
            return torch.distributions.MultivariateNormal(mean, covariance)

        namespace = dict(torch=torch, optim=torch.optim, F=F, logging=logging,
                         MultivariateNormal=distribution)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-ca>', 'exec'), namespace)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 3)
                self.classifier_pool = nn.ModuleList([nn.Linear(3, 2, bias=False) for _ in range(2)])

            def forward(self, x, fc_only=False):
                return torch.cat([F.linear(F.normalize(x, dim=1), F.normalize(h.weight, dim=1))
                                  for h in self.classifier_pool], dim=1)

        torch.manual_seed(7)
        net = Net()
        initial = copy.deepcopy(net.state_dict())
        matrix = torch.randn(4, 3, 3)
        covariances = matrix @ matrix.transpose(-1, -2) + .1 * torch.eye(3)
        means = torch.randn(4, 3)
        learner = SimpleNamespace(args=dict(ca_epochs=1, ca_lrate=.01, scale=20),
                                  _network=net, _cur_task=1, _total_classes=4, _known_classes=2,
                                  _device=torch.device('cpu'), logit_norm=.1, task_sizes=[2, 2],
                                  _class_means=means.clone(), _class_covs=covariances.clone())
        results = []
        for alpha in (None, 0., .5, 1.):
            net.load_state_dict(initial)
            if alpha is not None:
                learner.args['ca_cov_shrinkage'] = alpha
            captured.clear()
            torch.manual_seed(1993)
            namespace['_stage2_compact_classifier'](learner, 2)
            results.append((copy.deepcopy(net.state_dict()), torch.get_rng_state().clone()))
            for c, (mean, cov) in enumerate(captured):
                expected = covariances[c] if alpha is None else (
                    (1 - alpha) * covariances[c] + alpha * torch.diag(covariances[c].diagonal()))
                torch.testing.assert_close(cov, expected, rtol=0, atol=0)
                torch.testing.assert_close(cov.diagonal(), covariances[c].diagonal(), rtol=0, atol=0)
                torch.testing.assert_close(mean, means[c] * (.95 if c < 2 else 1.), rtol=0, atol=0)
                self.assertTrue(bool((torch.linalg.eigvalsh(cov) > 0).all()))
            self.assertEqual(len(captured), 4)
            self.assertTrue(torch.equal(learner._class_covs, covariances))
            self.assertTrue(torch.equal(learner._class_means, means))
            self.assertTrue(torch.equal(net.backbone.weight, initial['backbone.weight']))
            self.assertIsNone(net.backbone.weight.grad)
        for key in initial:
            self.assertTrue(torch.equal(results[0][0][key], results[1][0][key]))
        for result in results[1:]:
            self.assertTrue(torch.equal(results[0][1], result[1]))
        self.assertFalse(torch.equal(results[0][0]['classifier_pool.0.weight'],
                                     results[2][0]['classifier_pool.0.weight']))

    def test_recipe_is_single_candidate_against_saved_baseline(self):
        root = Path(__file__).resolve().parents[1]
        base = json.loads((root / 'scripts/sweeps/imgr10_anchor2p5_save_t10_3090.json').read_text())
        candidate = json.loads((root / 'scripts/sweeps/imgr10_ca_cov_shrinkage_3090.json').read_text())
        self.assertEqual(len(candidate['variants']), 1)
        old = {**base['common_overrides'], **base['variants'][0]['overrides']}
        new = {**candidate['common_overrides'], **candidate['variants'][0]['overrides']}
        old.pop('wandb_group'); new.pop('wandb_group')
        self.assertEqual(new.pop('ca_cov_shrinkage'), .5)
        self.assertFalse(new.pop('ca_stats_transport'))
        self.assertEqual(new.pop('ca_cross_task_margin_weight'), 0.)
        self.assertEqual(old, new)
        script = root / 'scripts/9_28_imgr10_ca_cov_shrinkage_3090.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        for mode, tasks in (([], '10'), (['--smoke'], '2')):
            output = subprocess.check_output(['bash', str(script), '--dry-run', *mode], cwd='/tmp', text=True)
            commands = [line for line in output.splitlines() if 'main.py --config' in line]
            self.assertEqual(len(commands), 1)
            tokens = shlex.split(commands[0])
            settings = dict(tokens[i + 1].split('=', 1) for i, token in enumerate(tokens) if token == '--set')
            self.assertEqual(settings['max_tasks'], tasks)
            self.assertEqual(settings['ca_cov_shrinkage'], '0.5')
            self.assertEqual(settings['seed'], '[1993]')
            self.assertEqual(settings['save_task_weights'], 'true')
            self.assertNotIn('data_path', settings)
            self.assertNotIn('device', settings)


if __name__ == '__main__':
    unittest.main()
