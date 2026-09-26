import ast
import copy
import logging
import json
import shlex
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import MultivariateNormal


def ca_method(source):
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_stage2_compact_classifier')
    ns = dict(torch=torch, optim=optim, F=F, logging=logging, MultivariateNormal=MultivariateNormal)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<ca>', 'exec'), ns)
    return ns[method.name]


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.classifier_pool = nn.ModuleList([nn.Linear(3, 2, bias=False) for _ in range(2)])
        self.batches = []

    def forward(self, inputs, fc_only=False):
        assert fc_only
        self.batches.append(inputs.detach().clone())
        return torch.cat([F.linear(F.normalize(inputs, dim=1), F.normalize(h.weight, dim=1))
                          for h in self.classifier_pool], dim=1)


def learner(enabled):
    return SimpleNamespace(
        args=dict(ca_epochs=2, ca_lrate=.01, scale=20, ca_real_new_features=enabled),
        _network=TinyNetwork(), _device=torch.device('cpu'), _cur_task=1,
        _known_classes=2, _total_classes=4, task_sizes=[2, 2], logit_norm=.1,
        _class_means=torch.tensor([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.], [5., 5., 5.]]),
        _class_covs=torch.eye(3).repeat(4, 1, 1)*.01,
        _ca_new_features={2: torch.tensor([[0., 0., 20.], [0., 0., 21.]]),
                          3: torch.tensor([[20., 20., 20.], [21., 21., 21.]])})


class CARealNewTests(unittest.TestCase):
    def test_cache_reuses_current_train_extraction_and_skips_task0(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_compute_class_mean')
        ns = dict(torch=torch, np=np, DataLoader=lambda dataset, **kwargs: dataset)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<stats>', 'exec'), ns)
        for task, enabled, ca in ((0, True, True), (1, False, True), (1, True, False), (1, True, True)):
            calls = []
            def dataset(classes, **kwargs):
                calls.append((list(classes), kwargs))
                return None, None, int(classes[0])
            pools = {i: np.array([[i, 1., 0.], [i, 2., 1.], [i, 3., 2.]], dtype=np.float32)
                     for i in range(4)}
            obj = SimpleNamespace(args={'ca_real_new_features': enabled, 'ca': ca},
                                  _cur_task=task, _known_classes=2 if task else 0,
                                  _total_classes=4 if task else 2, feature_dim=3,
                                  _ca_new_features={99: torch.ones(1)},
                                  _extract_vectors=lambda loader: (pools[loader], None))
            if task:
                obj._class_means = torch.ones(2, 3)
                obj._class_covs = torch.eye(3).repeat(2, 1, 1)
            ns[method.name](obj, SimpleNamespace(get_dataset=dataset))
            self.assertEqual(len(calls), 2)
            for _, options in calls:
                self.assertEqual(options, dict(source='train', mode='test', ret_data=True))
            expected = {2, 3} if task and enabled and ca else set()
            self.assertEqual(set(obj._ca_new_features), expected)
            for key in expected:
                torch.testing.assert_close(obj._ca_new_features[key], torch.tensor(pools[key]))
                self.assertFalse(obj._ca_new_features[key].requires_grad)
                self.assertEqual(obj._ca_new_features[key].device.type, 'cpu')
            if task:
                torch.testing.assert_close(obj._class_means[:2], torch.ones(2, 3))

    def test_default_exactly_matches_prior_ca(self):
        old = ca_method(subprocess.check_output(['git', 'show', 'e78251c:methods/dlora.py'], text=True))
        new = ca_method(Path('methods/dlora.py').read_text())
        outputs = []
        for method in (old, new):
            torch.manual_seed(31)
            obj = learner(False)
            method(obj, 2)
            outputs.append((obj._network.state_dict(), torch.get_rng_state()))
        self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))
        for key in outputs[0][0]:
            self.assertTrue(torch.equal(outputs[0][0][key], outputs[1][0][key]), key)

    def test_candidate_preserves_rng_old_samples_and_backbone(self):
        method = ca_method(Path('methods/dlora.py').read_text())
        runs = []
        for enabled in (False, True):
            torch.manual_seed(31)
            obj = learner(enabled)
            before = copy.deepcopy(obj._network.state_dict())
            method(obj, 2)
            for key in ('backbone.weight', 'backbone.bias'):
                self.assertTrue(torch.equal(before[key], obj._network.state_dict()[key]))
            self.assertIsNone(obj._network.backbone.weight.grad)
            if enabled:
                self.assertEqual(obj._ca_new_features, {})
            runs.append((obj, torch.get_rng_state()))
        self.assertTrue(torch.equal(runs[0][1], runs[1][1]))
        a, b = (torch.cat(obj._network.batches) for obj, _ in runs)
        self.assertEqual(a.shape, (2*4*256, 3))
        self.assertEqual(b.shape, a.shape)
        old = ((a[:, 0] > 9) | (a[:, 1] > 9))
        self.assertEqual(int(old.sum()), 2*2*256)
        self.assertTrue(torch.equal(a[old], b[old]))
        pool = torch.cat(list(learner(True)._ca_new_features.values()))
        self.assertTrue((b[~old, None, :] == pool[None, :, :]).all(-1).any(-1).all())
        self.assertEqual(int((b[~old, 0] == 0).sum()), 2*256)
        self.assertEqual(int((b[~old, 0] >= 20).sum()), 2*256)
        self.assertFalse(torch.equal(runs[0][0]._network.classifier_pool[1].weight,
                                     runs[1][0]._network.classifier_pool[1].weight))

    def test_machine_scripts_match_specs_and_only_source_changes(self):
        for gpu in ('3090', '5090'):
            spec = json.loads(Path(f'scripts/sweeps/imgr10_ca_real_new_{gpu}.json').read_text())
            self.assertEqual(spec['seeds'], [1993])
            self.assertEqual([v['overrides'] for v in spec['variants']],
                             [{'ca_real_new_features': False}, {'ca_real_new_features': True}])
            self.assertEqual(spec['common_overrides']['old_competition_weight'], 0)
            script = f'scripts/9_26_imgr10_ca_real_new_{gpu}.sh'
            subprocess.run(['bash', '-n', script], check=True)
            commands = subprocess.check_output(['bash', script, '--dry-run'], text=True).split('    python main.py')[1:]
            self.assertEqual(len(commands), 2)
            for command, variant in zip(commands, spec['variants']):
                tokens = shlex.split(command.replace('\\\n', ' '))
                settings = dict(tokens[i+1].split('=', 1) for i, token in enumerate(tokens) if token == '--set')
                for k, v in {**spec['common_overrides'], **variant['overrides']}.items():
                    self.assertEqual(settings[k], v if isinstance(v, str) else json.dumps(v, separators=(',', ':')))
                self.assertNotIn('data_path', settings)


if __name__ == '__main__':
    unittest.main()
