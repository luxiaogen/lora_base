"""Two-center CA: population moments, training-only stats, and real CA execution."""
import ast
import copy
import logging
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def method(name, source=None):
    tree = ast.parse(source or Path('methods/dlora.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(torch=torch, np=np, logging=logging, optim=torch.optim, F=F,
              MultivariateNormal=torch.distributions.MultivariateNormal,
              DataLoader=lambda dataset, **kwargs: dataset)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<actual-ca>', 'exec'), ns)
    return ns[name]


def stats_learner(enabled):
    pools = {i: np.array([[-3., i], [-1., i], [1., i], [3., i]], dtype=np.float32)
             for i in range(4)}
    calls = []

    def dataset(classes, **kwargs):
        calls.append((list(classes), kwargs))
        return None, None, int(classes[0])

    obj = SimpleNamespace(args={'ca': True, 'ca_two_centers': enabled},
                          _cur_task=0, _known_classes=0, _total_classes=2, feature_dim=2,
                          _extract_vectors=lambda loader: (pools[loader], None))
    return obj, SimpleNamespace(get_dataset=dataset), calls


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.classifier_pool = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(2)])
        self.batches = []

    def forward(self, inputs, fc_only=False):
        assert fc_only
        self.batches.append(inputs.detach().clone())
        return torch.cat([F.linear(F.normalize(inputs, dim=1), F.normalize(h.weight, dim=1))
                          for h in self.classifier_pool], dim=1)


class TwoCenterTests(unittest.TestCase):
    def test_statistics_preserve_overall_moments_and_only_extract_new_classes(self):
        obj, data, calls = stats_learner(True)
        compute = method('_compute_class_mean')
        rng = torch.get_rng_state().clone()
        compute(obj, data)
        # Baseline variance is 20/3; the mixture residual variance must be 8/3.
        self.assertAlmostEqual(obj._class_covs[0, 0, 0].item(), 8 / 3 + .001, places=5)
        offsets = obj._ca_mixture_offsets.clone()
        probs = obj._ca_mixture_probs.clone()
        covs = obj._class_covs.clone()
        torch.testing.assert_close((offsets * probs[..., None]).sum(1), torch.zeros(2, 2))
        for c in range(2):
            total = covs[c] + (offsets[c].T * probs[c]) @ offsets[c]
            torch.testing.assert_close(total, torch.diag(torch.tensor([20 / 3 + .001, .001])))
        obj._cur_task, obj._known_classes, obj._total_classes = 1, 2, 4
        compute(obj, data)
        self.assertEqual([c[0] for c in calls], [[0], [1], [2], [3]])
        self.assertTrue(all(c[1] == dict(source='train', mode='test', ret_data=True) for c in calls))
        self.assertTrue(torch.equal(offsets, obj._ca_mixture_offsets[:2]))
        self.assertTrue(torch.equal(probs, obj._ca_mixture_probs[:2]))
        self.assertTrue(torch.equal(covs, obj._class_covs[:2]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(obj._ca_new_features, {})

    def test_fitting_unequal_clusters_and_degenerate_class(self):
        from utils.ca_two_centers import fit_two_centers
        x = torch.tensor([[0., 0.], [0., 0.], [6., 0.]])
        rng = torch.get_rng_state().clone()
        offsets, probs, cov = fit_two_centers(x)
        torch.testing.assert_close(probs.sort().values, torch.tensor([1 / 3, 2 / 3]))
        torch.testing.assert_close(probs @ offsets, torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(cov + (offsets.T * probs) @ offsets,
                                   torch.diag(torch.tensor([12.001, .001])))
        for x in (torch.ones(4, 3), torch.ones(1, 3)):
            offsets, probs, cov = fit_two_centers(x)
            self.assertTrue(torch.equal(offsets, torch.zeros(2, 3)))
            self.assertEqual(probs.sum().item(), 1.)
            torch.testing.assert_close(cov, torch.eye(3) * .001)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_full_correlated_covariance_is_preserved(self):
        from utils.ca_two_centers import fit_two_centers
        x = torch.tensor([[1., 2., -1.], [2., 3., 0.], [0., 4., 1.],
                          [7., -2., 3.], [9., -3., 2.], [8., -1., 4.], [10., -2., 5.]])
        original = x.clone()
        a, p, shared = fit_two_centers(x)
        expected = torch.cov(x.double().T) + torch.eye(3) * .001
        torch.testing.assert_close((shared + (a.T * p) @ a).double(), expected,
                                   rtol=1e-6, atol=1e-6)
        self.assertTrue(bool((torch.linalg.eigvalsh(shared) > 0).all()))
        self.assertTrue(torch.equal(x, original))
        b, q, second = fit_two_centers(x)
        self.assertTrue(torch.equal(a, b) and torch.equal(p, q) and torch.equal(shared, second))

    def test_sampler_preserves_decayed_mean_and_covariance_in_expectation(self):
        from utils.ca_two_centers import fit_two_centers, add_center_offsets
        x = torch.tensor([[0., 0.], [0., 0.], [6., 0.]])
        a, p, shared = fit_two_centers(x)
        torch.manual_seed(19)
        # Old class: only the overall mean is shrunk. Component offsets are not.
        base = torch.distributions.MultivariateNormal(torch.tensor([1.8, 0.]), shared)
        samples = base.sample((100000,))
        rng = torch.get_rng_state().clone()
        samples = add_center_offsets(samples, a, p, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(samples.mean(0), torch.tensor([1.8, 0.]), rtol=0, atol=.025)
        torch.testing.assert_close(torch.cov(samples.T), torch.diag(torch.tensor([12.001, .001])),
                                   rtol=.02, atol=.001)

    def test_ca_changes_both_old_and_new_samples_not_rng_or_backbone(self):
        ca = method('_stage2_compact_classifier')
        results = []
        for enabled in (False, True):
            torch.manual_seed(1993)
            obj, data, _ = stats_learner(enabled)
            obj._total_classes = 4
            method('_compute_class_mean')(obj, data)
            obj._network = TinyNet()
            obj._device, obj._cur_task, obj._known_classes = torch.device('cpu'), 1, 2
            obj.task_sizes, obj.logit_norm = [2, 2], .1
            obj.args.update(ca_epochs=2, ca_lrate=.01, scale=20)
            before = copy.deepcopy(obj._network.state_dict())
            means, covs = obj._class_means.clone(), obj._class_covs.clone()
            ca(obj, 2)
            batches = torch.cat(obj._network.batches)
            self.assertEqual(batches.shape, (2 * 4 * 256, 2))
            for key in ('backbone.weight', 'backbone.bias'):
                self.assertTrue(torch.equal(before[key], obj._network.state_dict()[key]))
            self.assertIsNone(obj._network.backbone.weight.grad)
            self.assertFalse(torch.equal(before['classifier_pool.0.weight'],
                                         obj._network.classifier_pool[0].weight))
            self.assertTrue(torch.equal(means, obj._class_means))
            self.assertTrue(torch.equal(covs, obj._class_covs))
            # y separates the four fixture classes even after the old-class mean decay.
            counts = torch.bincount(batches[:, 1].round().long(), minlength=4)
            self.assertEqual(counts.tolist(), [512, 512, 512, 512])
            results.append((batches, torch.get_rng_state()))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        for c in range(4):
            rows = results[0][0][:, 1].round() == c
            self.assertFalse(torch.equal(results[0][0][rows], results[1][0][rows]))

    def test_off_matches_prechange_statistics_ca_weights_and_rng(self):
        source = subprocess.check_output(['git', 'show', '1bceb48:methods/dlora.py'], text=True)
        runs = []
        for text in (source, None):
            torch.manual_seed(32)
            obj, data, _ = stats_learner(False)
            obj._total_classes = 4
            method('_compute_class_mean', text)(obj, data)
            obj._network = TinyNet()
            obj._device, obj._cur_task, obj._known_classes = torch.device('cpu'), 1, 2
            obj.task_sizes, obj.logit_norm = [2, 2], .1
            obj.args.update(ca_epochs=1, ca_lrate=.01, scale=20)
            method('_stage2_compact_classifier', text)(obj, 2)
            runs.append((obj, torch.get_rng_state()))
        self.assertTrue(torch.equal(runs[0][1], runs[1][1]))
        for name, value in runs[0][0]._network.state_dict().items():
            self.assertTrue(torch.equal(value, runs[1][0]._network.state_dict()[name]), name)
        self.assertTrue(torch.equal(runs[0][0]._class_covs, runs[1][0]._class_covs))
        self.assertFalse(hasattr(runs[1][0], '_ca_mixture_offsets'))


if __name__ == '__main__':
    unittest.main()
