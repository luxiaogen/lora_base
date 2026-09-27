import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.head_start import collect_features, prototype_init, fit_head


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.heads = nn.ModuleList([nn.Linear(3, 2, bias=False) for _ in range(2)])
        self.tasks = []

    def extract_vector(self, x, task_id=None):
        self.tasks.append(task_id)
        return self.encoder(x)

    @property
    def classifier_pool(self):
        return self.heads


class HeadStartTests(unittest.TestCase):
    def test_prototype_direction_and_original_norm(self):
        head = nn.Linear(3, 2, bias=False)
        norms = head.weight.detach().norm(dim=1)
        features = torch.tensor([[2., 0, 0], [0., 1, 0], [0., 0, 4]])
        prototype_init(head, features, torch.tensor([0, 0, 1]))
        expected = F.normalize(torch.tensor([[1., 1, 0], [0., 0, 1]]), dim=1)
        torch.testing.assert_close(F.normalize(head.weight, dim=1), expected)
        torch.testing.assert_close(head.weight.norm(dim=1), norms)

    def test_collection_read_only_and_task_explicit(self):
        net = Network().train()
        net.encoder.eval()
        loader = DataLoader(TensorDataset(torch.arange(6), torch.randn(6, 3),
                                         torch.tensor([2, 3, 2, 3, 2, 3])), batch_size=2,
                            generator=torch.Generator().manual_seed(42))
        weights = {k: v.clone() for k, v in net.state_dict().items()}
        rng = torch.get_rng_state().clone()
        loader_rng = loader.generator.get_state().clone()
        x, y = collect_features(net, loader, 'cpu', task=0, known_classes=2)
        self.assertEqual(net.tasks, [0, 0, 0])
        self.assertEqual(x.shape, (6, 3))
        self.assertEqual(y.tolist(), [0, 1, 0, 1, 0, 1])
        self.assertFalse(x.requires_grad)
        self.assertTrue(net.training)
        self.assertFalse(net.encoder.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(loader_rng, loader.generator.get_state()))
        for k, v in net.state_dict().items():
            torch.testing.assert_close(v, weights[k], rtol=0, atol=0)

    def test_fit_only_changes_new_head_and_is_rng_isolated(self):
        net = Network()
        head = net.heads[1]
        head.requires_grad_(False)  # post-merge path must also work
        old = {k: v.clone() for k, v in net.state_dict().items()}
        x = torch.randn(16, 3, requires_grad=True)
        y = torch.arange(16) % 2
        rng = torch.get_rng_state().clone()
        steps = fit_head(head, x, y, epochs=2, batch_size=5, lr=.02,
                         weight_decay=0., scale=20., margin=.1, seed=1994)
        self.assertEqual(steps, 8)
        self.assertFalse(head.weight.requires_grad)
        self.assertIsNone(head.weight.grad)
        self.assertIsNone(x.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(torch.equal(head.weight, old['heads.1.weight']))
        for k, v in net.state_dict().items():
            if k != 'heads.1.weight':
                torch.testing.assert_close(v, old[k], rtol=0, atol=0)

    def hook(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_prepare_incremental_head')
        ns = {'np': np, 'torch': torch, 'DataLoader': DataLoader, 'logging': logging}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<head-hook>', 'exec'), ns)
        return ns[method.name]

    def test_default_and_task0_are_noops(self):
        self.hook()(SimpleNamespace(_cur_task=1, args={}), None, 'pre')
        self.hook()(SimpleNamespace(_cur_task=0, args={'head_start_init': 'prototype',
                                                    'head_start_epochs': 5}), None, 'pre')

    def test_hook_order(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == 'incremental_train')
        source = ast.unparse(method)
        self.assertLess(source.index("self._prepare_incremental_head(data_manager, 'pre')"),
                        source.index('self._train(self.train_loader'))
        self.assertLess(source.index('self._train(self.train_loader'),
                        source.index("self._prepare_incremental_head(data_manager, 'post')"))
        self.assertLess(source.index("self._prepare_incremental_head(data_manager, 'post')"),
                        source.index('self._compute_class_mean'))

    def test_real_hook_pre_and_post_fit_preserve_other_parameters_and_rng(self):
        for stage in ('pre', 'post'):
            with self.subTest(stage=stage):
                net = Network().train()
                data = TensorDataset(torch.arange(8), torch.randn(8, 3), torch.tensor([2, 3]*4))
                requests = []
                def get_dataset(classes, source, mode):
                    requests.append((classes.tolist(), source, mode))
                    return data
                learner = SimpleNamespace(_cur_task=1, _known_classes=2, _total_classes=4,
                                          args={'head_start_epochs': 1, 'head_start_stage': stage},
                                          _network=net, _device='cpu', batch_size=4,
                                          lrate=.02, weight_decay=0., scale=20., margin=.1)
                old = {k: v.clone() for k, v in net.state_dict().items()}
                rng = torch.get_rng_state().clone()
                self.hook()(learner, SimpleNamespace(get_dataset=get_dataset), stage)
                self.assertEqual(requests, [([2, 3], 'train', 'test')])
                self.assertEqual(net.tasks, [0, 0] if stage == 'pre' else [1, 1])
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertFalse(torch.equal(net.heads[1].weight, old['heads.1.weight']))
                for k, v in net.state_dict().items():
                    if k != 'heads.1.weight':
                        torch.testing.assert_close(v, old[k], rtol=0, atol=0)
