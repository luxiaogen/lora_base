import copy
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.attention import Attention_LoRA
from utils.p_region_diagnostic import matched_removals, temporary_removal, compare_regions, summarize


class TinyData(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        random.random(); np.random.rand(); torch.rand(1)
        return i, torch.eye(4)[i], i


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(torch.eye(4))
        self._p_region_removals, _ = matched_removals(torch.eye(4), torch.diag(torch.tensor([1., 1., 0., 0.])))

    def interface(self, x):
        return self.qkv(x)


class RegionTests(unittest.TestCase):
    def test_norms_support_and_zero(self):
        delta = torch.tensor([[1., 2., 3., 4.]])
        mask = torch.tensor([[1., 1., 0., 0.]])
        removals, stats = matched_removals(delta, mask)
        self.assertAlmostEqual(removals['conflict'].norm().item(), removals['nonconflict'].norm().item(), places=6)
        self.assertFalse((removals['conflict'] * (1-mask)).any())
        self.assertFalse((removals['nonconflict'] * mask).any())
        self.assertLessEqual(stats['conflict_fraction'], 0.5)
        self.assertLessEqual(stats['nonconflict_fraction'], 0.5)
        for zero_mask in (torch.zeros_like(mask), torch.ones_like(mask)):
            removals, _ = matched_removals(delta, zero_mask)
            self.assertFalse(removals['conflict'].any())
            self.assertFalse(removals['nonconflict'].any())

    def test_restoration_even_on_exception(self):
        net = TinyNet()
        original = net.qkv.weight.detach().clone()
        with self.assertRaises(RuntimeError):
            with temporary_removal([net], 'conflict'):
                self.assertFalse(torch.equal(original, net.qkv.weight))
                raise RuntimeError('test')
        self.assertTrue(torch.equal(original, net.qkv.weight))

    def test_read_only_rng_modes_and_repeatability(self):
        net = TinyNet()
        net.qkv.eval()
        original = copy.deepcopy(net.state_dict())
        modes = [m.training for m in net.modules()]
        random.seed(3); np.random.seed(3); torch.manual_seed(3)
        expected = (random.random(), np.random.rand(), torch.rand(2))
        random.seed(3); np.random.seed(3); torch.manual_seed(3)
        loader = DataLoader(TinyData(), batch_size=2)
        result = compare_regions(net, [net], loader, 'cpu', [2, 2])
        actual = (random.random(), np.random.rand(), torch.rand(2))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertEqual(result, compare_regions(net, [net], loader, 'cpu', [2, 2]))
        self.assertEqual(modes, [m.training for m in net.modules()])
        for key, value in original.items():
            self.assertTrue(torch.equal(value, net.state_dict()[key]))

    def test_corrected_broken_partition(self):
        labels = torch.tensor([0, 2])
        base = torch.tensor([[3., 0., 1., 0.], [3., 0., 1., 0.]])
        candidate = torch.tensor([[1., 0., 3., 0.], [1., 0., 3., 0.]])
        stats = summarize(base, candidate, labels, [2, 2])
        self.assertEqual(stats['old']['broken'], 1)
        self.assertEqual(stats['new']['corrected'], 1)
        self.assertEqual(stats['new']['cross_margin_change']['mean'], 4.)

    def test_actual_attention_capture_preserves_merge_and_rng(self):
        torch.manual_seed(9)
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        module._init_params(dict(use_slora=True, use_plora=True, lora_A_init='kaiming',
                                 slora_gamma=0.5, plora_gamma=0.75,
                                 dual_mask_conflict_energy_adaptive=True,
                                 dual_mask_conflict_energy_ratio_floor=True))
        module.before_task(0)
        module.after_task(0)
        module.before_task(1)
        with torch.no_grad():
            module.P_lora[1].B.weight.normal_(0, 0.1)
            module.S_lora[1].B.weight.normal_(0, 0.1)
        enabled = copy.deepcopy(module)
        enabled.p_region_diagnostic = True
        raw = enabled.plora_gamma * (enabled.P_lora[1].B_weight @ enabled.P_lora[1].A_weight)
        ratio, strength = enabled._conflict_parameters()
        safe = enabled._compose_merge_delta(raw, True, ratio, strength)
        _, mask = enabled._merge_base_and_conflict(raw, True, ratio)
        expected_removals, _ = matched_removals(safe, mask)
        inputs = torch.randn(2, 3, 4)
        module.eval(); enabled.eval()
        before = enabled(inputs, task=1)
        module.after_task(1)
        rng = torch.get_rng_state().clone()
        enabled.after_task(1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(module.qkv.weight, enabled.qkv.weight))
        self.assertTrue(torch.allclose(before, enabled(inputs, task=1), atol=1e-5))
        self.assertIsNone(enabled.P_lora[1])
        self.assertIsNotNone(enabled._p_region_removals)
        for mode in expected_removals:
            self.assertTrue(torch.equal(expected_removals[mode], enabled._p_region_removals[mode]))
        self.assertTrue(torch.equal(module(inputs, task=1), enabled(inputs, task=1)))


if __name__ == '__main__':
    unittest.main()
