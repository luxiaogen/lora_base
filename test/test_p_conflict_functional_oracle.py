import copy
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from models.attention import Attention_LoRA
from utils.p_conflict_functional_oracle import (
    balanced_holdout_indices,
    compare_functional_components,
    component_specs,
    energy_matched_plan,
    materialize_component,
    plan_energy,
    reconstruct_conflict_component,
    temporary_plan,
)


class TinyData(Dataset):
    labels = np.array([0, 0, 0, 0, 2, 2, 2, 2])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        random.random(); np.random.rand(); torch.rand(1)
        return index, torch.eye(6)[self.labels[index]], int(self.labels[index])


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(torch.eye(6))
        self._p_conflict_functional_state = {
            "A": torch.tensor([[1., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.]]),
            "B": torch.tensor([
                [0.2, 0.0], [0.0, 0.0], [0.0, 0.2],
                [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            ]),
            "gate": torch.ones(6, 6),
            "gamma": 1.0,
            "rank_groups": 2,
        }


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()

    def interface(self, x):
        return self.attn.qkv(x)


class FunctionalOracleTests(unittest.TestCase):
    def test_balanced_holdout_is_disjoint_and_deterministic(self):
        labels = np.repeat(np.arange(3), 6)
        selector, evaluator = balanced_holdout_indices(labels, per_class=4)
        self.assertEqual(selector, [0, 1, 6, 7, 12, 13])
        self.assertEqual(evaluator, [2, 3, 8, 9, 14, 15])
        self.assertFalse(set(selector) & set(evaluator))

    def test_energy_matching_never_amplifies(self):
        specs = [(0, "q", i) for i in range(3)]
        norms = {specs[0]: 3.0, specs[1]: 2.0, specs[2]: 1.0}
        plan = energy_matched_plan(specs, norms, 5.0, specs)
        self.assertAlmostEqual(plan_energy(plan, norms), 5.0, places=6)
        self.assertTrue(all(0.0 < scale <= 1.0 for scale in plan.values()))

    def test_temporary_plan_restores_exact_rows(self):
        module = TinyAttention()
        original = module.qkv.weight.detach().clone()
        spec = (0, "q", 0)
        with self.assertRaises(RuntimeError):
            with temporary_plan([module], {spec: 1.0}):
                self.assertFalse(torch.equal(original, module.qkv.weight))
                raise RuntimeError("test")
        self.assertTrue(torch.equal(original, module.qkv.weight))

    def test_read_only_diagnostic_restores_state_and_rng(self):
        net = TinyNet()
        original = copy.deepcopy(net.state_dict())
        selector, evaluator = balanced_holdout_indices(TinyData.labels, per_class=4)
        selector_loader = DataLoader(Subset(TinyData(), selector), batch_size=2, shuffle=False)
        evaluator_loader = DataLoader(Subset(TinyData(), evaluator), batch_size=2, shuffle=False)
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        expected = (random.random(), np.random.rand(), torch.rand(2))
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        report = compare_functional_components(
            net, [net.attn], selector_loader, evaluator_loader, "cpu", [2, 2], random_seed=7,
        )
        actual = (random.random(), np.random.rand(), torch.rand(2))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertEqual(report["components_total"], 6)
        self.assertEqual(report["restored_max_abs_logit_diff"], 0.0)
        for key, value in original.items():
            self.assertTrue(torch.equal(value, net.state_dict()[key]))

    def test_attention_capture_reconstructs_retained_conflict(self):
        torch.manual_seed(11)
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        module._init_params(dict(
            use_slora=True, use_plora=True, lora_A_init="kaiming",
            slora_gamma=0.5, plora_gamma=0.75,
            dual_mask_conflict_energy_adaptive=True,
            dual_mask_conflict_energy_ratio_floor=True,
            dual_mask_p_functional_oracle_diagnostic=True,
            dual_mask_p_functional_rank_groups=2,
        ))
        module.before_task(0)
        module.after_task(0)
        module.before_task(1)
        with torch.no_grad():
            module.P_lora[1].B.weight.normal_(0, 0.1)
            module.S_lora[1].B.weight.normal_(0, 0.1)
        raw = module.plora_gamma * (module.P_lora[1].B_weight @ module.P_lora[1].A_weight)
        ratio, strength = module._conflict_parameters()
        safe = module._compose_merge_delta(raw, True, ratio, strength)
        _, conflict = module._merge_base_and_conflict(raw, True, ratio)
        expected = (safe * conflict).float()
        module.after_task(1)
        self.assertIsNotNone(module._p_conflict_functional_state)
        self.assertEqual(len(component_specs([module])), 6)
        self.assertTrue(torch.allclose(reconstruct_conflict_component(module), expected, atol=1e-6))
        self.assertIsNone(module.P_lora[1])


if __name__ == "__main__":
    unittest.main()
