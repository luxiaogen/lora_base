import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.attention import (  # noqa: E402
    Attention_LoRA,
    _global_top_ratio_threshold,
)


def make_args(**overrides):
    args = {
        "use_slora": True,
        "use_plora": True,
        "dual_mask_enabled": True,
        "dual_mask_importance": "svd",
        "dual_mask_general_ratio": 0.5,
        "dual_mask_svd_rank": 2,
        "dual_mask_conflict_ratio": 0.1,
        "dual_mask_conflict_strength": 0.5,
        "lora_A_init": "kaiming",
    }
    args.update(overrides)
    return args


class DualMaskOffAndLoRISelectorTests(unittest.TestCase):
    def test_disabled_dualmask_keeps_raw_update(self):
        module = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=2)
        module._init_params(make_args(dual_mask_enabled=False))
        module.rebuild_dual_masks()
        delta = torch.arange(12, dtype=torch.float32).reshape(6, 2)

        self.assertEqual(module.general_mask.count_nonzero().item(), 0)
        self.assertEqual(module.isolated_mask.count_nonzero().item(), 12)
        self.assertTrue(torch.equal(module._safe_delta(delta, isolated=False), delta))
        self.assertTrue(torch.equal(module._safe_delta(delta, isolated=True), delta))
        module.dual_mask_conflict_merge_mode = "protect_only"
        self.assertTrue(torch.equal(
            module._compose_merge_delta(delta, False, 0.1, 0.5),
            delta,
        ))

    def test_lori_style_threshold_is_global_across_layers(self):
        first = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        second = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        args = make_args(dual_mask_importance="lori_global_magnitude")
        first._init_params(args)
        second._init_params(args)
        with torch.no_grad():
            first.pretrained_weight.copy_(torch.arange(12).reshape(6, 2))
            second.pretrained_weight.copy_(torch.arange(12, 24).reshape(6, 2))

        threshold = _global_top_ratio_threshold(
            [first.pretrained_weight, second.pretrained_weight],
            0.1,
        )
        for module in (first, second):
            module.set_global_magnitude_threshold(threshold.item())
            module.rebuild_dual_masks()

        self.assertEqual(threshold.item(), 22.0)
        self.assertEqual(first.general_mask.count_nonzero().item(), 0)
        self.assertEqual(second.general_mask.count_nonzero().item(), 2)
        self.assertEqual(
            first.general_mask.count_nonzero().item()
            + second.general_mask.count_nonzero().item(),
            2,
        )


if __name__ == "__main__":
    unittest.main()
