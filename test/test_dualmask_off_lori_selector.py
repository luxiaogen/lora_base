import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.attention import (  # noqa: E402
    Attention_LoRA,
    _global_top_ratio_masks,
    _normalize_score,
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

    def test_lori_style_selector_is_global_across_layers(self):
        first = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        second = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        args = make_args(dual_mask_importance="lori_global_magnitude")
        first._init_params(args)
        second._init_params(args)
        with torch.no_grad():
            first.pretrained_weight.copy_(torch.arange(12).reshape(6, 2))
            second.pretrained_weight.copy_(torch.arange(12, 24).reshape(6, 2))

        masks = _global_top_ratio_masks(
            [first.pretrained_weight.abs(), second.pretrained_weight.abs()],
            0.1,
        )
        for module, mask in zip((first, second), masks):
            module.set_global_protect_mask(mask)
            module.rebuild_dual_masks()

        self.assertEqual(first.general_mask.count_nonzero().item(), 0)
        self.assertEqual(second.general_mask.count_nonzero().item(), 2)
        self.assertEqual(
            first.general_mask.count_nonzero().item()
            + second.general_mask.count_nonzero().item(),
            2,
        )

    def test_global_top_ratio_masks_select_exact_count_with_ties(self):
        scores = [torch.ones(6), torch.ones(4)]

        masks = _global_top_ratio_masks(scores, 0.3)

        self.assertEqual(sum(mask.count_nonzero().item() for mask in masks), 3)
        self.assertEqual([tuple(mask.shape) for mask in masks], [(6,), (4,)])

    def test_svd_and_magnitude_global_modes_use_same_budget(self):
        svd_module = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        magnitude_module = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        weight = torch.tensor([
            [0.3923, -0.2236],
            [-0.3195, -1.2050],
            [1.0445, -0.6332],
            [0.5731, 0.5409],
            [-0.3919, -1.0427],
            [1.3186, 0.7476],
        ])

        for module, mode in (
            (svd_module, "svd_global_top10"),
            (magnitude_module, "magnitude_global_top10"),
        ):
            module._init_params(make_args(
                dual_mask_importance=mode,
                dual_mask_svd_rank=1,
            ))
            with torch.no_grad():
                module.pretrained_weight.copy_(weight)
            score = _normalize_score(module._combined_importance())
            mask = _global_top_ratio_masks([score], 0.1)[0]
            module.set_global_protect_mask(mask)
            module.rebuild_dual_masks()

        self.assertEqual(svd_module.general_mask.count_nonzero().item(), 1)
        self.assertEqual(magnitude_module.general_mask.count_nonzero().item(), 1)
        self.assertFalse(torch.equal(
            svd_module.general_mask,
            magnitude_module.general_mask,
        ))
        self.assertTrue(torch.equal(
            svd_module.isolated_mask,
            1.0 - svd_module.general_mask,
        ))

    def test_legacy_svd_does_not_require_a_global_mask(self):
        module = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        module._init_params(make_args(dual_mask_importance="svd"))

        module.rebuild_dual_masks()

        self.assertFalse(module.global_protect_mask_ready.item())
        self.assertGreater(module.general_mask.count_nonzero().item(), 0)

    def test_matched_top10_script_contains_only_the_two_importance_modes(self):
        script = (PROJECT_ROOT / "scripts" / "9_21_imgr10_importance_matched_top10_seed1993.sh").read_text()

        self.assertEqual(script.count("python main.py --config"), 2)
        self.assertEqual(script.count("dual_mask_importance=svd_global_top10"), 1)
        self.assertEqual(script.count("dual_mask_importance=magnitude_global_top10"), 1)
        self.assertNotIn("data_path=", script)


if __name__ == "__main__":
    unittest.main()
