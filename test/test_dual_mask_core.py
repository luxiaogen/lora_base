import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ideas.dual_mask_branch.attention import (  # noqa: E402
    Attention_LoRA,
    _energy_coverage_mask,
    _select_svd_rank,
    _top_ratio_mask,
)
from ideas.dual_mask_branch.metrics import (  # noqa: E402
    build_prototypes,
    prototype_accuracy,
    split_prototype_competence,
)
from main import apply_overrides  # noqa: E402
# from trainer import _resolve_devices  # noqa: E402


def make_args(**overrides):
    args = {
        "use_slora": True,
        "use_plora": True,
        "dual_mask_importance": "svd",
        "dual_mask_general_ratio": 0.4,
        "dual_mask_svd_rank": 2,
        "dual_mask_svd_energy_coverage": 0.0,
        "dual_mask_grad_alpha": 0.5,
        "dual_mask_conflict_ratio": 0.1,
        "dual_mask_protect_strength": 0.5,
        "dual_mask_conflict_strength": 0.5,
        "dual_mask_competence_adaptive": False,
        "dual_mask_energy_coverage_low": 0.7,
        "dual_mask_energy_coverage_high": 0.95,
        "dual_mask_protect_strength_low": 0.3,
        "dual_mask_protect_strength_high": 0.9,
        "dual_mask_private_rank_min_ratio": 0.25,
        "lora_A_init": "kaiming",
    }
    args.update(overrides)
    return args


class MaskSelectionTests(unittest.TestCase):
    def test_energy_coverage_uses_smallest_sufficient_set(self):
        score = torch.tensor([[4.0, 3.0], [2.0, 1.0]])

        mask = _energy_coverage_mask(score, coverage=0.75)

        self.assertTrue(torch.equal(mask, torch.tensor([[1.0, 1.0], [1.0, 0.0]])))

    def test_zero_conflict_ratio_disables_conflict_mask(self):
        score = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        mask = _top_ratio_mask(score, ratio=0.0)

        self.assertEqual(mask.count_nonzero().item(), 0)

    def test_svd_rank_is_selected_by_energy_coverage(self):
        singular_values = torch.tensor([4.0, 3.0, 1.0])

        rank = _select_svd_rank(singular_values, max_rank=3, energy_coverage=0.9)

        self.assertEqual(rank, 2)

    def test_svd_energy_selection_respects_configured_rank_cap(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_svd_rank=2,
                dual_mask_svd_energy_coverage=0.999,
            )
        )

        module._svd_importance(torch.eye(4).repeat(3, 1))

        self.assertLessEqual(module.last_svd_rank, 2)


class PretrainedAnchorTests(unittest.TestCase):
    def test_pretrained_anchor_is_immutable_after_first_capture(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        with torch.no_grad():
            module.qkv.weight.copy_(torch.arange(48, dtype=torch.float32).reshape(12, 4))
        module._init_params(make_args())
        anchor = module.pretrained_weight.clone()

        with torch.no_grad():
            module.qkv.weight.add_(10.0)
        module.capture_pretrained_anchor()

        self.assertTrue(torch.equal(module.pretrained_weight, anchor))
        self.assertGreater(module.relative_weight_drift(), 0.0)

    def test_competence_controls_protection_and_private_rank(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=4, n_tasks=2)
        module._init_params(make_args(dual_mask_competence_adaptive=True))

        module.set_pretrained_competence(1.0)

        self.assertAlmostEqual(module.effective_energy_coverage, 0.95)
        self.assertAlmostEqual(module.effective_protect_strength, 0.9)
        self.assertEqual(module.current_private_rank, 1)

        module.set_pretrained_competence(0.0)

        self.assertAlmostEqual(module.effective_energy_coverage, 0.7)
        self.assertAlmostEqual(module.effective_protect_strength, 0.3)
        self.assertEqual(module.current_private_rank, 4)

    def test_adaptive_coverage_rebuilds_mask_after_task_zero(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=4, n_tasks=2)
        module._init_params(make_args(dual_mask_competence_adaptive=True))
        module.cur_task = 1
        with torch.no_grad():
            score = torch.arange(1, 49, dtype=torch.float32).reshape(12, 4)
            module.w0_importance.copy_(score / score.max())

        module.set_pretrained_competence(0.0)
        module.rebuild_dual_masks()
        low_density = module.general_mask.float().mean().item()

        module.set_pretrained_competence(1.0)
        module.rebuild_dual_masks()
        high_density = module.general_mask.float().mean().item()

        self.assertGreater(high_density, low_density)




class LoRALifecycleTests(unittest.TestCase):
    def test_task_zero_does_not_allocate_unused_private_lora(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        module._init_params(make_args())

        module.before_task(0)

        self.assertIsNotNone(module.S_lora[0])
        self.assertIsNone(module.P_lora[0])

    def test_merge_releases_lora_without_changing_output(self):
        torch.manual_seed(0)
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        module._init_params(make_args())
        base_parameter_count = sum(parameter.numel() for parameter in module.parameters())
        module.before_task(0)
        module.eval()

        with torch.no_grad():
            module.S_lora[0].B.weight.normal_(mean=0.0, std=0.05)

        inputs = torch.randn(2, 3, 4)
        output_before_merge = module(inputs, task=0)

        module.after_task(0)
        output_after_merge = module(inputs, task=0)

        self.assertIsNone(module.S_lora[0])
        self.assertIsNone(module.P_lora[0])
        self.assertEqual(
            sum(parameter.numel() for parameter in module.parameters()),
            base_parameter_count,
        )
        self.assertTrue(
            torch.allclose(
                output_before_merge,
                output_after_merge,
                atol=1e-5,
                rtol=1e-5,
            )
        )

        module.before_task(1)
        with torch.no_grad():
            module.S_lora[1].B.weight.normal_(mean=0.0, std=0.05)
            module.P_lora[1].B.weight.normal_(mean=0.0, std=0.05)

        output_before_merge = module(inputs, task=1)
        module.after_task(1)
        output_after_merge = module(inputs, task=1)

        self.assertIsNone(module.S_lora[1])
        self.assertIsNone(module.P_lora[1])
        self.assertEqual(
            sum(parameter.numel() for parameter in module.parameters()),
            base_parameter_count,
        )
        self.assertTrue(
            torch.allclose(
                output_before_merge,
                output_after_merge,
                atol=1e-5,
                rtol=1e-5,
            )
        )