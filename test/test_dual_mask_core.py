import sys
import types
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ideas.dual_mask_branch.attention import (  # noqa: E402
    Attention_LoRA,
    _energy_coverage_mask,
    _energy_coverage_with_ratio_floor_mask,
    _masked_top_ratio_mask,
    _select_svd_rank,
    _top_ratio_mask,
)
from ideas.dual_mask_branch.metrics import (  # noqa: E402
    build_prototypes,
    functional_merge_diagnostics,
    prototype_accuracy,
    select_functional_merge_candidate,
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
        "dual_mask_conflict_ratio": 0.1,
        "dual_mask_conflict_strength": 0.5,
        "dual_mask_conflict_energy_adaptive": False,
        "dual_mask_conflict_energy_ratio_floor": True,
        "dual_mask_private_conflict_mode": "global",
        "dual_mask_competence_adaptive": False,
        "dual_mask_protect_strength_mode": "legacy_linear",
        "lora_A_init": "kaiming",
    }
    args.update(overrides)
    return args


class MaskSelectionTests(unittest.TestCase):
    def test_functional_merge_diagnostics_detect_anchor_damage(self):
        anchor = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [1.0, 0.0],
             [0.0, 1.0], [0.1, 0.9], [0.0, 1.0]]
        )
        targets = torch.tensor([0, 0, 0, 1, 1, 1])
        indices = torch.arange(6)

        safe = functional_merge_diagnostics(
            anchor,
            anchor.clone(),
            targets,
            indices,
            holdout_mod=3,
            scale=10.0,
        )
        damaged_features = anchor.clone()
        damaged_features[0] = torch.tensor([0.0, 1.0])
        damaged_features[3] = torch.tensor([1.0, 0.0])
        damaged = functional_merge_diagnostics(
            anchor,
            damaged_features,
            targets,
            indices,
            holdout_mod=3,
            scale=10.0,
        )

        self.assertAlmostEqual(safe["anchor_damage"], 0.0)
        self.assertGreater(damaged["anchor_damage"], safe["anchor_damage"])

    def test_functional_merge_selector_respects_damage_budget(self):
        candidates = [
            {
                "beta": 0.0,
                "current_accuracy": 1.0,
                "current_loss": 0.1,
                "anchor_damage": 0.2,
            },
            {
                "beta": 0.5,
                "current_accuracy": 0.9,
                "current_loss": 0.2,
                "anchor_damage": 0.01,
            },
        ]

        selected = select_functional_merge_candidate(
            candidates,
            tolerance=0.05,
        )
        self.assertEqual(selected["beta"], 0.5)

    def test_conflict_distribution_diagnostics(self):
        uniform_score = torch.ones(2, 5)
        uniform_mask = torch.zeros_like(uniform_score)
        uniform_mask[:, 0] = 1.0
        uniform_entropy, uniform_top_energy = (
            Attention_LoRA._conflict_distribution_stats(
                uniform_score,
                uniform_mask,
            )
        )
        concentrated_score = torch.zeros(2, 5)
        concentrated_score[0, 0] = 1.0
        concentrated_mask = torch.zeros_like(concentrated_score)
        concentrated_mask[0, 0] = 1.0
        concentrated_entropy, concentrated_top_energy = (
            Attention_LoRA._conflict_distribution_stats(
                concentrated_score,
                concentrated_mask,
            )
        )

        self.assertAlmostEqual(uniform_entropy.item(), 1.0)
        self.assertAlmostEqual(uniform_top_energy.item(), 0.2)
        self.assertAlmostEqual(
            Attention_LoRA._conflict_energy50_ratio(uniform_score).item(),
            0.5,
        )
        self.assertAlmostEqual(concentrated_entropy.item(), 0.0)
        self.assertAlmostEqual(concentrated_top_energy.item(), 1.0)
        self.assertAlmostEqual(
            Attention_LoRA._conflict_energy50_ratio(concentrated_score).item(),
            0.1,
        )

    def test_conflict_gate_suppression_excludes_other_gates(self):
        raw_delta = torch.ones(3, 2)
        conflict_mask = torch.ones_like(raw_delta)
        suppression = Attention_LoRA._conflict_gate_suppression(
            raw_delta,
            conflict_mask,
            torch.tensor(0.5),
        )

        self.assertAlmostEqual(suppression, 0.5)

    def test_energy_coverage_uses_smallest_sufficient_set(self):
        score = torch.tensor([[4.0, 3.0], [2.0, 1.0]])

        mask = _energy_coverage_mask(score, coverage=0.75)

        self.assertTrue(torch.equal(mask, torch.tensor([[1.0, 1.0], [1.0, 0.0]])))

    def test_conflict_energy_adaptive_covers_half_with_ratio_floor(self):
        score = torch.arange(1, 101, dtype=torch.float32).reshape(10, 10)

        mask = _energy_coverage_with_ratio_floor_mask(
            score,
            ratio=0.1,
            coverage=0.5,
        )

        selected_energy = score[mask.bool()].sum() / score.sum()
        self.assertGreaterEqual(mask.float().mean().item(), 0.1)
        self.assertGreaterEqual(selected_energy.item(), 0.5)

    def test_conflict_energy_adaptive_zero_ratio_still_disables_mask(self):
        score = torch.arange(1, 101, dtype=torch.float32).reshape(10, 10)

        mask = _energy_coverage_with_ratio_floor_mask(
            score,
            ratio=0.0,
            coverage=0.5,
        )

        self.assertEqual(mask.count_nonzero().item(), 0)

    def test_zero_conflict_ratio_disables_conflict_mask(self):
        score = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        mask = _top_ratio_mask(score, ratio=0.0)

        self.assertEqual(mask.count_nonzero().item(), 0)

    def test_masked_top_ratio_uses_only_valid_coordinates(self):
        score = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
        valid_mask = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        )

        mask = _masked_top_ratio_mask(score, valid_mask, ratio=0.5)

        self.assertTrue(torch.equal(mask, torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
        )))

    def test_conflict_mask_uses_configured_top_ratio(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_conflict_ratio=0.25,
            )
        )
        module.set_pretrained_competence(0.5)
        module.w0_importance.fill_(1.0)
        delta = torch.arange(
            1,
            module.qkv.weight.numel() + 1,
            dtype=module.qkv.weight.dtype,
        ).reshape_as(module.qkv.weight)

        score, mask = module._joint_conflict(delta)
        expected = _top_ratio_mask(score, ratio=0.25)

        self.assertTrue(torch.equal(mask, expected))

    def test_conflict_energy_adaptive_expands_diffuse_conflict_mask(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_conflict_ratio=0.1,
                dual_mask_conflict_energy_adaptive=True,
            )
        )
        module.w0_importance.fill_(1.0)
        delta = torch.arange(
            1,
            module.qkv.weight.numel() + 1,
            dtype=module.qkv.weight.dtype,
        ).reshape_as(module.qkv.weight)

        score, mask = module._joint_conflict(delta)
        fixed_mask = _top_ratio_mask(score, ratio=0.1)

        self.assertGreaterEqual(mask.count_nonzero().item(), fixed_mask.count_nonzero().item())
        self.assertGreaterEqual(
            (score * mask).sum().item() / score.sum().item(),
            0.5,
        )

    def test_conflict_energy_adaptive_can_use_pure_energy50_without_ratio_floor(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_conflict_ratio=0.1,
                dual_mask_conflict_energy_adaptive=True,
                dual_mask_conflict_energy_ratio_floor=False,
            )
        )
        module.w0_importance.fill_(1.0)
        delta = torch.arange(
            module.qkv.weight.numel(),
            0,
            -1,
            dtype=module.qkv.weight.dtype,
        ).reshape_as(module.qkv.weight)
        delta.flatten()[0] = 1000.0

        score, mask = module._joint_conflict(delta)
        selected_energy = (score * mask).sum() / score.sum()

        self.assertLess(mask.float().mean().item(), 0.1)
        self.assertGreaterEqual(selected_energy.item(), 0.5)

    def test_old_overlap_conflict_controller_only_strengthens_beta(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(dual_mask_conflict_old_overlap_adaptive=True)
        )
        module.set_pretrained_old_overlap_risk(0.2)

        ratio, strength = module._conflict_parameters()

        self.assertAlmostEqual(ratio, 0.1)
        self.assertAlmostEqual(strength, 0.6)

    def test_private_none_mode_leaves_only_plastic_mask(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        module._init_params(
            make_args(dual_mask_private_conflict_mode="none")
        )
        module.cur_task = 1
        module.general_mask.zero_()
        module.general_mask[:, :2] = 1.0
        module.w0_importance.fill_(1.0)
        delta = torch.ones_like(module.qkv.weight)

        safe_delta = module._safe_delta(delta, isolated=True)

        self.assertTrue(torch.equal(safe_delta, 1.0 - module.general_mask))

    def test_private_delta_uses_only_plastic_and_conflict_gates(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        module._init_params(
            make_args(
                dual_mask_private_conflict_mode="global",
                dual_mask_conflict_ratio=1.0,
                dual_mask_conflict_strength=0.5,
            )
        )
        module.cur_task = 1
        module.general_mask.zero_()
        module.general_mask[:, :2] = 1.0
        module.w0_importance.fill_(1.0)
        delta = torch.arange(
            1,
            module.qkv.weight.numel() + 1,
            dtype=module.qkv.weight.dtype,
        ).reshape_as(module.qkv.weight)

        module.effective_protect_strength = 0.0
        without_protection = module._safe_delta(delta, isolated=True)
        module.effective_protect_strength = 1.0
        with_full_protection = module._safe_delta(delta, isolated=True)

        expected = 0.5 * delta * (1.0 - module.general_mask)
        self.assertTrue(torch.equal(without_protection, expected))
        self.assertTrue(torch.equal(with_full_protection, expected))

    def test_private_plastic_mode_selects_within_plastic_region(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
        module._init_params(
            make_args(
                dual_mask_private_conflict_mode="plastic",
                dual_mask_conflict_ratio=0.5,
            )
        )
        module.general_mask.zero_()
        module.general_mask[:, :2] = 1.0
        module.w0_importance.fill_(1.0)
        delta = torch.arange(
            1,
            module.qkv.weight.numel() + 1,
            dtype=module.qkv.weight.dtype,
        ).reshape_as(module.qkv.weight)

        _, mask = module._joint_conflict(
            delta,
            valid_mask=1.0 - module.general_mask,
        )

        self.assertEqual(
            torch.count_nonzero(mask * module.general_mask).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(mask).item(),
            int(torch.count_nonzero(1.0 - module.general_mask).item() * 0.5),
        )

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

    def test_soft_svd_importance_uses_all_energy_weighted_directions(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        weight = torch.arange(1, 49, dtype=torch.float32).reshape(12, 4)

        score = module._soft_svd_importance(weight)

        u, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
        spectral_weights = singular_values.pow(2)
        spectral_weights = spectral_weights / spectral_weights.sum()
        row_score = (u.pow(2) * spectral_weights.unsqueeze(0)).sum(dim=1)
        col_score = (vh.t().pow(2) * spectral_weights.unsqueeze(0)).sum(dim=1)
        expected = row_score.unsqueeze(1) * col_score.unsqueeze(0)

        self.assertTrue(torch.allclose(score, expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(module.last_svd_rank, singular_values.numel())
        self.assertAlmostEqual(module.last_svd_energy_coverage, 1.0)

    def test_soft_svd_mode_does_not_use_hard_rank_cap(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        with torch.no_grad():
            module.qkv.weight.copy_(
                torch.arange(1, 49, dtype=torch.float32).reshape(12, 4)
            )
        module._init_params(
            make_args(
                dual_mask_importance="soft_svd",
                dual_mask_svd_rank=1,
            )
        )

        score = module._combined_importance()

        self.assertEqual(score.shape, module.qkv.weight.shape)
        self.assertEqual(module.last_svd_rank, 4)
        self.assertAlmostEqual(module.last_svd_energy_coverage, 1.0)

    def test_removed_importance_modes_are_rejected(self):
        removed_modes = (
            "soft_topk",
            "grad",
            "gradient",
            "svd_grad",
            "grad_svd",
            "hybrid",
            "soft_svd_grad",
            "soft_spectral_grad",
            "soft_topk_grad",
            "spectral_loss",
            "soft_spectral",
        )
        for mode in removed_modes:
            with self.subTest(mode=mode):
                module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
                with self.assertRaisesRegex(
                    ValueError,
                    "Unsupported dual_mask_importance",
                ):
                    module._init_params(
                        make_args(dual_mask_importance=mode)
                    )

    def test_task0_gate_mode_is_validated(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported dual_mask_task0_gate_mode",
        ):
            module._init_params(make_args(dual_mask_task0_gate_mode="invalid"))

    def test_protect_strength_mode_is_validated(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported dual_mask_protect_strength_mode",
        ):
            module._init_params(
                make_args(dual_mask_protect_strength_mode="invalid")
            )


class PretrainedAnchorTests(unittest.TestCase):
    def test_task0_only_anchor_regularization_skips_later_tasks(self):
        if "easydict" not in sys.modules:
            easydict = types.ModuleType("easydict")

            class EasyDict(dict):
                __getattr__ = dict.__getitem__
                __setattr__ = dict.__setitem__

            easydict.EasyDict = EasyDict
            sys.modules["easydict"] = easydict

        from ideas.dual_mask_branch.learner import Learner

        module = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=2)
        module._init_params(
            make_args(
                dual_mask_task0_gate_mode="unmasked",
                slora_gamma=1.0,
            )
        )
        learner = Learner.__new__(Learner)
        learner.args = make_args(
            dual_mask_reg_weight=0.0,
            dual_mask_anchor_reg_enabled=True,
            dual_mask_anchor_reg_weight=0.6,
            dual_mask_anchor_reg_task0_only=True,
        )
        learner._network = torch.nn.ModuleList([module])

        module.before_task(0)
        module.set_task_and_stage(0, layer_idx=0)
        learner._cur_task = 0
        self.assertIsNotNone(learner._extra_training_loss())
        self.assertIn("anchor_reg", learner._last_training_loss_metrics)

        module.before_task(1)
        module.set_task_and_stage(1, layer_idx=0)
        learner._cur_task = 1
        self.assertIsNone(learner._extra_training_loss())
        self.assertEqual(learner._last_training_loss_metrics, {})

        learner.args.pop("dual_mask_anchor_reg_task0_only")
        self.assertIsNotNone(learner._extra_training_loss())
        self.assertIn("anchor_reg", learner._last_training_loss_metrics)

    def test_anchor_regularization_uses_effective_weight_and_reaches_lora(self):
        module = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_task0_gate_mode="unmasked",
                slora_gamma=1.0,
            )
        )
        module.before_task(0)
        module.set_task_and_stage(0, layer_idx=0)
        unit = module.S_lora[0]
        with torch.no_grad():
            module.pretrained_weight.fill_(1.0)
            module.qkv.weight.copy_(module.pretrained_weight)
            module.qkv.weight[0, 0] += 1.0
            unit.A_weight.copy_(torch.eye(2))
            unit.B_weight.zero_()
            unit.B_weight[0, 0] = 1.0

        regularization = module.anchor_regularization()

        # Only one of 12 weights drifts by 2: (1 accumulated + 1 current).
        self.assertAlmostEqual(regularization.item(), 4.0 / 12.0, places=6)
        regularization.backward()
        self.assertAlmostEqual(unit.B_weight.grad[0, 0].item(), 4.0 / 12.0, places=6)
        self.assertAlmostEqual(unit.A_weight.grad[0, 0].item(), 4.0 / 12.0, places=6)

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

    def test_fixed_balanced_policy_matches_legacy_defaults(self):
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

    def test_competence_strength_mode_uses_control_competence_directly(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=4, n_tasks=2)
        module._init_params(
            make_args(
                dual_mask_competence_adaptive=True,
                dual_mask_plasticity_adaptive=True,
                dual_mask_protect_strength_mode="competence",
            )
        )

        module.set_pretrained_competence(1.0, plasticity_demand=0.25)

        self.assertAlmostEqual(module.pretrained_control_competence, 0.75)
        self.assertAlmostEqual(module.effective_protect_strength, 0.75)

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

    def test_plasticity_demand_reduces_protection_and_restores_private_rank(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=4, n_tasks=2)
        module._init_params(
            make_args(
                dual_mask_competence_adaptive=True,
                dual_mask_plasticity_adaptive=True,
            )
        )

        module.set_pretrained_competence(1.0, plasticity_demand=0.3)

        self.assertAlmostEqual(module.pretrained_competence, 1.0)
        self.assertAlmostEqual(module.pretrained_plasticity_demand, 0.3)
        self.assertAlmostEqual(module.pretrained_control_competence, 0.7)
        self.assertAlmostEqual(module.effective_energy_coverage, 0.875)
        self.assertAlmostEqual(module.effective_protect_strength, 0.72)
        self.assertEqual(module.current_private_rank, 2)

    def test_plasticity_demand_is_noop_when_controller_is_disabled(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=4, n_tasks=2)
        module._init_params(
            make_args(
                dual_mask_competence_adaptive=True,
                dual_mask_plasticity_adaptive=False,
            )
        )

        module.set_pretrained_competence(1.0, plasticity_demand=0.3)

        self.assertAlmostEqual(module.pretrained_control_competence, 1.0)
        self.assertAlmostEqual(module.effective_energy_coverage, 0.95)
        self.assertAlmostEqual(module.effective_protect_strength, 0.9)
        self.assertEqual(module.current_private_rank, 1)


class LoRALifecycleTests(unittest.TestCase):

    def test_functional_merge_strength_overrides_forward_and_merge(self):
        module = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_conflict_ratio=1.0,
                dual_mask_conflict_strength=0.5,
            )
        )
        module.general_mask.zero_()
        module.w0_importance.fill_(1.0)
        delta = (
            torch.arange(module.qkv.weight.numel(), dtype=module.qkv.weight.dtype)
            .reshape_as(module.qkv.weight)
            .add_(1.0)
        )

        module.set_functional_merge_strength(0.0)
        no_suppression = module._safe_delta(delta, isolated=False)
        _, merge_strength = module._conflict_parameters()
        self.assertEqual(merge_strength, 0.0)
        self.assertTrue(torch.equal(no_suppression, delta))

        module.set_functional_merge_strength(0.5)
        suppressed = module._safe_delta(delta, isolated=False)
        _, merge_strength = module._conflict_parameters()
        self.assertEqual(merge_strength, 0.5)
        self.assertTrue(torch.allclose(suppressed, delta * 0.5))

    def test_conflict_merge_modes_are_explicitly_validated(self):
        for mode in ("none", "suppress", "relocate", "suppress_relocate"):
            module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
            module._init_params(
                make_args(dual_mask_conflict_merge_mode=mode)
            )
            self.assertEqual(module.dual_mask_conflict_merge_mode, mode)

        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        with self.assertRaises(ValueError):
            module._init_params(
                make_args(dual_mask_conflict_merge_mode="unknown")
            )

    def test_low_rank_relocation_stays_in_safe_support_and_recovers_activation(self):
        module = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_relocation_steps=80,
                dual_mask_relocation_lr=0.1,
            )
        )
        module.before_task(0)
        unit = module.S_lora[0]
        with torch.no_grad():
            unit.A_weight.copy_(torch.eye(2))
            unit.B_weight.zero_()
            unit.B_weight[0, 0] = 1.0
            unit.B_weight[0, 1] = 1.0

        target_delta = torch.zeros_like(module.qkv.weight)
        target_delta[0, 0] = 1.0
        safe_support = torch.zeros_like(target_delta)
        safe_support[:, 1] = 1.0
        inputs = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [0.5, 0.5]]
        )

        relocated, metrics = module._fit_low_rank_relocation(
            unit,
            gamma=1.0,
            target_delta=target_delta,
            safe_support=safe_support,
            inputs=inputs,
        )

        self.assertEqual(relocated[:, 0].count_nonzero().item(), 0)
        self.assertLess(metrics["activation_error"], 0.05)
        self.assertGreater(metrics["recovered_energy"], 0.95)

    def test_low_energy_relocation_uses_relative_not_absolute_fit_scale(self):
        module = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_relocation_steps=20,
                dual_mask_relocation_lr=0.1,
            )
        )
        module.before_task(0)
        unit = module.S_lora[0]
        with torch.no_grad():
            unit.A_weight.copy_(torch.eye(2))
            unit.B_weight.zero_()
            # Match the scale observed for trained LoRA B weights. The safe
            # column can represent the target exactly, so recovery should not
            # depend on using an artificially unit-scale basis.
            unit.B_weight[0, 0] = 1e-4
            unit.B_weight[0, 1] = 1e-4

        target_delta = torch.zeros_like(module.qkv.weight)
        target_delta[0, 0] = 1e-4
        safe_support = torch.zeros_like(target_delta)
        safe_support[:, 1] = 1.0
        inputs = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [0.5, 0.5]]
        )

        _, metrics = module._fit_low_rank_relocation(
            unit,
            gamma=1.0,
            target_delta=target_delta,
            safe_support=safe_support,
            inputs=inputs,
        )

        self.assertGreater(metrics["recovered_energy"], 0.9)
        self.assertLess(metrics["activation_error"], 0.35)

    def test_default_merge_mode_is_identical_to_safe_delta(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_conflict_ratio=1.0,
                dual_mask_conflict_strength=0.5,
            )
        )
        module.general_mask.fill_(1.0)
        module.w0_importance.fill_(1.0)
        module.effective_protect_strength = 0.25
        raw_delta = torch.arange(1, 49, dtype=torch.float32).reshape(12, 4)

        merged = module._compose_merge_delta(
            raw_delta,
            isolated=False,
            conflict_ratio=1.0,
            conflict_strength=0.5,
        )

        self.assertEqual(module.dual_mask_conflict_merge_mode, "suppress")
        self.assertTrue(
            torch.equal(
                merged,
                module._safe_delta(
                    raw_delta,
                    isolated=False,
                    conflict_ratio=1.0,
                    conflict_strength=0.5,
                ),
            )
        )

    def test_prepared_relocation_is_merged_once_and_then_cleared(self):
        module = Attention_LoRA(dim=2, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_conflict_merge_mode="suppress_relocate",
                dual_mask_conflict_ratio=0.1,
                dual_mask_conflict_strength=0.5,
                dual_mask_relocation_steps=80,
                dual_mask_relocation_lr=0.1,
            )
        )
        module.before_task(0)
        module.general_mask.zero_()
        module.general_mask[:, 0] = 1.0
        module.w0_importance.zero_()
        module.w0_importance[0, 0] = 1.0
        module.effective_protect_strength = 0.0
        unit = module.S_lora[0]
        with torch.no_grad():
            unit.A_weight.copy_(torch.eye(2))
            unit.B_weight.zero_()
            unit.B_weight[0, 0] = 1.0
            unit.B_weight[0, 1] = 1.0

        inputs = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [0.5, 0.5]]
        )
        raw_delta = module.slora_gamma * (unit.B_weight @ unit.A_weight)
        expected_suppressed = module._safe_delta(
            raw_delta,
            isolated=False,
            conflict_ratio=0.1,
            conflict_strength=0.5,
        )
        weight_before = module.qkv.weight.detach().clone()

        module.prepare_conflict_relocation(0, inputs)
        relocation = module._pending_relocations["S"].clone()
        module.after_task(0)

        self.assertGreater(relocation.norm().item(), 0.0)
        self.assertTrue(
            torch.allclose(
                module.qkv.weight,
                weight_before + expected_suppressed + relocation,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        self.assertEqual(module._pending_relocations, {})

    def test_relocation_collection_caches_bounded_attention_inputs(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_conflict_merge_mode="relocate",
                dual_mask_relocation_vectors=3,
            )
        )
        module.before_task(0)
        inputs = torch.randn(2, 4, 4, requires_grad=True)

        module.begin_relocation_input_collection()
        module(inputs, task=0)
        cached = module.end_relocation_input_collection()

        self.assertEqual(cached.shape, (3, 4))
        self.assertFalse(cached.requires_grad)

    def test_task0_gate_ablation_modes_leave_later_tasks_unchanged(self):
        delta = torch.arange(1, 49, dtype=torch.float32).reshape(12, 4)

        def build_module(mode: str):
            module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=2)
            module._init_params(
                make_args(
                    dual_mask_task0_gate_mode=mode,
                    dual_mask_conflict_ratio=1.0,
                    dual_mask_conflict_strength=0.5,
                )
            )
            module.general_mask.fill_(1.0)
            module.w0_importance.copy_(
                torch.arange(1, 49, dtype=delta.dtype).reshape_as(delta)
            )
            module.effective_protect_strength = 0.25
            return module

        full = build_module("full")
        self.assertTrue(
            torch.allclose(
                full._safe_delta(delta, isolated=False),
                0.375 * delta,
            )
        )

        protect_only = build_module("protect_only")
        self.assertTrue(
            torch.allclose(
                protect_only._safe_delta(delta, isolated=False),
                0.75 * delta,
            )
        )

        unmasked = build_module("unmasked")
        self.assertTrue(torch.equal(unmasked._safe_delta(delta, isolated=False), delta))

        for module in (full, protect_only, unmasked):
            module.cur_task = 1
            self.assertTrue(
                torch.allclose(
                    module._safe_delta(delta, isolated=False),
                    0.375 * delta,
                )
            )

    def test_task0_unmasked_mode_disables_dual_mask_regularization(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(make_args(dual_mask_task0_gate_mode="unmasked"))
        module.before_task(0)
        unit = module.S_lora[0]
        with torch.no_grad():
            unit.B_weight.fill_(1.0)

        loss = module._joint_conflict_regularization(unit, isolated=False)

        self.assertTrue(loss.requires_grad)
        self.assertEqual(loss.item(), 0.0)

    def test_task0_protect_only_regularization_excludes_conflict_term(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_task0_gate_mode="protect_only",
                dual_mask_conflict_ratio=1.0,
            )
        )
        module.before_task(0)
        module.general_mask.fill_(1.0)
        module.w0_importance.fill_(1.0)
        unit = module.S_lora[0]
        with torch.no_grad():
            unit.B_weight.fill_(1.0)

        safe_delta = module._safe_delta(
            unit.B_weight @ unit.A_weight,
            isolated=False,
        )
        expected = (module.w0_importance * safe_delta.pow(2)).mean()

        self.assertTrue(
            torch.allclose(
                module._joint_conflict_regularization(unit, isolated=False),
                expected,
            )
        )

    def test_balanced_strength_is_uniform_inside_protect_mask(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_conflict_ratio=0.0,
                dual_mask_conflict_strength=0.0,
                dual_mask_competence_adaptive=True,
            )
        )
        module.set_pretrained_competence(1.0)
        module.general_mask.fill_(1.0)
        importance = torch.linspace(0.0, 1.0, steps=48).reshape(12, 4)
        module.w0_importance.copy_(importance)
        delta = torch.ones_like(module.qkv.weight)

        safe_delta = module._safe_delta(delta, isolated=False)

        self.assertTrue(torch.allclose(safe_delta, torch.full_like(delta, 0.1)))

    def test_unprotected_positions_are_not_suppressed(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(
                dual_mask_conflict_ratio=0.0,
                dual_mask_conflict_strength=0.0,
            )
        )
        module.general_mask.zero_()
        module.w0_importance.fill_(1.0)
        delta = torch.ones_like(module.qkv.weight)

        safe_delta = module._safe_delta(delta, isolated=False)

        self.assertTrue(torch.equal(safe_delta, delta))

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

    def test_energy_adaptive_conflict_merge_preserves_output(self):
        torch.manual_seed(0)
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        module._init_params(
            make_args(dual_mask_conflict_energy_adaptive=True)
        )
        module.before_task(0)
        module.eval()
        with torch.no_grad():
            module.S_lora[0].B.weight.normal_(mean=0.0, std=0.05)

        inputs = torch.randn(2, 3, 4)
        output_before_merge = module(inputs, task=0)
        module.after_task(0)
        output_after_merge = module(inputs, task=0)

        self.assertTrue(torch.isfinite(output_after_merge).all())
        self.assertGreaterEqual(module.last_effective_conflict_ratio.item(), 0.1)
        self.assertTrue(
            torch.allclose(
                output_before_merge,
                output_after_merge,
                atol=1e-5,
                rtol=1e-5,
            )
        )
