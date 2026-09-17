import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from test import test_private_rank as fixtures
from utils import dual_mask_metrics


class PConflictMergeFilterTests(unittest.TestCase):
    def make_module(self, **overrides):
        defaults = dict(
            dual_mask_conflict_ratio=1.0,
            dual_mask_conflict_strength=0.5,
            dual_mask_conflict_energy_adaptive=False,
        )
        defaults.update(overrides)
        module = fixtures.PrivateRankTests().make_module(**defaults)
        module.before_task(1)
        module.set_task_and_stage(1, 0)
        module.general_mask.zero_()
        module.w0_importance.fill_(1.0)
        return module

    def test_private_gate_removes_only_private_conflict_component(self):
        module = self.make_module()
        delta = torch.arange(1, 49, dtype=torch.float32).reshape(12, 4)

        baseline_private = module._safe_delta(delta, isolated=True)
        baseline_shared = module._safe_delta(delta, isolated=False)
        self.assertTrue(hasattr(module, "set_p_conflict_merge_gate"))

        module.set_p_conflict_merge_gate(0.0)
        filtered_private = module._safe_delta(delta, isolated=True)
        filtered_shared = module._safe_delta(delta, isolated=False)

        self.assertEqual(filtered_private.count_nonzero().item(), 0)
        torch.testing.assert_close(filtered_shared, baseline_shared)
        module.set_p_conflict_merge_gate(None)
        torch.testing.assert_close(
            module._safe_delta(delta, isolated=True),
            baseline_private,
        )

    def test_filtered_private_delta_is_merged_once_and_runtime_gate_resets(self):
        module = self.make_module()
        with torch.no_grad():
            module.S_lora[1].B_weight.zero_()
            module.P_lora[1].A.weight.fill_(1.0)
            rows = torch.arange(1, 13, dtype=module.P_lora[1].B.weight.dtype)
            module.P_lora[1].B.weight.copy_(
                rows[:, None].expand_as(module.P_lora[1].B.weight)
            )
        before_weight = module.qkv.weight.detach().clone()
        self.assertTrue(hasattr(module, "set_p_conflict_merge_gate"))
        module.set_p_conflict_merge_gate(0.0)

        module.after_task(1)

        torch.testing.assert_close(module.qkv.weight, before_weight)
        self.assertIsNone(module.S_lora[1])
        self.assertIsNone(module.P_lora[1])
        self.assertIsNone(module.p_conflict_merge_gate)

    def test_margin_diagnostic_uses_deterministic_per_class_holdout(self):
        diagnostic = getattr(
            dual_mask_metrics,
            "p_conflict_merge_margin_diagnostics",
            None,
        )
        self.assertIsNotNone(diagnostic)
        indices = torch.arange(6)
        targets = torch.tensor([0, 0, 0, 1, 1, 1])
        logits_with = torch.tensor([
            [2.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        logits_without = torch.tensor([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])

        result = diagnostic(
            logits_with,
            logits_without,
            targets,
            indices,
            holdout_mod=3,
        )

        self.assertEqual(result["sample_count"], 2)
        self.assertAlmostEqual(result["margin_with"], 2.0)
        self.assertAlmostEqual(result["margin_without"], 1.0)
        self.assertAlmostEqual(result["gain"], 1.0)
        self.assertEqual(result["gate"], 1.0)

    def test_negative_margin_gain_drops_component(self):
        diagnostic = getattr(
            dual_mask_metrics,
            "p_conflict_merge_margin_diagnostics",
            None,
        )
        self.assertIsNotNone(diagnostic)
        result = diagnostic(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            holdout_mod=None,
        )

        self.assertLess(result["gain"], 0.0)
        self.assertEqual(result["gate"], 0.0)

    def test_learner_selects_each_layer_gate_before_merge(self):
        if "easydict" not in sys.modules:
            easydict = types.ModuleType("easydict")

            class EasyDict(dict):
                __getattr__ = dict.__getitem__
                __setattr__ = dict.__setitem__

            easydict.EasyDict = EasyDict
            sys.modules["easydict"] = easydict
        from methods.dlora import Learner

        class GateModule:
            def __init__(self):
                self.p_conflict_merge_gate = None

            def set_p_conflict_merge_gate(self, gate):
                self.p_conflict_merge_gate = gate

        class GateNetwork(nn.Module):
            def __init__(self, modules):
                super().__init__()
                self.gate_modules = modules

            def forward(self, inputs):
                logits = inputs.clone()
                if self.gate_modules[0].p_conflict_merge_gate == 0.0:
                    logits = logits - (inputs > 0).to(inputs)
                if self.gate_modules[1].p_conflict_merge_gate == 0.0:
                    logits = logits + (inputs > 0).to(inputs)
                return {"logits": logits}

        modules = [GateModule(), GateModule()]
        learner = Learner.__new__(Learner)
        learner._network = GateNetwork(modules)
        learner._device = torch.device("cpu")
        learner._cur_task = 1
        learner._known_classes = 2
        learner.args = {"dual_mask_competence_holdout_mod": 3}
        learner._iter_lora_modules = lambda: iter(modules)
        inputs = torch.tensor([
            [2.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ])
        loader = DataLoader(
            TensorDataset(
                torch.arange(6),
                inputs,
                torch.tensor([2, 2, 2, 3, 3, 3]),
            ),
            batch_size=3,
            shuffle=False,
        )
        self.assertTrue(hasattr(learner, "_calibrate_p_conflict_merge"))

        learner._calibrate_p_conflict_merge(loader)

        self.assertEqual(modules[0].p_conflict_merge_gate, 1.0)
        self.assertEqual(modules[1].p_conflict_merge_gate, 0.0)
        self.assertEqual(
            [item["sample_count"] for item in learner._p_conflict_merge_filter],
            [2, 2],
        )


class PConflictMergeFilterScriptTests(unittest.TestCase):
    def test_5090_script_runs_one_qkv_seed_without_data_path_override(self):
        script = Path("scripts/9_17_imgr10_p_conflict_merge_filter_5090.sh")
        self.assertTrue(script.exists())
        env = dict(os.environ, DRY_RUN="1")

        completed = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        command = completed.stdout

        self.assertIn("seed=\\[1993\\]", command)
        self.assertIn("dual_mask_p_conflict_merge_filter=true", command)
        self.assertIn("dual_mask_p_conflict_diagnostics=false", command)
        self.assertIn("ca_epochs=5", command)
        self.assertIn("dual_mask_qk_all_tasks=false", command)
        self.assertIn("dual_mask_qv_all_tasks=false", command)
        self.assertNotIn("data_path=", command)


if __name__ == "__main__":
    unittest.main()
