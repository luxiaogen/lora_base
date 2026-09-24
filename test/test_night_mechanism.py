import copy
import ast
import json
import logging
import random
from pathlib import Path
import tempfile
import unittest

import torch
import numpy as np

from test import test_global_conflict_budget
from utils.dual_mask_budget import select_global_budget_masks, select_mixed_budget_masks
from utils.dual_mask_budget import select_projection_budget_masks
from utils.update_overlap import UpdateOverlapRecorder, describe, cosine


class MixedBudgetTests(unittest.TestCase):
    def test_endpoints_budget_binary_and_valid_with_ties(self):
        scores = [torch.ones(6, 4), torch.arange(24.).reshape(6, 4)]
        refs = [torch.zeros_like(s) for s in scores]
        refs[0][:2] = 1
        refs[1][-2:] = 1
        valid = [torch.ones_like(s) for s in scores]
        valid[1][0] = 0
        state = torch.get_rng_state().clone()
        for rho in (0., .25, .5, .75, 1.):
            masks = select_mixed_budget_masks(scores, refs, valid, rho)
            self.assertEqual(sum(int(m.sum()) for m in masks), 16)
            for m, v in zip(masks, valid):
                self.assertTrue(((m == 0) | (m == 1)).all())
                self.assertEqual(int((m * (1 - v)).sum()), 0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for actual, expected in zip(select_mixed_budget_masks(scores, refs, valid, 1.), refs):
            self.assertTrue(torch.equal(actual, expected))
        for actual, expected in zip(select_mixed_budget_masks(scores, refs, valid, 0.),
                                    select_global_budget_masks(scores, refs, valid)):
            self.assertTrue(torch.equal(actual, expected))

    def test_local_reservation_and_global_remainder(self):
        scores = [torch.tensor([4., 3., 2., 1.]), torch.tensor([8., 7., 6., 5.])]
        refs = [torch.tensor([1., 1., 0., 0.]) for _ in scores]
        result = select_mixed_budget_masks(scores, refs, [torch.ones(4)] * 2, .5)
        self.assertEqual([m.tolist() for m in result], [[1., 0., 0., 0.], [1., 1., 1., 0.]])
        zeros = [torch.zeros(4)] * 2
        self.assertEqual(sum(int(m.sum()) for m in select_mixed_budget_masks(scores, zeros, [torch.ones(4)] * 2, .5)), 0)


class GateAndOverlapTests(unittest.TestCase):
    make = staticmethod(test_global_conflict_budget.GlobalBudgetSelectionTests._make_attention)

    def test_s_p_gates_do_not_remove_protection_or_plasticity(self):
        for s_on in (False, True):
            for p_on in (False, True):
                module = self.make(dual_mask_s_conflict_enabled=s_on, dual_mask_p_conflict_enabled=p_on)
                module.general_mask.zero_()
                module.general_mask[:, :2] = 1
                module.effective_protect_strength = .4
                module.set_global_conflict_masks(torch.ones_like(module.qkv.weight), torch.ones_like(module.qkv.weight))
                delta = torch.ones_like(module.qkv.weight)
                for isolated, enabled in ((False, s_on), (True, p_on)):
                    base = delta * ((1 - module.general_mask) if isolated else (1 - .4 * module.general_mask))
                    expected = base * (.5 if enabled else 1.)
                    self.assertTrue(torch.equal(module._safe_delta(delta, isolated, .25, .5), expected))
                    self.assertTrue(torch.equal(module._compose_merge_delta(delta, isolated, .25, .5), expected))
                module.cur_task = 0
                module.dual_mask_task0_gate_mode = "unmasked"
                self.assertTrue(torch.equal(module._safe_delta(delta, False), delta))

    def test_reg_stays_independent_when_gates_are_off(self):
        module = self.make(dual_mask_s_conflict_enabled=False, dual_mask_p_conflict_enabled=False)
        module.before_task(1)
        with torch.no_grad():
            module.S_lora[1].A.weight.copy_(torch.arange(8.).reshape(2, 4) / 8)
            module.S_lora[1].B.weight.fill_(1)
        module.w0_importance.copy_(torch.linspace(.1, 1, 48).reshape(12, 4))
        module.dual_mask_conflict_reg_enabled = True
        on = module._joint_conflict_regularization(module.S_lora[1], False)
        module.dual_mask_conflict_reg_enabled = False
        off = module._joint_conflict_regularization(module.S_lora[1], False)
        self.assertGreater(float(on.detach()), float(off.detach()))
        self.assertGreater(float(off.detach()), 0.)

    def test_recorder_is_read_only_and_handles_zero_updates(self):
        raw = torch.arange(48.).reshape(12, 4)
        before = raw.clone()
        state = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            recorder = UpdateOverlapRecorder(directory, 0)
            recorder.record(0, "S", raw, raw * .5)
            recorder.record(1, "S", raw, raw * .5)
            recorder.record(1, "P", raw * 0, raw * 0)
            records = [json.loads(line) for line in recorder.output.read_text().splitlines()]
            pair = next(r for r in records if r["kind"] == "pair")
            self.assertAlmostEqual(pair["cosine"], 1., places=6)
            for value in pair["spectral_alignment_qkv"]:
                self.assertAlmostEqual(value, 1., places=6)
            self.assertIsNone(cosine(raw, raw * 0))
            self.assertEqual(describe(raw * 0)["effective_rank"], [None] * 3)
            cache = Path(recorder.cache.name)
            recorder.close()
            self.assertFalse(cache.exists())
            self.assertTrue(recorder.output.exists())
            self.assertEqual(python_state, random.getstate())
            self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(raw, before))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_direction_and_spectral_alignment_are_different_metrics(self):
        first = torch.zeros(12, 4)
        first[:, 0] = 1
        second = torch.zeros_like(first)
        second[:, 1] = 1
        self.assertAlmostEqual(cosine(first, -first), -1., places=6)
        self.assertEqual(cosine(first, second), 0.)
        a, b, opposite = describe(first), describe(second), describe(-first)
        self.assertEqual(cosine(a["grams"], b["grams"]), 0.)
        self.assertAlmostEqual(cosine(a["grams"], opposite["grams"]), 1., places=6)
        self.assertEqual(a["effective_rank"], [1.] * 3)

    def test_diagnostic_does_not_change_merged_weights_logits_or_rng(self):
        module = self.make()
        module.before_task(1)
        with torch.no_grad():
            module.S_lora[1].B.weight.fill_(.1)
            module.P_lora[1].B.weight.fill_(.2)
        plain, diagnostic = copy.deepcopy(module), copy.deepcopy(module)
        with tempfile.TemporaryDirectory() as directory:
            diagnostic.dual_mask_update_overlap = True
            diagnostic.dual_mask_applied_budget_log = True
            diagnostic._update_overlap_dir = directory
            state = torch.get_rng_state().clone()
            plain.after_task(1)
            after_plain = torch.get_rng_state().clone()
            torch.set_rng_state(state)
            diagnostic.after_task(1)
            self.assertTrue(torch.equal(after_plain, torch.get_rng_state()))
            for key, value in plain.state_dict().items():
                self.assertTrue(torch.equal(value, diagnostic.state_dict()[key]), key)
            x = torch.arange(12.).reshape(1, 3, 4)
            plain.eval()
            diagnostic.eval()
            self.assertTrue(torch.equal(plain(x, 1), diagnostic(x, 1)))

    def test_disabled_p_reports_zero_actual_suppression(self):
        module = self.make(dual_mask_p_conflict_enabled=False)
        raw = torch.arange(48.).reshape(12, 4)
        safe = module._compose_merge_delta(raw, True, .25, .5)
        stats = module._private_merge_diagnostic(raw, safe, .25, .5)
        self.assertEqual(stats["selected"], 0)
        self.assertEqual(stats["removed_norm"], 0)
        self.assertEqual(stats["merge_error"], 0)

    def test_all_gates_off_logs_no_conflict_gate_suppression(self):
        module = self.make(dual_mask_s_conflict_enabled=False, dual_mask_p_conflict_enabled=False)
        module.before_task(1)
        with torch.no_grad():
            module.S_lora[1].B.weight.fill_(1)
            module.P_lora[1].B.weight.fill_(1)
        module.after_task(1)
        self.assertAlmostEqual(float(module.last_conflict_gate_suppression), 0., places=6)
        self.assertAlmostEqual(float(module.last_private_conflict_gate_suppression), 0., places=6)

    def test_cpu_training_refresh_and_merge_smoke(self):
        # Execute the actual learner orchestration methods without importing the
        # pretrained ViT/data dependencies. This is a toy CPU smoke, not ImageNet.
        source = ast.parse((Path(__file__).resolve().parents[1] / "methods/dlora.py").read_text())
        learner_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Learner")
        names = {"_global_conflict_enabled", "_refresh_global_conflict_masks", "_iter_lora_modules"}
        methods = [node for node in learner_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"torch": torch, "logging": logging,
                     "Attention_LoRA": test_global_conflict_budget.Attention_LoRA,
                     "select_global_budget_masks": select_global_budget_masks,
                     "select_projection_budget_masks": select_projection_budget_masks,
                     "select_mixed_budget_masks": select_mixed_budget_masks}
        exec(compile(ast.Module(body=methods, type_ignores=[]), "methods/dlora.py", "exec"), namespace)
        Harness = type("Harness", (), {name: namespace[name] for name in names})
        for rho in (0., .25, .5, .75, 1.):
            harness = Harness()
            harness.args = {"dual_mask_conflict_granularity": "mixed", "dual_mask_conflict_local_fraction": rho}
            harness._cur_task = 1
            harness._network = torch.nn.ModuleList([self.make(granularity="mixed") for _ in range(2)])
            for module in harness._network:
                module.before_task(1)
                module.set_task_and_stage(1, 0)
            optimizer = torch.optim.SGD([p for p in harness._network.parameters() if p.requires_grad], lr=.01)
            x = torch.arange(12.).reshape(1, 3, 4) / 12
            for _ in range(2):
                optimizer.zero_grad()
                harness._refresh_global_conflict_masks()
                loss = sum(module(x, 1).square().mean() for module in harness._network)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()
            harness._refresh_global_conflict_masks(log_summary=True)
            for module in harness._network:
                module.eval()
                before = module(x, 1).detach()
                module.after_task(1)
                self.assertTrue(torch.allclose(before, module(x, 1), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
