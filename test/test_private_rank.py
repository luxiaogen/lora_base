import unittest

import torch

from models.attention import Attention_LoRA


class PrivateRankTests(unittest.TestCase):
    def make_module(self, **overrides):
        args = dict(use_slora=True, use_plora=True, lora_A_init="kaiming",
                    dual_mask_competence_adaptive=True, dual_mask_plasticity_adaptive=True,
                    dual_mask_protect_strength_mode="competence", dual_mask_task0_gate_mode="unmasked",
                    dual_mask_conflict_energy_adaptive=True, dual_mask_conflict_energy_ratio_floor=True,
                    dual_mask_conflict_ratio=0.1, dual_mask_conflict_strength=0.5)
        args.update(overrides)
        module = Attention_LoRA(dim=4, num_heads=1, r=4, n_tasks=3)
        module._init_params(args)
        module.set_pretrained_competence(0.8, 0.25)
        return module

    def test_fixed_rank_changes_only_private_capacity(self):
        torch.manual_seed(7)
        adaptive = self.make_module()
        torch.manual_seed(7)
        fixed = self.make_module(dual_mask_private_rank=3)
        for task in (1, 2):
            for module in (adaptive, fixed):
                module.set_pretrained_competence(0.8, 0.25)
                module.before_task(task)
                module.set_task_and_stage(task, 0)
            self.assertEqual(fixed.current_private_rank, 3)
            self.assertEqual(adaptive.current_private_rank, 2)
            self.assertEqual(fixed.S_lora[task].r, adaptive.S_lora[task].r)
            self.assertEqual(fixed.P_lora[task].r, 3)
            self.assertEqual(fixed.P_lora[task].B.weight.numel(), 36)
            self.assertTrue(fixed.P_lora[task].B.weight.requires_grad)
            for field in ('pretrained_control_competence', 'effective_energy_coverage', 'effective_protect_strength'):
                self.assertEqual(getattr(fixed, field), getattr(adaptive, field))
            for field in ('general_mask', 'isolated_mask', 'w0_importance'):
                self.assertTrue(torch.equal(getattr(fixed, field), getattr(adaptive, field)))
            self.assertEqual(fixed._conflict_parameters(), adaptive._conflict_parameters())

    def test_task0_fixed_override_preserves_initialization_and_rng(self):
        states = []
        for overrides in ({}, {'dual_mask_private_rank': 0}, {'dual_mask_private_rank': 3}):
            torch.manual_seed(1993)
            modules = [self.make_module(**overrides) for _ in range(2)]
            for module in modules:
                module.before_task(0)
            states.append(([m.state_dict() for m in modules], torch.get_rng_state()))
        for candidate, rng in states[1:]:
            self.assertTrue(torch.equal(rng, states[0][1]))
            for reference, actual in zip(states[0][0], candidate):
                self.assertEqual(reference.keys(), actual.keys())
                for key in reference:
                    self.assertTrue(torch.equal(reference[key], actual[key]), key)

    def test_zero_override_preserves_later_task_initialization(self):
        states = []
        for overrides in ({}, {'dual_mask_private_rank': 0}):
            torch.manual_seed(1993)
            module = self.make_module(**overrides)
            module.before_task(1)
            states.append((module.state_dict(), torch.get_rng_state()))
        self.assertTrue(torch.equal(states[0][1], states[1][1]))
        for key, value in states[0][0].items():
            self.assertTrue(torch.equal(value, states[1][0][key]), key)

    def test_fixed_rank_without_competence_controller(self):
        module = self.make_module(dual_mask_competence_adaptive=False, dual_mask_private_rank=3)
        module.before_task(1)
        self.assertEqual(module.P_lora[1].r, 3)
        self.assertEqual(module.S_lora[1].r, 4)

    def test_fixed_rank_training_and_merge_smoke(self):
        torch.manual_seed(19)
        module = self.make_module(dual_mask_private_rank=3)
        module.before_task(1)
        module.set_task_and_stage(1, 0)
        x = torch.randn(2, 3, 4)
        target = torch.randn_like(x)
        optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=0.1)
        for _ in range(3):
            optimizer.zero_grad()
            loss = (module(x, task=1) - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        self.assertGreater(module.P_lora[1].B.weight.norm().item(), 0)
        module.eval()
        with torch.no_grad():
            before = module(x, task=1)
            module.after_task(1)
            after = module(x, task=1)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
        self.assertIsNone(module.P_lora[1])


if __name__ == '__main__':
    unittest.main()
