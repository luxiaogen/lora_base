"""可在独立目录运行：python -m unittest discover -s tests -v。"""
import unittest
import torch

from models.attention import Attention_LoRA, _exact_top_ratio_mask, _energy_coverage_mask
from utils.depth_permissions import calibrate_center, depth_value, install_depth_permissions
from utils.dual_mask_metrics import split_prototype_competence


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1993)
        torch.set_num_threads(1)

    def test_exact_top10_full_qkv_including_ties(self):
        score = torch.zeros(48, 16)
        selected = _exact_top_ratio_mask(score, .1)
        self.assertEqual(int(selected.sum()), int(score.numel() * .1))
        self.assertEqual(int(_exact_top_ratio_mask(score, 0).sum()), 0)
        self.assertEqual(int(_exact_top_ratio_mask(score, 1).sum()), score.numel())

    def test_importance_mass_not_coordinate_density(self):
        score = torch.tensor([9., .5, .5])
        self.assertEqual(_energy_coverage_mask(score, .9).tolist(), [1., 0., 0.])

    def test_depth_calibration_repeatable_and_monotone(self):
        scores = [torch.rand(18, 6) for _ in range(12)]
        rng = torch.get_rng_state().clone()
        center = calibrate_center(scores)
        self.assertEqual(center, calibrate_center(scores))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        values = [depth_value(center, .05, layer, 12) for layer in range(12)]
        self.assertAlmostEqual(values[0] - values[-1], .1)
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_task0_unmasked_anchor_and_task1_permissions(self):
        layer = Attention_LoRA(16, num_heads=4, qkv_bias=True, r=8, n_tasks=2)
        layer._init_params({})
        layer.before_task(0)
        layer.set_task_and_stage(0, 0)
        delta = torch.randn_like(layer.qkv.weight)
        self.assertTrue(torch.equal(layer._safe_delta(delta, False), delta))
        self.assertTrue(layer.S_lora[0].A_weight.requires_grad)
        layer.after_task(0)
        layer.set_pretrained_old_overlap_risk(.2)
        layer.before_task(1)
        layer.set_task_and_stage(1, 0)
        self.assertEqual(layer.P_lora[1].r, 8)
        self.assertFalse(layer.S_lora[1].A_weight.requires_grad)
        self.assertFalse(layer.P_lora[1].A_weight.requires_grad)
        self.assertAlmostEqual(layer._conflict_parameters()[1], .6)
        for isolated in (False, True):
            safe, gate, mask = layer._safe_delta(delta, isolated, return_details=True)
            self.assertEqual(int(mask.sum()), int(delta.numel() * .1))
            self.assertLessEqual(float(safe.norm()), float(delta.norm()))
            if isolated:
                self.assertEqual(int(torch.count_nonzero(safe * layer.general_mask)), 0)

    def test_zero_B_has_gradient_and_merge_happens_once(self):
        layer = Attention_LoRA(16, num_heads=4, qkv_bias=True, r=8, n_tasks=2)
        layer._init_params({})
        layer.before_task(1)
        layer.set_task_and_stage(1, 0)
        x = torch.randn(2, 5, 16)
        layer(x, 1).square().mean().backward()
        self.assertGreater(float(layer.S_lora[1].B_weight.grad.norm()), 0.)
        self.assertGreater(float(layer.P_lora[1].B_weight.grad.norm()), 0.)
        with torch.no_grad():
            layer.S_lora[1].B_weight.normal_(std=.01)
            layer.P_lora[1].B_weight.normal_(std=.01)
        layer.eval()
        before = layer(x, 1).detach()
        anchor = layer.pretrained_weight.clone()
        layer.after_task(1)
        torch.testing.assert_close(before, layer(x, 1), atol=2e-6, rtol=2e-6)
        weight = layer.qkv.weight.clone()
        layer.after_task(1)
        self.assertTrue(torch.equal(weight, layer.qkv.weight))
        self.assertTrue(torch.equal(anchor, layer.pretrained_weight))

    def test_NCM_train_holdout(self):
        labels = torch.arange(2).repeat_interleave(10)
        features = torch.eye(2)[labels]
        indices = torch.arange(20)
        accuracy, prototypes, classes = split_prototype_competence(features, labels, indices)
        self.assertEqual(accuracy, 1.)
        self.assertEqual(classes.tolist(), [0, 1])
        torch.testing.assert_close(prototypes, torch.eye(2))

    def test_incremental_NCM_does_not_change_permissions_or_rank(self):
        from methods.dlora import Learner
        from torch.utils.data import DataLoader, TensorDataset

        class AnchorNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = Attention_LoRA(64, num_heads=4, r=64, n_tasks=2)

            def extract_vector(self, inputs):
                return inputs

        learner = object.__new__(Learner)
        learner._network, learner._device = AnchorNetwork(), torch.device('cpu')
        learner._cur_task, learner._known_classes = 1, 2
        learner._w0_class_means = {0: torch.eye(4)[0], 1: torch.eye(4)[1]}
        layer = learner._network.attention
        layer._init_params({})
        layer.before_task(1)
        delta = torch.arange(1., 12289.).reshape(192, 64)
        before = layer._safe_delta(delta, False).clone()
        labels = torch.tensor([2, 3]).repeat_interleave(10)
        loader = DataLoader(TensorDataset(torch.arange(20), torch.eye(4)[labels], labels), batch_size=10)
        learner._prepare_w0_prototypes(loader)
        self.assertTrue(torch.equal(before, layer._safe_delta(delta, False)))
        self.assertEqual(layer.P_lora[1].r, 64)
        self.assertEqual(layer.effective_energy_coverage, .9)

    def test_calibrated_depth_coverage_keeps_fixed_alpha_and_rng(self):
        layers = [Attention_LoRA(64, num_heads=4, r=64, n_tasks=2) for _ in range(12)]
        for layer in layers:
            layer._init_params({})
            layer.before_task(1)
        delta = torch.arange(1., 12289.).reshape(192, 64)
        center = install_depth_permissions(layers, 1, {})
        original_masks = [layer.general_mask.clone() for layer in layers]
        rng = torch.get_rng_state().clone()
        install_depth_permissions(layers, 1, {}, center)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for layer, mask in zip(layers, original_masks):
            self.assertEqual(layer.effective_protect_strength, .5)
            self.assertTrue(torch.equal(layer.general_mask, mask))
        self.assertAlmostEqual(layers[0].effective_energy_coverage - layers[-1].effective_energy_coverage, .1)


if __name__ == '__main__':
    unittest.main()
