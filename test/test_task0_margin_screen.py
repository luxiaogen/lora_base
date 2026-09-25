import json
from pathlib import Path
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from methods.dlora import Learner
from models.losses import AngularPenaltySMLoss, representation_steering_loss
from utils.task0_validation import evaluate_task0_holdout


class Task0MarginScreenTests(unittest.TestCase):
    def test_margin_override_applies_only_to_task0(self):
        learner = Learner.__new__(Learner)
        learner.args = {'task0_margin': 0.15}
        learner.margin = 0.1
        learner._cur_task = 0
        self.assertEqual(learner._training_margin(), 0.15)
        learner._cur_task = 1
        self.assertEqual(learner._training_margin(), 0.1)
        learner.args = {}
        learner._cur_task = 0
        self.assertEqual(learner._training_margin(), 0.1)

    def test_holdout_reports_raw_class_margin(self):
        class TinyDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                logits = ([0.8, 0.3], [0.2, 0.6])[index]
                return index, torch.tensor(logits), index

        class IdentityNetwork(torch.nn.Module):
            def forward(self, inputs):
                return {'logits': inputs}

        metrics = evaluate_task0_holdout(
            IdentityNetwork(),
            DataLoader(TinyDataset(), batch_size=2),
            AngularPenaltySMLoss(loss_type='cosface', s=20.0, m=0.1),
            device=torch.device('cpu'),
            cuda_devices=[],
            known_classes=0,
        )
        self.assertAlmostEqual(metrics['class_margin'], 0.45, places=6)
        self.assertEqual(metrics['accuracy'], 100.0)

    def test_holdout_reports_feature_geometry(self):
        class TinyDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return index, torch.tensor(([1., 0.], [-1., 0.])[index // 2]), index // 2

        class FeatureNetwork(torch.nn.Module):
            def forward(self, inputs):
                return {'logits': inputs, 'features': inputs}

        metrics = evaluate_task0_holdout(
            FeatureNetwork(),
            DataLoader(TinyDataset(), batch_size=2),
            AngularPenaltySMLoss(loss_type='cosface', s=20.0, m=0.1),
            device=torch.device('cpu'),
            cuda_devices=[],
            known_classes=0,
        )
        self.assertAlmostEqual(metrics['class_variance'], 0.0, places=6)
        self.assertAlmostEqual(metrics['min_centroid_margin'], 2.0, places=6)

    def test_representation_steering_loss_and_task0_warmup(self):
        labels = torch.tensor([0, 0, 1, 1])
        separated = torch.tensor([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]])
        collapsed = torch.tensor([[1., 0.]] * 4, requires_grad=True)
        self.assertAlmostEqual(float(representation_steering_loss(separated, labels)), 0.0)
        raw_loss = representation_steering_loss(collapsed, labels)
        self.assertAlmostEqual(float(raw_loss), 0.9, places=6)
        raw_loss.backward()
        self.assertTrue(torch.isfinite(collapsed.grad).all())

        learner = Learner.__new__(Learner)
        learner.args = {'dual_mask_reg_weight': 0.0, 'task0_rs_weight': 0.5}
        learner._cur_task = 0
        learner._iter_lora_modules = lambda: []
        output = {'features': collapsed.detach()}
        self.assertEqual(float(learner._extra_training_loss(output=output, targets=labels, epoch=0)), 0.0)
        self.assertAlmostEqual(float(learner._extra_training_loss(output=output, targets=labels, epoch=4)), 0.45)
        learner._cur_task = 1
        self.assertIsNone(learner._extra_training_loss(output=output, targets=labels, epoch=4))

    def test_four_arm_screen_changes_one_factor_at_a_time(self):
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root / 'scripts/sweeps/imgr10_task0_steering_screen_t3_3090.json').read_text())
        self.assertEqual(spec['seeds'], [1993])
        self.assertEqual([run['name'] for run in spec['variants']],
                         ['baseline', 'task0_margin015', 'anchor5', 'task0_rs'])
        common = spec['common_overrides']
        self.assertEqual(common['max_tasks'], 3)
        self.assertTrue(common['task0_validation_enabled'])
        self.assertTrue(common['disable_fused_sdpa'])
        self.assertEqual(common['task0_margin'], 0.1)
        self.assertEqual(common['task0_rs_weight'], 0.0)
        self.assertEqual(common['dual_mask_anchor_reg_weight'], 10.0)
        self.assertNotIn('data_path', common)
        self.assertEqual(spec['variants'][0]['overrides'], {})
        self.assertEqual(spec['variants'][1]['overrides'], {'task0_margin': 0.15})
        self.assertEqual(spec['variants'][2]['overrides'], {'dual_mask_anchor_reg_weight': 5.0})
        self.assertEqual(spec['variants'][3]['overrides'], {'task0_rs_weight': 0.5})

        script = (root / 'scripts/9_25_imgr10_task0_steering_screen_t3_3090.sh').read_text()
        self.assertEqual(script.count('python main.py'), 4)
        self.assertEqual(script.count('--set max_tasks=3'), 4)
        self.assertNotIn('data_path=', script)
        self.assertIn('cd "$(dirname "$0")/.."', script)


if __name__ == '__main__':
    unittest.main()
