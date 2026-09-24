import random
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.losses import AngularPenaltySMLoss
from utils.data_manager import DataManager
from utils.task0_validation import evaluate_task0_holdout
from utils.task0_repro import tensor_hash, model_fingerprint, log_record


class Task0ReproTests(unittest.TestCase):
    def test_task0_holdout_split_is_deterministic_balanced_and_disjoint(self):
        manager = DataManager.__new__(DataManager)
        manager._train_data = np.arange(20)
        manager._train_targets = np.repeat([0, 1], 10)
        manager._train_trsf = [lambda image: image]
        manager._test_trsf = [lambda image: image]
        manager._common_trsf = []
        manager.use_path = False

        train_a, holdout_a = manager.get_dataset_with_deterministic_holdout(
            [0, 1], source='train', holdout_mod=5,
        )
        train_b, holdout_b = manager.get_dataset_with_deterministic_holdout(
            [0, 1], source='train', holdout_mod=5,
        )

        self.assertEqual(len(train_a), 16)
        self.assertEqual(len(holdout_a), 4)
        self.assertEqual(np.bincount(train_a.labels).tolist(), [8, 8])
        self.assertEqual(np.bincount(holdout_a.labels).tolist(), [2, 2])
        self.assertFalse(set(train_a.images.tolist()) & set(holdout_a.images.tolist()))
        np.testing.assert_array_equal(train_a.images, train_b.images)
        np.testing.assert_array_equal(holdout_a.images, holdout_b.images)

    def test_task0_holdout_evaluation_restores_rng_and_training_mode(self):
        class TinyDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                inputs = torch.tensor([float(index), 1.0])
                return index, inputs, index % 2

        class TinyNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, inputs):
                return {'logits': torch.nn.functional.normalize(self.linear(inputs), dim=1)}

        network = TinyNetwork()
        network.train()
        loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
        loss = AngularPenaltySMLoss(loss_type='cosface', s=20.0, m=0.1)
        torch_state = torch.get_rng_state().clone()
        np_state = np.random.get_state()
        py_state = random.getstate()

        metrics = evaluate_task0_holdout(
            network,
            loader,
            loss,
            device=torch.device('cpu'),
            cuda_devices=[],
            known_classes=0,
        )

        self.assertTrue(network.training)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertEqual(py_state, random.getstate())
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)

    def test_hash_includes_names_shapes_and_values(self):
        x = torch.arange(6).reshape(2, 3)
        self.assertEqual(tensor_hash([('x', x)]), tensor_hash([('x', x.clone())]))
        self.assertNotEqual(tensor_hash([('x', x)]), tensor_hash([('x', x + 1)]))
        self.assertNotEqual(tensor_hash([('x', x)]), tensor_hash([('y', x)]))
        self.assertNotEqual(tensor_hash([('x', x)]), tensor_hash([('x', x.reshape(3, 2))]))
        tensor_hash([('scalar', torch.tensor(1)), ('bf16', x.to(torch.bfloat16))])

    def test_observation_preserves_rng_weights_and_mode(self):
        model = torch.nn.Linear(3, 2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        torch_state = torch.get_rng_state().clone()
        np_state = np.random.get_state()
        py_state = random.getstate()
        with self.assertLogs(level='INFO') as captured:
            log_record('initial', **model_fingerprint(model))
        self.assertIn('Task0Repro', captured.output[0])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(py_state, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertTrue(model.training)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))

    def test_hash_changes_after_update(self):
        model = torch.nn.Linear(3, 2)
        before = model_fingerprint(model)
        with torch.no_grad():
            model.weight.add_(1)
        self.assertNotEqual(before['model_sha256'], model_fingerprint(model)['model_sha256'])

    def test_observation_does_not_change_training(self):
        def run(observe):
            torch.manual_seed(1993)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
            for _ in range(3):
                inputs = torch.randn(4, 3)
                targets = torch.randn(4, 2)
                if observe:
                    model_fingerprint(model)
                    tensor_hash([('inputs', inputs)])
                optimizer.zero_grad()
                (model(inputs) - targets).square().mean().backward()
                optimizer.step()
            return model_fingerprint(model)
        self.assertEqual(run(False), run(True))

    def test_sweep_is_three_identical_task0_runs(self):
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root / 'scripts/sweeps/imgr10_task0_repro_3090.json').read_text())
        self.assertEqual(spec['seeds'], [1993])
        self.assertEqual(len(spec['variants']), 3)
        self.assertTrue(all(v['overrides'] == {} for v in spec['variants']))
        common = spec['common_overrides']
        self.assertEqual(common['max_tasks'], 1)
        self.assertEqual(common['init_epoch'], 20)
        self.assertEqual(common['dual_mask_reg_weight'], 0.01)
        self.assertNotIn('data_path', common)
        script = (root / 'scripts/9_24_imgr10_task0_repro_3090.sh').read_text()
        self.assertEqual(script.count('python main.py'), 3)
        self.assertEqual(script.count('--set max_tasks=1'), 3)
        self.assertNotIn('data_path=', script)
        self.assertIn('cd "$(dirname "$0")/.."', script)

    def test_batch_sweep_changes_only_the_trace_length(self):
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root / 'scripts/sweeps/imgr10_task0_batch_repro_3090.json').read_text())
        self.assertEqual(spec['seeds'], [1993])
        self.assertEqual(len(spec['variants']), 2)
        self.assertTrue(all(v['overrides'] == {} for v in spec['variants']))
        common = spec['common_overrides']
        self.assertEqual(common['max_tasks'], 1)
        self.assertEqual(common['init_epoch'], 1)
        self.assertEqual(common['dual_mask_reg_weight'], 0.01)
        self.assertTrue(common['task0_repro_batch_diagnostic'])
        self.assertNotIn('data_path', common)
        script = (root / 'scripts/9_24_imgr10_task0_batch_repro_3090.sh').read_text()
        self.assertEqual(script.count('python main.py'), 2)
        self.assertEqual(script.count('--set init_epoch=1'), 2)
        self.assertNotIn('data_path=', script)
        self.assertIn('cd "$(dirname "$0")/.."', script)

    def test_math_sdpa_sweep_changes_only_backend(self):
        root = Path(__file__).resolve().parents[1]
        sweep_dir = root / 'scripts/sweeps'
        baseline = json.loads((sweep_dir / 'imgr10_task0_batch_repro_3090.json').read_text())
        candidate = json.loads((sweep_dir / 'imgr10_task0_math_sdpa_repro_3090.json').read_text())
        self.assertEqual(candidate['datasets'], baseline['datasets'])
        self.assertEqual(candidate['seeds'], baseline['seeds'])
        self.assertEqual([v['name'] for v in candidate['variants']],
                         [v['name'] for v in baseline['variants']])
        self.assertTrue(all(v['overrides'] == {} for v in candidate['variants']))
        common = dict(candidate['common_overrides'])
        self.assertTrue(common.pop('disable_fused_sdpa'))
        common.pop('wandb_group')
        baseline_common = dict(baseline['common_overrides'])
        baseline_common.pop('wandb_group')
        self.assertEqual(common, baseline_common)
        script = (root / 'scripts/9_24_imgr10_task0_math_sdpa_repro_3090.sh').read_text()
        self.assertEqual(script.count('python main.py'), 2)
        self.assertEqual(script.count('--set disable_fused_sdpa=true'), 2)
        self.assertNotIn('data_path=', script)
        self.assertIn('cd "$(dirname "$0")/.."', script)

    def test_full_math_sdpa_sweep_matches_full_task0_baseline(self):
        root = Path(__file__).resolve().parents[1]
        sweep_dir = root / 'scripts/sweeps'
        baseline = json.loads((sweep_dir / 'imgr10_task0_repro_3090.json').read_text())
        candidate = json.loads((sweep_dir / 'imgr10_task0_math_sdpa_full_repro_3090.json').read_text())
        self.assertEqual(candidate['datasets'], baseline['datasets'])
        self.assertEqual(candidate['seeds'], baseline['seeds'])
        self.assertEqual([v['name'] for v in candidate['variants']],
                         [v['name'] for v in baseline['variants']])
        self.assertTrue(all(v['overrides'] == {} for v in candidate['variants']))
        common = dict(candidate['common_overrides'])
        self.assertTrue(common.pop('disable_fused_sdpa'))
        common.pop('wandb_group')
        baseline_common = dict(baseline['common_overrides'])
        baseline_common.pop('wandb_group')
        self.assertEqual(common, baseline_common)
        script = (root / 'scripts/9_24_imgr10_task0_math_sdpa_full_repro_3090.sh').read_text()
        self.assertEqual(script.count('python main.py'), 3)
        self.assertEqual(script.count('--set init_epoch=20'), 3)
        self.assertEqual(script.count('--set disable_fused_sdpa=true'), 3)
        self.assertNotIn('task0_repro_batch_diagnostic', script)
        self.assertNotIn('data_path=', script)
        self.assertIn('cd "$(dirname "$0")/.."', script)

    def test_task0_holdout_tuning_and_full_baseline_scripts_are_separate(self):
        root = Path(__file__).resolve().parents[1]
        sweep_dir = root / 'scripts/sweeps'
        tuning = json.loads((sweep_dir / 'imgr10_task0_holdout_tuning_3090.json').read_text())
        self.assertEqual(tuning['seeds'], [1993])
        self.assertEqual([variant['name'] for variant in tuning['variants']],
                         ['base_e20_lr002', 'e30_lr002', 'e20_lr001'])
        common = tuning['common_overrides']
        self.assertEqual(common['max_tasks'], 1)
        self.assertTrue(common['task0_validation_enabled'])
        self.assertEqual(common['task0_validation_holdout_mod'], 5)
        self.assertTrue(common['disable_fused_sdpa'])
        self.assertNotIn('data_path', common)

        full = json.loads((sweep_dir / 'imgr10_math_sdpa_baseline_t10_seed1993_3090.json').read_text())
        self.assertEqual(full['seeds'], [1993])
        self.assertEqual(len(full['variants']), 1)
        full_common = full['common_overrides']
        self.assertEqual(full_common['max_tasks'], 10)
        self.assertFalse(full_common['task0_validation_enabled'])
        self.assertEqual(full_common['init_epoch'], 20)
        self.assertEqual(full_common['init_lr'], 0.02)
        self.assertEqual(full_common['ca_epochs'], 5)
        self.assertEqual(full_common['dual_mask_reg_weight'], 0.01)
        self.assertTrue(full_common['disable_fused_sdpa'])
        self.assertNotIn('data_path', full_common)


if __name__ == '__main__':
    unittest.main()
