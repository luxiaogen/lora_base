import importlib
import importlib.util
import csv
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self):
        self.labels = np.repeat(np.arange(4), 6)
        self.images = np.asarray([f'class{label}/image{i}' for i, label in enumerate(self.labels)])
        self.accessed = []
        self.features = torch.eye(4)[torch.from_numpy(self.labels)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        self.accessed.append(index)
        x = self.features[index]
        return index, x, int(self.labels[index])


class IdentityEncoder(torch.nn.Module):
    def extract_vector(self, inputs):
        return inputs


class FrozenReadoutRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.run_frozen_readout'),
                             'Standalone readout runner is not implemented yet')
        self.runner = importlib.import_module('scripts.run_frozen_readout')

    def test_frozen_encoder_bypasses_nonzero_lora_and_preserves_original_qkv(self):
        from models.attention import Attention_LoRA, FrozenA_TrainableB
        network = torch.nn.Module()
        network.attention = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=1)
        network.attention.S_lora[0] = FrozenA_TrainableB(4, 12, 2, torch.ones(2, 4), torch.ones(12, 2))
        original = network.attention.qkv.weight.detach().clone()
        self.assertTrue(callable(getattr(self.runner, 'freeze_original_encoder', None)))
        self.runner.freeze_original_encoder(network, dict(use_slora=True, use_plora=True), 'cpu')
        self.assertEqual(network.numtask, 1)
        self.assertFalse(network.training)
        self.assertTrue(all(not p.requires_grad for p in network.parameters()))
        torch.testing.assert_close(network.attention.qkv.weight, original)
        torch.testing.assert_close(network.attention.pretrained_weight, original)
        torch.testing.assert_close(network.attention._contrib_from_units(torch.ones(2, 3, 4), 0),
                                   torch.zeros(2, 3, 12))

    def test_stream_current_tasks_once_and_never_write_features_or_weights(self):
        train, test = FeatureDataset(), FeatureDataset()
        args = dict(embd_dim=4, total_sessions=2, init_cls=2, increment=2, batch_size=3,
                    num_workers=0, seed=1993, two_expert_calibration_holdout_mod=3,
                    dual_mask_competence_holdout_mod=2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.runner.run_readouts(IdentityEncoder(), train, test, args, 'cpu', out,
                                                  Path(tmp) / 'missing_reference', 2)
            self.assertEqual(result['ridge_average'], 100.)
            self.assertEqual(result['ncm_average'], 100.)
            self.assertEqual(result['tasks'], 2)
            self.assertEqual(sorted(train.accessed), list(range(12)) + [13, 14, 16, 17, 19, 20, 22, 23])
            self.assertEqual(len(set(train.accessed)), len(train.accessed))
            self.assertEqual(sorted(test.accessed), list(range(24)))
            self.assertEqual(len(set(test.accessed)), len(test.accessed))
            self.assertFalse(any(p.suffix in ('.pt', '.npz', '.npy') for p in out.rglob('*')))
            seal = json.loads((out / 'task0_alpha.json').read_text())
            self.assertEqual(seal['source'], 'task0_train_holdout_only')
            first = json.loads((out / 'task_00.json').read_text())
            self.assertEqual(first['alpha'], seal['selected_alpha'])
            self.assertEqual(first['reference_status'], 'cache_missing')
            self.assertIsNone(result['ridge_complement_oracle_average'])
            self.assertEqual(result['comparison_status'], 'paired_incomplete')

    def test_ncm_reproduction_mismatch_is_reported_without_claiming_a_matched_oracle(self):
        args = dict(embd_dim=4, total_sessions=2, init_cls=2, increment=2, batch_size=3,
                    num_workers=0, seed=1993, two_expert_calibration_holdout_mod=3,
                    dual_mask_competence_holdout_mod=2, _reference_protocol_compatible=True)
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / 'reference'
            reference.mkdir()
            with (reference / 'task_00_test_report_only.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['index', 'target', 'base_prediction', 'anchor_prediction'])
                writer.writerows((i, i // 6, i // 6, 1 - i // 6) for i in range(12))
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.runner.run_readouts(IdentityEncoder(), FeatureDataset(), FeatureDataset(),
                    args, 'cpu', Path(tmp) / 'out', reference, 1)
            self.assertEqual(result['ncm_last'], 100.)
            self.assertEqual(result['last_report']['reference_ncm_agreement_percent'], 0.)
            self.assertEqual(result['last_report']['reference_status'], 'ncm_reproduction_mismatch')
            self.assertIsNone(result['ridge_complement_oracle_average'])

    def test_alpha_is_selected_before_refitting_all_allowed_training_samples(self):
        train, test = FeatureDataset(), FeatureDataset()
        args = dict(embd_dim=4, total_sessions=2, init_cls=2, increment=2, batch_size=3,
                    num_workers=0, seed=1993, two_expert_calibration_holdout_mod=3,
                    dual_mask_competence_holdout_mod=2)
        real_select = self.runner.select_alpha
        selection_samples = []

        def select_before_test(stats, *values):
            selection_samples.append(stats.samples)
            self.assertEqual(test.accessed, [])
            return real_select(stats, *values)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            with patch.object(self.runner, 'select_alpha', side_effect=select_before_test):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = self.runner.run_readouts(IdentityEncoder(), train, test, args,
                        'cpu', out, Path(tmp) / 'missing_reference', 2)
            self.assertEqual(selection_samples, [6])
            self.assertEqual(json.loads((out / 'task_00.json').read_text())['cumulative_fit'], 12)
            self.assertEqual(result['last_report']['cumulative_fit'], 20)
            self.assertEqual(len(train.accessed), 20)
            self.assertTrue(json.loads((out / 'task0_alpha.json').read_text())['refit_with_holdout'])

    def test_ncm_matches_original_full_prototypes_not_competence_fit_subset(self):
        from utils.dual_mask_metrics import build_prototypes
        train, test = FeatureDataset(), FeatureDataset()
        train.features[:6:2] = torch.tensor([0., 10., 0., 0.])
        test.features[:6] = torch.tensor([1., 2., 0., 0.])
        prototypes, _ = build_prototypes(train.features[:12], torch.tensor(train.labels[:12]))
        expected = (torch.nn.functional.normalize(test.features[:12], dim=1) @ prototypes.T).argmax(1)
        args = dict(embd_dim=4, total_sessions=2, init_cls=2, increment=2, batch_size=3,
                    num_workers=0, seed=1993, two_expert_calibration_holdout_mod=3,
                    dual_mask_competence_holdout_mod=2, _reference_protocol_compatible=True)
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / 'reference'
            reference.mkdir()
            with (reference / 'task_00_test_report_only.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['index', 'target', 'base_prediction', 'anchor_prediction'])
                writer.writerows((i, i // 6, 1, int(expected[i])) for i in range(12))
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.runner.run_readouts(IdentityEncoder(), train, test, args,
                    'cpu', Path(tmp) / 'out', reference, 1)
            self.assertEqual(result['paired_reference_tasks'], 1)
            self.assertEqual(result['last_report']['reference_ncm_agreement_percent'], 100.)
            self.assertEqual(result['last_report']['base_vs_ncm']['total']['rescued'], 6)

    def test_reproduced_ncm_pairs_with_cached_base_for_complement_counts(self):
        args = dict(embd_dim=4, total_sessions=2, init_cls=2, increment=2, batch_size=3,
                    num_workers=0, seed=1993, two_expert_calibration_holdout_mod=3,
                    dual_mask_competence_holdout_mod=2, _reference_protocol_compatible=True)
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / 'reference'
            reference.mkdir()
            with (reference / 'task_00_test_report_only.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['index', 'target', 'base_prediction', 'anchor_prediction'])
                writer.writerows((i, i // 6, 0, i // 6) for i in range(12))
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.runner.run_readouts(IdentityEncoder(), FeatureDataset(), FeatureDataset(),
                    args, 'cpu', Path(tmp) / 'out', reference, 1)
            self.assertEqual(result['base_average'], 50.)
            self.assertEqual(result['ridge_complement_oracle_average'], 100.)
            self.assertEqual(result['last_report']['base_vs_ridge']['total']['rescued'], 6)
            self.assertEqual(result['last_report']['base_vs_ridge']['total']['harmed'], 0)
            self.assertEqual(result['comparison_status'], 'paired_complete')

    def test_reference_config_preserves_local_paths_and_machine_specific_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / 'diagnostics'
            ref.mkdir()
            (ref.parent / 'run.json').write_text(json.dumps(dict(effective_config=dict(
                seed=[1993], data_path='/other_machine', device='9',
                two_expert_calibration_holdout_mod=7, embd_dim=768, shuffle=True,
                dataset='ImageNet_R', disable_fused_sdpa=True,
                init_cls=20, increment=20, total_sessions=10))))
            config, source = self.runner.configuration(dict(data_path='/local_machine', device='0'), ref, '5090')
            self.assertEqual(config['data_path'], '/local_machine')
            self.assertEqual(config['device'], '0')
            self.assertEqual(config['two_expert_calibration_holdout_mod'], 7)
            self.assertEqual(config['seed'], 1993)
            self.assertEqual(source, str(ref.parent / 'run.json'))

    def test_reference_with_another_seed_cannot_be_paired_even_if_labels_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / 'diagnostics'
            ref.mkdir()
            (ref.parent / 'run.json').write_text(json.dumps(dict(effective_config=dict(
                seed=[1996], dataset='ImageNet_R', disable_fused_sdpa=True,
                shuffle=True, init_cls=20, increment=20, total_sessions=10))))
            config, _ = self.runner.configuration(dict(data_path='/local', device='0'), ref, '3090')
            self.assertEqual(config['seed'], 1993)
            self.assertFalse(config.get('_reference_protocol_compatible', True))
            self.assertEqual(config['two_expert_calibration_holdout_mod'], 0)

    def test_wrappers_dry_run_do_not_require_cuda_dataset_or_model_loading(self):
        root = Path(__file__).resolve().parents[1]
        for machine in ('3090', '5090'):
            result = subprocess.run(['bash', f'scripts/10_01_imgr10_frozen_readout_{machine}.sh', '--dry-run'],
                                    cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['machine'], machine)
            self.assertEqual(plan['seed'], 1993)
            self.assertEqual(plan['tasks'], 10)
            self.assertFalse(plan['train_dualmask'])
            self.assertFalse(plan['save_weights'])


if __name__ == '__main__':
    unittest.main()
