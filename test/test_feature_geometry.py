import tempfile
import json
import subprocess
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.feature_geometry import analyze_geometry, remap_labels, extract_features, top_confused_pairs


class GeometryTests(unittest.TestCase):
    def test_perfect_clusters_and_margin(self):
        x = np.repeat(np.eye(3), 2, axis=0)
        y = np.repeat(np.arange(3), 2)
        result = analyze_geometry(x, y, x, y, np.eye(3))
        np.testing.assert_array_equal(result['confusion'], np.eye(3) * 2)
        np.testing.assert_allclose(result['margin'], 1)
        np.testing.assert_allclose(result['dispersion'], 0)
        np.testing.assert_allclose(result['nearest_center_distance'], np.sqrt(2))
        self.assertEqual(result['accuracy'], 100)

    def test_head_error_distinguished_from_feature_overlap(self):
        x = np.eye(3)
        y = np.arange(3)
        result = analyze_geometry(x, y, x, y, x[[1, 0, 2]])
        self.assertAlmostEqual(result['accuracy'], 100 / 3)
        np.testing.assert_array_equal(result['nearest_center_prediction'], y)
        np.testing.assert_allclose(result['margin'], [-1, -1, 1])
        self.assertEqual(top_confused_pairs(result['confusion'], 2), [(0, 1)])

    def test_remapping_uses_saved_order(self):
        np.testing.assert_array_equal(remap_labels([2, 0, 1], [1, 2, 0]), [1, 2, 0])
        np.testing.assert_array_equal(remap_labels([1, 2], [1, 2, 0]), [0, 1])

    def test_feature_collection_does_not_change_weights(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(3, 3)
                self.numtask = 2
            def extract_vector(self, x, task_id=None):
                return self.encoder(x)
        net = Net().eval()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        data = TensorDataset(torch.arange(5), torch.randn(5, 3), torch.arange(5) % 2)
        x, y, ids = extract_features(net, DataLoader(data, batch_size=2), 'cpu', 'test')
        self.assertEqual(x.shape, (5, 3))
        np.testing.assert_array_equal(ids, np.arange(5))
        for name, value in net.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_missing_training_class_is_not_fabricated_as_center(self):
        result = analyze_geometry(np.eye(3)[:2], np.arange(2), np.eye(3), np.arange(3), np.eye(3))
        self.assertTrue(np.isnan(result['centers'][2]).all())
        self.assertTrue(np.isnan(result['dispersion'][2]))
        self.assertEqual(result['train_counts'].tolist(), [1, 1, 0])

    def test_report_end_to_end_with_synthetic_data(self):
        from scripts.plot_feature_geometry import report
        rng = np.random.default_rng(7)
        labels = np.repeat(np.arange(4), 20)
        centers = rng.normal(size=(4, 8))
        train = centers[labels] + rng.normal(size=(80, 8)) * .4
        test = centers[labels] + rng.normal(size=(80, 8)) * .5
        heads = centers[[1, 0, 3, 2]]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            np.savez_compressed(cache / 'features.npz', train_features=train, train_labels=labels,
                                test_features=test, test_labels=labels, test_ids=np.arange(80), heads=heads)
            (cache / 'metadata.json').write_text(json.dumps(dict(
                increments=[2, 2], task=1, class_order=[2, 3, 0, 1],
                checkpoint_last_accuracy=0., checkpoint='SYNTHETIC_TEST_ONLY', checkpoint_sha256='fixture')))
            summary = report(cache, cache / 'figures', pairs=2, per_class=15)
            for name in ('01_confusion', '02_tsne_pairs', '03_center_head_alignment'):
                for extension in ('png', 'pdf'):
                    self.assertGreater((cache / 'figures' / f'{name}.{extension}').stat().st_size, 1000)
            self.assertEqual(summary['test_samples'], 80)
            self.assertEqual(len((cache / 'figures' / 'classes.csv').read_text().splitlines()), 5)

    def test_extractor_is_inference_only_and_script_checkpoint_is_exact(self):
        source = Path('scripts/extract_feature_geometry.py').read_text()
        self.assertNotIn('optimizer', source.split('def main():')[1])
        self.assertNotIn('.backward(', source)
        self.assertIn('iIMAGENET_R.test_trsf', source)
        self.assertNotIn('DataManager(', source)
        self.assertNotIn('.download_data(', source)
        self.assertIn("network.load_state_dict(payload['model_state_dict'], strict=True)", source)
        subprocess.run(['bash', '-n', 'scripts/9_28_imgr10_feature_geometry_3090.sh'], check=True)
        script = Path('scripts/9_28_imgr10_feature_geometry_3090.sh').read_text()
        self.assertIn('20260928_133709_314001/task_09.pt', script)

    def test_cached_logits_equal_real_network_all_seen_interface_formula(self):
        x = np.array([[1., 2, -1], [-1, 0, 3]])
        heads = np.array([[2., 0, -1], [0, 1, 1], [-1, 0, 3]])
        logits = torch.nn.functional.normalize(torch.tensor(x), dim=1) @ torch.nn.functional.normalize(torch.tensor(heads), dim=1).T
        result = analyze_geometry(np.eye(3), np.arange(3), x, np.array([0, 2]), heads)
        np.testing.assert_array_equal(result['prediction'], logits.argmax(1).numpy())

    def test_existing_imagefolder_order_and_missing_split_do_not_mutate_data(self):
        from PIL import Image
        from scripts.extract_feature_geometry import load_split
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('a', 'b', 'c'):
                directory = root / 'train' / name
                directory.mkdir(parents=True)
                Image.new('RGB', (256, 256), color='red').save(directory / 'image.png')
            before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
            dataset = load_split(root, 'train', [2, 0, 1], 2)
            self.assertEqual(dataset.labels.tolist(), [0, 1])
            self.assertEqual([Path(p).parent.name for p in dataset.images], ['c', 'a'])
            self.assertEqual(dataset[0][1].shape, (3, 224, 224))
            with self.assertRaises(FileNotFoundError):
                load_split(root, 'test', [2, 0, 1], 2)
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob('*')))


if __name__ == '__main__':
    unittest.main()
