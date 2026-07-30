import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from trainer import (  # noqa: E402
    _close_experiment_tracker,
    _init_experiment_tracker,
    _log_experiment_summary,
    _log_experiment_task,
)


class FakeWriter:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.scalars = []
        self.texts = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, name, value, global_step):
        self.scalars.append((name, value, global_step))

    def add_text(self, name, value, global_step):
        self.texts.append((name, value, global_step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


class FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, metrics, step):
        self.logged.append((metrics, step))

    def finish(self):
        self.finished = True


class FakeModule:
    def __init__(self, strength, density):
        self.effective_energy_coverage = 0.94
        self.effective_protect_strength = 0.88
        self.current_private_rank = 9
        self._strength = strength
        self.general_mask = torch.tensor([density], dtype=torch.float32)

    def current_protect_strength(self):
        return self._strength


class ExperimentTrackingTests(unittest.TestCase):
    def setUp(self):
        self.base_args = {
            'dataset': 'CUB',
            'model_name': 'dual_mask_branch',
            'prefix': 'B1',
            'seed': 1993,
            'total_sessions': 10,
            'device': ['0'],
        }

    def test_none_backend_imports_neither_tracker(self):
        args = dict(self.base_args, experiment_tracker='none')
        with patch.dict(
            sys.modules,
            {'torch.utils.tensorboard': None, 'wandb': None},
        ):
            self.assertIsNone(_init_experiment_tracker(args))

    def test_invalid_backend_is_rejected(self):
        args = dict(self.base_args, experiment_tracker='both')
        with self.assertRaisesRegex(ValueError, 'none, tensorboard, wandb'):
            _init_experiment_tracker(args)

    def test_tensorboard_uses_one_directory_per_seed(self):
        args = dict(
            self.base_args,
            experiment_tracker='tensorboard',
            tensorboard_logdir='logs/test_tensorboard',
        )
        module = SimpleNamespace(SummaryWriter=FakeWriter)
        with patch.dict(sys.modules, {'torch.utils.tensorboard': module}):
            backend, writer = _init_experiment_tracker(args)

        expected = Path('logs/test_tensorboard/CUB/10_tasks/B1/seed_1993')
        self.assertEqual(backend, 'tensorboard')
        self.assertEqual(Path(writer.log_dir), expected)
        self.assertIn('"seed": 1993', writer.texts[0][1])

    def test_wandb_uses_one_named_run_per_seed(self):
        captured = {}
        run = FakeWandbRun()

        def fake_init(**kwargs):
            captured.update(kwargs)
            return run

        args = dict(
            self.base_args,
            experiment_tracker='wandb',
            wandb_project='dual-mask',
            wandb_mode='offline',
        )
        with patch.dict(
            sys.modules,
            {'wandb': SimpleNamespace(init=fake_init)},
        ):
            backend, initialized_run = _init_experiment_tracker(args)

        self.assertEqual(backend, 'wandb')
        self.assertIs(initialized_run, run)
        self.assertEqual(captured['project'], 'dual-mask')
        self.assertEqual(captured['mode'], 'offline')
        self.assertEqual(captured['name'], 'CUB_B1_seed1993')

    def test_metrics_are_shared_by_both_backends(self):
        writer = FakeWriter()
        modules = [
            FakeModule(strength=0.88, density=0.8),
            FakeModule(strength=0.70, density=0.84),
        ]
        model = SimpleNamespace(
            _w0_competence=0.97,
            _feature_drift_curve=[0.03],
            _weight_drift_curve=[0.015],
            _iter_lora_modules=lambda: iter(modules),
        )

        _log_experiment_task(
            ('tensorboard', writer),
            model,
            task_id=3,
            cnn_accy={'top1': 90.0, 'grouped': {'old': 89.0, 'new': 94.0}},
            cnn_accy_with_task={'top1': 96.0},
            task_prediction_accuracy=0.91,
            w0_accuracy=88.0,
            train_seconds=12.0,
            eval_seconds=3.0,
            forgetting=4.0,
            backward=-3.0,
        )
        metrics = {name: value for name, value, step in writer.scalars if step == 3}
        self.assertEqual(metrics['accuracy/old'], 89.0)
        self.assertEqual(metrics['w0/competence'], 97.0)
        self.assertEqual(metrics['dual_mask/layer_1/protect_strength'], 0.70)

        run = FakeWandbRun()
        _log_experiment_summary(
            ('wandb', run),
            [95.0, 90.0],
            [92.0, 88.0],
            [0.98, 0.91],
            30.0,
        )
        summary, step = run.logged[-1]
        self.assertEqual(step, 2)
        self.assertEqual(summary['summary/average_accuracy'], 92.5)
        self.assertEqual(summary['summary/w0_ncm_last'], 88.0)

        _close_experiment_tracker(('tensorboard', writer))
        _close_experiment_tracker(('wandb', run))
        self.assertTrue(writer.closed)
        self.assertTrue(run.finished)


if __name__ == '__main__':
    unittest.main()
