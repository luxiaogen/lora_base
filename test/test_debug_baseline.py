"""调试main的入口、固定配方及日志格式契约。"""
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DebugBaselineTests(unittest.TestCase):
    def test_default_config_is_current_ImageNet_R_B(self):
        cfg = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
        self.assertEqual(cfg['seed'], 1993)
        self.assertEqual(cfg['rank'], 64)
        self.assertEqual(cfg['init_epoch'], 20)
        self.assertEqual(cfg['ca_epochs'], 5)
        self.assertEqual(cfg['margin'], .1)
        self.assertEqual(cfg['logit_norm'], .1)
        self.assertEqual(cfg['max_tasks'], 10)

    def test_log_uses_main_format(self):
        from trainer import LOG_FORMAT
        record = logging.LogRecord('debug', logging.INFO, '/tmp/trainer.py', 8,
                                   'Average Accuracy: %.3f', (87.297,), None)
        line = logging.Formatter(LOG_FORMAT).format(record)
        self.assertIn('[trainer.py] => Average Accuracy: 87.297', line)

    def test_entry_accepts_config_and_operational_overrides(self):
        spec = importlib.util.spec_from_file_location('debug_main', ROOT / 'main.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.load_args(['--config', str(ROOT / 'exps/dlora/imgr10.json'),
                                 '--set', 'data_path=/tmp/images', '--epochs', '1', '--max-tasks', '2'])
        self.assertEqual(args['data_path'], str(Path('/tmp/images').resolve()))
        self.assertEqual(args['init_epoch'], 1)
        self.assertEqual(args['epochs'], 1)
        self.assertEqual(args['max_tasks'], 2)

    def test_entry_rejects_silent_dataset_or_rank_mismatch(self):
        spec = importlib.util.spec_from_file_location('debug_main', ROOT / 'main.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for setting in ('dataset=CIFAR100', 'rank=96'):
            with self.subTest(setting=setting), self.assertRaises(SystemExit):
                module.load_args(['--set', setting])

    def test_entry_rejects_removed_research_controls(self):
        spec = importlib.util.spec_from_file_location('debug_main', ROOT / 'main.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for setting in ('layerwise_alpha=true', 'dual_mask_competence_adaptive=true'):
            with self.subTest(setting=setting), self.assertRaises(SystemExit):
                module.load_args(['--set', setting])

    def test_external_config_cannot_restore_removed_controls(self):
        spec = importlib.util.spec_from_file_location('debug_main', ROOT / 'main.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
        for key in ('layerwise_alpha', 'dual_mask_competence_adaptive', 'dual_mask_depth_strength'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'config.json'
                path.write_text(json.dumps(dict(base, **{key: True})))
                with self.assertRaises(SystemExit):
                    module.load_args(['--config', str(path)])

    def test_metadata_reports_actual_fixed_B_and_inactive_task0_P(self):
        import torch
        from trainer import _metadata
        args = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
        args['device'] = [torch.device('cpu')]
        fixed = _metadata(args)['recipe']
        self.assertEqual(fixed['protect_alpha'], .5)
        self.assertEqual(fixed['svd_energy_coverage'], .95)
        self.assertEqual(fixed['task0_P_init_rank'], 39)
        self.assertFalse(fixed['task0_P_active'])


if __name__ == '__main__':
    unittest.main()
