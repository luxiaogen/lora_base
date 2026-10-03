import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_protect_position as queue
from analyze_protect_position import METRICS, factor_effects, summarize


class ProtectPositionQueueTests(unittest.TestCase):
    def test_four_cells_change_only_the_two_factors(self):
        common = queue.settings_for('A')
        for mode in queue.MODES:
            config = queue.settings_for(mode)
            self.assertEqual(config['dual_mask_protect_position'], 'wpre' if mode in ('A', 'B') else 'permuted')
            self.assertEqual(config['dual_mask_conflict_score_mode'], 'conflict' if mode in ('A', 'C') else 'magnitude')
            self.assertEqual({k: v for k, v in config.items() if k not in ('dual_mask_protect_position', 'dual_mask_conflict_score_mode')},
                             {k: v for k, v in common.items() if k not in ('dual_mask_protect_position', 'dual_mask_conflict_score_mode')})
            for key, expected in dict(seed=[1993], max_tasks=10, init_epoch=20, epochs=20, ca_epochs=5,
                    dual_mask_anchor_reg_weight=2.5, disable_fused_sdpa=True,
                    save_task_weights=False, dual_mask_vis_save_weight=False,
                    dual_mask_position_audit=True, dual_mask_conflict_reg_original_score=True,
                    plora_a_init_mode='off', plora_train_a=False,
                    p_direction_score='off', p_score_counterfactual_report=False,
                    dual_mask_composed_conflict='off', dual_mask_private_conflict_mode='global',
                    dual_mask_conflict_merge_mode='suppress', dual_mask_conflict_granularity='layer',
                    dual_mask_s_protect_enabled=True, dual_mask_uniform_norm_matched=False,
                    wpre_distill_weight=0, ridge_fusion_enabled=False, old_model_distill_weight=0,
                    p_old_gradient_oracle=False, stage_audit=False).items():
                self.assertEqual(config[key], expected)
            self.assertNotIn('data_path', config)
            self.assertNotIn('device', config)

    def test_smoke_and_formal_are_explicit(self):
        for mode in queue.MODES:
            smoke = queue.settings_for(mode, True)
            self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs'], smoke['ca_epochs']), (2, 1, 1, 1))
            self.assertTrue(smoke['stage_audit'])
            command, config = queue.command_for(mode, Path('test_outputs') / mode)
            self.assertIn('max_tasks=10', command)
            self.assertIn('exps/dlora/imgr10.json', command)
            self.assertFalse(config['save_task_weights'])

    def fixture(self, directory, modes=('A',), tasks=10, phase='formal', exit_code=0, diagnostics=False):
        records = []
        for mode in modes:
            output = directory / mode
            output.mkdir()
            config = queue.settings_for(mode)
            (output / 'run.json').write_text(json.dumps(dict(machine='3090', phase=phase,
                hardware={'hostname': 'synthetic-fixture', 'gpus': ['synthetic-fixture']},
                effective_config=config, source_sha256={'models/attention.py': 'synthetic-fixture'}, software={})))
            content = (
                "[trainer.py] => CNN: {'total': 82.6, 'old': 82.1, 'new': 87.1}\n" * tasks +
                '[trainer.py] => Average Accuracy: 87.214\n[trainer.py] => Forgetting: 5.8\n')
            if diagnostics:
                for task in range(tasks):
                    for layer in range(12):
                        content += 'ProtectionPositionMask ' + json.dumps(dict(task=task, layer=layer)) + '\n'
                        for branch in (('S',) if task == 0 else ('S', 'P')):
                            for projection in ('Q', 'K', 'V'):
                                row = dict(task=task, layer=layer, branch=branch, projection=projection,
                                    raw_norm=1., effective_norm=.7, total_removed_norm=.3,
                                    total_removed_ratio=.3, conflict_removed_norm=.1,
                                    conflict_removed_ratio=.125, protect_density=.5,
                                    effective_conflict_density=.1, merge_error=0.)
                                content += 'ProtectionPositionUpdate ' + json.dumps(row) + '\n'
            (output / 'training.log').write_text(content)
            records.append(dict(mode=mode, exit_code=exit_code, minutes=70))
        return records

    def test_partial_failed_and_smoke_cannot_form_factor_evidence(self):
        for tasks, phase, code in ((3, 'formal', 0), (10, 'smoke', 0), (10, 'formal', 1)):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                records = self.fixture(directory, queue.MODES, tasks, phase, code)
                with contextlib.redirect_stdout(io.StringIO()):
                    summarize(directory, records)
                report = json.loads((directory / 'factor_effects.json').read_text())
                self.assertFalse(report['ready'])
                self.assertEqual(report['effects'], {})
                self.assertFalse((directory / 'old_new.png').exists())

    def test_unexpected_configuration_difference_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = self.fixture(directory, queue.MODES)
            snapshot = json.loads((directory / 'D/run.json').read_text())
            snapshot['effective_config']['epochs'] = 30
            (directory / 'D/run.json').write_text(json.dumps(snapshot))
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(directory, records)
            report = json.loads((directory / 'factor_effects.json').read_text())
            self.assertFalse(report['ready'])
            self.assertIn(dict(mode='D', field='config.epochs'), report['unexpected_differences'])

    def test_missing_actual_update_diagnostics_is_not_complete_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = self.fixture(directory, queue.MODES)
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(directory, records)
            report = json.loads((directory / 'factor_effects.json').read_text())
            self.assertTrue(report['metrics_complete'])
            self.assertFalse(report['diagnostics_complete'])
            self.assertFalse(report['ready'])
            self.assertEqual(report['effects'], {})

    def test_complete_synthetic_fixture_produces_report_and_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = self.fixture(directory, queue.MODES, diagnostics=True)
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(directory, records)
            report = json.loads((directory / 'factor_effects.json').read_text())
            self.assertTrue(report['ready'])
            self.assertEqual(report['unexpected_differences'], [])
            self.assertTrue((directory / 'old_new.png').exists())
            self.assertTrue((directory / 'old_new.pdf').exists())

    def test_environment_and_task0_differences_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = self.fixture(directory, queue.MODES, diagnostics=True)
            snapshot = json.loads((directory / 'D/run.json').read_text())
            snapshot['hardware']['hostname'] = 'another-machine'
            (directory / 'D/run.json').write_text(json.dumps(snapshot))
            log = directory / 'D/training.log'
            log.write_text(log.read_text().replace("'total': 82.6", "'total': 81.6", 1))
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(directory, records)
            report = json.loads((directory / 'factor_effects.json').read_text())
            self.assertFalse(report['ready'])
            self.assertIn(dict(mode='D', field='hardware'), report['unexpected_differences'])
            self.assertIn(dict(mode='D', field='Task0'), report['unexpected_differences'])

    def test_factor_effects_have_defined_signs(self):
        cells = {mode: {key: value for key in METRICS} for mode, value in zip(queue.MODES, (4, 3, 2, 0))}
        effects = factor_effects(cells)
        self.assertEqual(effects['Average']['wpre_vs_permuted'], 2.5)
        self.assertEqual(effects['Average']['product_vs_magnitude'], 1.5)
        self.assertEqual(effects['Average']['interaction'], -1)
        self.assertEqual(effects['Average']['magnitude_vs_product_at_wpre'], -1)


if __name__ == '__main__':
    unittest.main()
