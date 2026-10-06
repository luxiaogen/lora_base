import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_relative_score as runner


class RelativeQueueTests(unittest.TestCase):
    def test_only_score_and_logging_change_from_existing_m(self):
        for machine, mode in (('3090', 'R0'), ('5090', 'S2')):
            reference = runner.queue.settings_for(machine, mode)
            runner.validate_settings(machine)
            for candidate in ('wpre_relative', 'task_relative'):
                actual = runner.settings_for(machine, candidate)
                changed = {k for k in reference.keys() | actual.keys() if reference.get(k) != actual.get(k)}
                self.assertEqual(changed, {'dual_mask_conflict_score_mode', 'wandb_group'})
                self.assertFalse(actual['save_task_weights'])
                self.assertEqual(actual['dual_mask_conflict_ratio'], .1)
                smoke = runner.settings_for(machine, candidate, True)
                self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2, 1, 1))

    def test_wrong_protocol_rejected_in_runner_not_training(self):
        bad = runner.settings_for('3090', 'wpre_relative')
        bad['dual_mask_conflict_granularity'] = 'model'
        with patch.object(runner, 'settings_for', return_value=bad):
            with self.assertRaises(ValueError):
                runner.validate_settings('3090')

    def test_missing_or_different_reference_cannot_be_compared(self):
        identity = dict(effective_config={}, training_source_sha256={'trainer.py': 'abc'},
                        hardware={'gpus': ['GPU1']}, software={'torch': 'x'})
        snapshot = dict(effective_config=runner.settings_for('3090', 'wpre_relative'),
            source_sha256={'trainer.py': 'abc'}, machine='3090', phase='formal',
            hardware=identity['hardware'], software=identity['software'], mode='wpre_relative')
        summary = dict(valid_performance=True, epoch_records_complete=True, position_diagnostics_complete=True)
        self.assertEqual(runner.matching_issues(summary, snapshot, '3090', 'wpre_relative', identity), [])
        for field, value in (('hardware', {}), ('source_sha256', {'trainer.py': 'changed'})):
            changed = copy.deepcopy(snapshot)
            changed[field] = value
            self.assertTrue(runner.matching_issues(summary, changed, '3090', 'wpre_relative', identity))
        summary['valid_performance'] = False
        self.assertIn('incomplete_or_unhealthy_T10', runner.matching_issues(summary, snapshot, '3090', 'wpre_relative', identity))

    def test_fixed_queue_analysis_failure_does_not_stop_second_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            def run(machine, name, path, revision, smoke=False, dry_run=False):
                return dict(mode=name, status='completed', exit_code=0, minutes=1)
            with patch.object(runner.queue.engine, 'run', side_effect=run) as execute, \
                    patch.object(runner.queue, 'analysis', runner), \
                    patch.object(runner, 'summarize', side_effect=RuntimeError('plot failed')):
                result = runner.queue.execute_queue('3090', ['wpre_relative', 'task_relative'],
                    directory, 'abc', hours=24)
            self.assertEqual(result, 0)
            self.assertEqual(execute.call_count, 4)
            self.assertTrue((directory / 'analysis_errors.jsonl').exists())
            self.assertEqual(len(json.loads((directory / 'queue.json').read_text())), 2)

    def test_smoke_failure_prevents_formal_and_resume_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(runner.queue.engine, 'run', return_value=dict(
                    mode='wpre_relative', status='failed', exit_code=1)) as execute:
                self.assertEqual(runner.queue.execute_queue('3090', ['wpre_relative', 'task_relative'],
                    directory, 'abc'), 1)
                self.assertEqual(execute.call_count, 1)
            (directory / 'queue.json').write_text(json.dumps([dict(mode='wpre_relative', status='completed')]))
            with patch.object(runner.queue.engine, 'run') as execute:
                runner.queue.execute_queue('3090', ['wpre_relative'], directory, 'abc')
                execute.assert_not_called()

    def test_resume_rejects_changed_revision_or_failed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'manifest.json').write_text(json.dumps(dict(machine='3090', revision='abc')))
            with self.assertRaises(ValueError):
                runner.check_resume(directory, '3090', 'different')
            (directory / 'queue.json').write_text(json.dumps([dict(status='failed')]))
            with self.assertRaises(ValueError):
                runner.check_resume(directory, '3090', 'abc')


if __name__ == '__main__':
    unittest.main()
