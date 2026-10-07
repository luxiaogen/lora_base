import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_tail_update as runner
import analyze_tail_update as analyzer
from analyze_tail_update import paired_results


class TailQueueTests(unittest.TestCase):
    def test_matrix_order_dataset_recipes_and_single_factor(self):
        for machine, primary, count in (('3090', ['M','C','T'], 9), ('5090', ['O','M','C','T','U'], 21)):
            self.assertEqual(len(runner.modes(machine)), count)
            self.assertEqual(runner.modes(machine)[:len(primary)], [v + '_seed1993' for v in primary])
            runner.validate_settings(machine)
            path = runner.spec()[machine]['config']
            source = json.loads((runner.ROOT / path).read_text())
            reference = runner.settings_for(machine, 'M_seed1993')
            effective = dict(source, **reference)
            for field in runner.baseline.DATASET_FIELDS:
                if field in source:
                    self.assertEqual(effective[field], source[field])
            for variant, rule in (('C','soft_tail'), ('T','step_tail_matched')):
                candidate = runner.settings_for(machine, variant + '_seed1993')
                changed = {k for k in reference.keys() | candidate.keys() if reference.get(k) != candidate.get(k)}
                self.assertEqual(changed, {'dual_mask_update_rule'})
                self.assertEqual(candidate['dual_mask_update_rule'], rule)
            for name in runner.modes(machine):
                settings = runner.settings_for(machine, name)
                self.assertEqual(settings['init_epoch'], 20)
                self.assertFalse(settings['save_task_weights'])
                self.assertEqual(settings['dual_mask_anchor_reg_weight'], 2.5)
                self.assertEqual(settings['task0_margin'], source['margin'])
                self.assertEqual(settings['epochs'], 40 if runner.identity(name)[0].endswith('40') else 20)
                smoke = runner.settings_for(machine, name, True)
                self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs']), (2,1,1))
            original = runner.settings_for(machine, 'O_seed1993')
            self.assertEqual(original['dual_mask_reg_weight'], .01)
            self.assertIsNone(original['dual_mask_fixed_coverage'])
            self.assertEqual(original['dual_mask_private_rank'], 0)

    def test_runner_rejects_wrong_score_protocol(self):
        source = runner.settings_for
        def bad(machine, name, smoke=False):
            config = source(machine, name, smoke)
            config['dual_mask_conflict_granularity'] = 'projection'
            return config
        with patch.object(runner, 'settings_for', side_effect=bad):
            with self.assertRaises(ValueError):
                runner.validate_settings('3090')

    def simulate(self, directory, **kwargs):
        calls = []
        def run(machine, name, path, revision, smoke=False, dry_run=False):
            calls.append((name, smoke))
            return dict(mode=name,status='completed',exit_code=0,minutes=1)
        with patch.object(runner.engine, 'run', side_effect=run), \
                patch.object(runner, 'summarize_safely'), contextlib.redirect_stdout(io.StringIO()):
            result = runner.execute('3090', runner.modes('3090'), directory, 'revision', **kwargs)
        return result, calls

    def test_one_smoke_per_recipe_then_formal_and_completed_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            result, calls = self.simulate(directory)
            self.assertEqual(result, 0)
            self.assertEqual(calls[:3], [(n,True) for n in runner.modes('3090')[:3]])
            self.assertEqual(calls[3:], [(n,False) for n in runner.modes('3090')])
            self.assertEqual(self.simulate(directory)[1], [])
            self.assertEqual(len(json.loads((directory / 'queue.json').read_text())),9)

    def test_deadline_defers_only_unstarted_formal_runs(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner.time,'monotonic', side_effect=[0,0,0,0,0,3600]):
            directory = Path(temp)
            result, calls = self.simulate(directory, hours=1)
            self.assertEqual(result, 0)
            self.assertEqual(len(calls),4)
            rows = json.loads((directory / 'queue.json').read_text())
            self.assertEqual(rows[0]['status'], 'completed')
            self.assertTrue(all(r['status'] == 'time_budget_pending' for r in rows[1:]))

    def test_smoke_and_training_failure_stop_but_analysis_failure_continues(self):
        for smoke_failure in (True,False):
            calls=[]
            def run(m,n,d,r,smoke=False,dry_run=False):
                calls.append((n,smoke))
                return dict(mode=n,status='failed' if smoke == smoke_failure else 'completed',exit_code=7 if smoke == smoke_failure else 0,minutes=1)
            with tempfile.TemporaryDirectory() as temp, patch.object(runner.engine,'run', side_effect=run), \
                    patch.object(runner,'summarize_safely'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.execute('3090',runner.modes('3090'),Path(temp),'r'),7)
                self.assertEqual(len(calls),1 if smoke_failure else 4)
        with tempfile.TemporaryDirectory() as temp, patch('analyze_tail_update.summarize',side_effect=RuntimeError('plot error')), \
                patch.object(runner.engine,'run',side_effect=lambda m,n,d,r,smoke=False,dry_run=False:
                    dict(mode=n,status='completed',exit_code=0,minutes=1)), contextlib.redirect_stdout(io.StringIO()):
            directory=Path(temp)
            self.assertEqual(runner.execute('3090',runner.modes('3090'),directory,'r'),0)
            self.assertTrue((directory / 'analysis_errors.jsonl').exists())
            self.assertEqual(len(json.loads((directory / 'queue.json').read_text())),9)

    def test_resume_does_not_reset_deadline(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner.time,'time',return_value=200):
            directory=Path(temp)
            (directory / 'budget.json').write_text(json.dumps(dict(deadline_unix=100,hours=10)))
            result,calls=self.simulate(directory)
            self.assertEqual(result,0)
            self.assertEqual(calls,[])
            self.assertTrue(all(r['status']=='time_budget_pending' for r in json.loads((directory / 'queue.json').read_text())))

    def test_resume_revision_and_fingerprint_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            (directory / 'manifest.json').write_text(json.dumps(dict(machine='3090',revision='r')))
            with self.assertRaises(ValueError):
                runner.check_resume(directory,'3090','different')
            row=dict(mode='M_seed1993',status='completed',exit_code=0)
            (directory / 'queue.json').write_text(json.dumps([row]))
            config=dict(json.loads((runner.ROOT / runner.spec()['3090']['config']).read_text()),
                        **runner.settings_for('3090','M_seed1993'))
            snapshot=dict(machine='3090',code_revision='r',phase='formal',effective_config=config,source_sha256={})
            measured=dict(runtime_error=False,tasks_reported=10,**{k:1 for k in runner.METRICS})
            with patch.object(runner,'read_run',return_value=(measured,snapshot,{})):
                runner.check_resume(directory,'3090','r')
                snapshot['source_sha256']={'models/attention.py':'bad-hash'}
                with self.assertRaises(ValueError):
                    runner.check_resume(directory,'3090','r')
                snapshot['source_sha256']={}
                config['dual_mask_update_rule']='soft_tail'
                with self.assertRaises(ValueError):
                    runner.check_resume(directory,'3090','r')

    def test_resume_rejects_live_queue_after_child_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            (directory / 'manifest.json').write_text(json.dumps(dict(machine='3090',revision='r')))
            (directory / 'active.json').write_text(json.dumps(dict(queue_pid=123,training_pid=456,status='completed')))
            with patch.object(runner.os,'kill') as kill:
                with self.assertRaises(ValueError):
                    runner.check_resume(directory,'3090','r')
                kill.assert_called_once_with(123,0)
            with patch.object(runner.os,'kill',side_effect=ProcessLookupError) as kill:
                runner.check_resume(directory,'3090','r')
                kill.assert_called_once_with(123,0)

    def test_partial_pairs_do_not_produce_three_seed_statistics(self):
        def metric(value):
            return {k:value for k in runner.METRICS}
        complete={('M',1993):metric(1),('C',1993):metric(2)}
        result=paired_results(complete,[1993,1996,1997])[0]
        self.assertFalse(result['complete_three_seeds'])
        self.assertNotIn('Average_mean',result)
        for seed in (1996,1997):
            complete['M',seed],complete['C',seed]=metric(1),metric(2)
        result=paired_results(complete,[1993,1996,1997])[0]
        self.assertEqual(result['Average_mean'],1)
        self.assertTrue(result['practical_followup_threshold_met'])

    def test_mismatch_clears_previous_comparison_csv_and_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            records=[dict(mode=v + '_seed' + str(s),status='completed',exit_code=0)
                     for s in (1993,1996,1997) for v in ('M','C')]
            epoch_log='\n'.join('LoRA learning rates: task=' + str(t) + ', epoch=' + str(e)
                                for t in range(10) for e in range(1,21))
            for record in records:
                path=directory / record['mode']
                path.mkdir()
                (path / 'training.log').write_text(epoch_log)
            mismatched=False
            def read(path, record):
                variant,_=runner.identity(record['mode'])
                config=dict(json.loads((runner.ROOT / runner.spec()['3090']['config']).read_text()),
                            **runner.settings_for('3090',record['mode']))
                snapshot=dict(mode=record['mode'],code_revision='r',machine='3090',phase='formal',
                    effective_config=config,software={'torch':'same'},hardware={'gpu':'same'},
                    source_sha256={'models/attention.py':'changed' if mismatched and record['mode']=='C_seed1997' else 'same'})
                row=dict(record,valid_performance=True,**{k:2 if variant=='C' else 1 for k in runner.METRICS})
                return row,snapshot,dict(tasks=[])
            with patch.object(analyzer,'read_run',side_effect=read), \
                    patch.object(analyzer,'historical_original'), patch.object(analyzer,'draw'):
                analyzer.summarize(directory,'3090',records)
            self.assertIn('Average_mean',(directory / 'pairs.csv').read_text())
            self.assertTrue(json.loads((directory / 'pairs.json').read_text())[0]['complete_three_seeds'])
            for name in ('paired_accuracy','old_new','intervention_and_gate_change'):
                for extension in ('png','pdf'):
                    (directory / (name + '.' + extension)).write_text('old generated plot')
            mismatched=True
            with patch.object(analyzer,'read_run',side_effect=read), patch.object(analyzer,'historical_original'):
                analyzer.summarize(directory,'3090',records)
            self.assertEqual(json.loads((directory / 'pairs.json').read_text()),[])
            self.assertEqual((directory / 'pairs.csv').read_text(),'')
            self.assertTrue(json.loads((directory / 'matching_issues.json').read_text()))
            self.assertNotIn('Average_mean',(directory / 'aggregate.csv').read_text())
            self.assertFalse(any(directory.glob('*.png')))
            self.assertFalse(any(directory.glob('*.pdf')))


if __name__ == '__main__':
    unittest.main()
