import copy
import json
from pathlib import Path
import subprocess
import unittest

from scripts.run_head_start_night import read_tasks, screen_result, full_candidates, run_settings


def rows():
    return {t: {'sample_sha256': str(t), 'class_count': (t+1)*20, 'source': 'train_holdout',
                'metrics': {g: {'accuracy': 90.} for g in ('total', 'old', 'new')}}
            for t in range(3)}


def spec(machine='3090'):
    return json.loads(Path(f'scripts/sweeps/imgr10_head_start_anchor_{machine}.json').read_text())


class HeadStartQueueTests(unittest.TestCase):
    def test_selection_requires_net_improvement(self):
        a, b = rows(), rows()
        self.assertFalse(screen_result(a, b)['pass'])
        for t in (1, 2):
            b[t]['metrics']['total']['accuracy'] += .3
            b[t]['metrics']['new']['accuracy'] += .6
        self.assertTrue(screen_result(a, b)['pass'])
        b[2]['metrics']['old']['accuracy'] -= .1
        self.assertFalse(screen_result(a, b)['pass'])

    def test_bad_single_task_cannot_hide_in_average(self):
        a, b = rows(), rows()
        for t in (1, 2):
            b[t]['metrics']['total']['accuracy'] += .3
        b[1]['metrics']['new']['accuracy'] -= .4
        b[2]['metrics']['new']['accuracy'] += 1
        self.assertFalse(screen_result(a, b)['pass'])

    def test_identity_completion_task0_and_anchor_exception(self):
        a, b = rows(), rows()
        for t in (1, 2):
            b[t]['metrics']['total']['accuracy'] += .3
        b[0]['metrics']['total']['accuracy'] -= 1
        self.assertFalse(screen_result(a, b)['pass'])
        self.assertTrue(screen_result(a, b, anchor=True)['pass'])
        b[1]['sample_sha256'] = 'different'
        self.assertFalse(screen_result(a, b, anchor=True)['pass'])
        del b[1]
        self.assertFalse(screen_result(a, b, anchor=True)['pass'])

    def test_only_holdout_lines_are_read(self):
        row = dict(rows()[0], task=0)
        text = 'CNN accuracy: 100\nStageAudit {}\nIncrementalHoldout ' + json.dumps(row)
        self.assertEqual(read_tasks(text), {0: row})

    def test_volume_control_matches_initialization_and_best_anchor_only(self):
        decisions = {n: {'pass': True, 'delta': {'total': d}}
                     for n, d in [('proto_warm', .3), ('anchor0', .4), ('anchor5', .5)]}
        selected = full_candidates(spec(), decisions)
        self.assertEqual([v['name'] for v in selected], ['proto_warm', 'anchor5', 'proto_head_post'])
        self.assertEqual(selected[-1]['overrides']['head_start_init'], 'prototype')
        decisions['warm'] = {'pass': True, 'delta': {'total': .3}}
        self.assertEqual([v['name'] for v in full_candidates(spec(), decisions)[-2:]],
                         ['head_post', 'proto_head_post'])
        self.assertEqual(full_candidates(spec(), {}), [])

    def test_full_removes_holdout_but_keeps_T10_protocol(self):
        s = spec()
        for phase in ('screen', 'full'):
            settings = run_settings(s, s['variants'][1], phase)
            self.assertEqual(settings['incremental_holdout'], phase == 'screen')
            self.assertEqual(settings['max_tasks'], 3 if phase == 'screen' else 10)
            self.assertEqual(settings['total_sessions'], 10)
            self.assertEqual(settings['seed'], [1993])
            self.assertEqual(settings['ca_epochs'], 5)
            self.assertEqual(settings['init_epoch'], 20)
            self.assertFalse(settings['plora_train_a'])
            self.assertEqual(settings['old_model_distill_weight'], 0)
            self.assertTrue(settings['dual_mask_anchor_reg_task0_only'])
            self.assertNotIn('data_path', settings)

    def test_anchor_only_changes_its_weight(self):
        s = spec()
        variants = [v for v in s['variants'] if v['name'].startswith('anchor')]
        self.assertEqual([v['overrides'] for v in variants],
                         [{'dual_mask_anchor_reg_weight': w} for w in (0, 2.5, 5, 20)])
        for v in variants:
            settings = run_settings(s, v, 'screen')
            self.assertEqual(settings['head_start_init'], 'random')
            self.assertEqual(settings['head_start_epochs'], 0)

    def test_reference_recipe_matches_3090_and_not_5090(self):
        s = spec()
        ref = json.loads(Path(s['reference']).read_text())
        expected = copy.deepcopy(ref['settings'])
        actual = copy.deepcopy(s['common_overrides'])
        expected.pop('wandb_group')
        actual.pop('wandb_group')
        self.assertEqual(expected, actual)
        self.assertEqual(ref['commit'], 'c416e0d')
        self.assertEqual(set(ref['run']['tasks']), {'0', '1', '2'})
        self.assertIsNone(spec('5090')['reference'])
        self.assertEqual(spec('5090')['common_overrides']['dual_mask_anchor_reg_weight'], 5)

    def test_dry_run_never_duplicates_full_baseline(self):
        output = subprocess.check_output(['bash', 'scripts/9_28_imgr10_head_start_anchor_3090.sh',
                                          '--dry-run'], text=True)
        self.assertIn('Reusing short control', output)
        self.assertNotIn('Starting: control', output)
        self.assertEqual(output.count('phase=screen'), 7)
        self.assertIn('phase=full', output)
        smoke = subprocess.check_output(['bash', 'scripts/9_28_imgr10_head_start_5090.sh',
                                         '--dry-run', '--smoke'], text=True)
        self.assertNotIn('phase=full', smoke)
        self.assertIn('Starting: head_post', smoke)
