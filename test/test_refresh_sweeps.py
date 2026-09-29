import json
from pathlib import Path
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RefreshSweepTests(unittest.TestCase):
    def spec(self, group):
        return json.loads((ROOT / 'scripts/sweeps' / (group + '.json')).read_text())

    def test_shared_baseline_and_only_requested_factors(self):
        spec = self.spec('imgr10_granularity_budget_refresh_3090')
        self.assertEqual([v['overrides'] for v in spec['variants']], [
            {}, {'dual_mask_conflict_granularity': 'projection'},
            {'dual_mask_conflict_granularity': 'model'},
            {'dual_mask_conflict_budget_multiplier': .5},
            {'dual_mask_conflict_budget_multiplier': 1.5},
            {'dual_mask_conflict_score_mode': 'magnitude'}])

    def test_fixed_coverage_endpoints_and_protocol(self):
        for group in ['imgr10_granularity_budget_refresh_3090', 'imgr10_coverage_refresh_5090']:
            spec = self.spec(group)
            common = spec['common_overrides']
            self.assertEqual(spec['seeds'], [1993])
            for key, value in {'dual_mask_anchor_reg_weight': 2.5, 'ca_epochs': 5,
                               'init_epoch': 20, 'epochs': 20, 'max_tasks': 10,
                               'disable_fused_sdpa': True, 'save_task_weights': False,
                               'dual_mask_uniform_norm_matched': False,
                               'p_conflict_freeze_epoch': 0}.items():
                self.assertEqual(common[key], value)
            self.assertNotIn('data_path', common)
            script = ROOT / 'scripts' / ('9_29_' + group + '.sh')
            subprocess.run(['bash', '-n', str(script)], check=True)
            text = script.read_text().replace('\\\n', ' ')
            commands = [line.strip() for line in text.splitlines() if line.strip().startswith('python main.py')]
            self.assertEqual(len(commands), 6)
            for command, variant in zip(commands, spec['variants']):
                tokens = shlex.split(command.split('2>&1')[0])
                settings = dict(tokens[i + 1].split('=', 1) for i, token in enumerate(tokens) if token == '--set')
                self.assertEqual(settings['save_task_weights'], 'false')
                self.assertEqual(settings['dual_mask_uniform_norm_matched'], 'false')
                for key, value in variant['overrides'].items():
                    self.assertEqual(settings[key], str(value))
            if 'coverage' in group:
                self.assertTrue(common['dual_mask_conflict_exact_topk'])
                self.assertFalse(common['dual_mask_conflict_energy_adaptive'])
                self.assertFalse(common['dual_mask_conflict_old_overlap_adaptive'])
                self.assertEqual([v['overrides']['dual_mask_conflict_ratio'] for v in spec['variants']], [.1, 0, .2, .4, .6, 1])


if __name__ == '__main__':
    unittest.main()
