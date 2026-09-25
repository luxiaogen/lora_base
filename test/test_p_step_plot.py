import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from scripts.plot_p_step_direction import parse_logs, render, summary


ROOT = Path(__file__).resolve().parents[1]


class PlotTests(unittest.TestCase):
    def test_parser_excludes_incomplete_runs_and_preserves_metrics(self):
        # Explicit synthetic fixture; never presented as an experimental result.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            log = directory / 'synthetic.log'
            row = dict(task=1, epoch=10, batch=1, layer=0, branch='P', projection='Q',
                       mode='baseline', proposal='reference', accepted=False, mix=0,
                       fixed_gate_predicted_gain=.1, reference_fixed_gate_predicted_gain=.1,
                       effective_norm=1., reference_norm=1., conflict_norm=.3,
                       reference_conflict_norm=.3, fixed_norm=.9, switch_norm=.2,
                       mask_count=3, reference_mask_count=3, mask_jaccard=.7, cosine=1.)
            lines = ["=> overrides: ['seed=[1993]', 'prefix=synthetic', 'max_tasks=2', 'p_step_direction=baseline']",
                     "=> CNN: {'total': 90., 'old': 0., 'new': 90.}",
                     "=> CNN: {'total': 80., 'old': 75., 'new': 85.}",
                     '=> Forgetting: 15.0\tBackward: -15.0', '=> Average Accuracy: 85.',
                     '=> Last Accuracy: 80.', 'PStepDirection ' + json.dumps(row),
                     "PStepProbe {'task': 1, 'epoch': 10, 'metrics': {'reference': {'loss': 0.3, 'accuracy': 80.0, 'margin': 0.2}, 'norm': {'loss': 0.29, 'accuracy': 80.0, 'margin': 0.21}, 'conflict': {'loss': 0.3, 'accuracy': 80.0, 'margin': 0.2}}}",
                     "PStepSummary {'task': 1, 'epoch': 10, 'mode': 'baseline', 'layers': 12, 'norm_accepted': 3, 'conflict_accepted': 2, 'applied': 0}",
                     "=> overrides: ['prefix=incomplete', 'max_tasks=3', 'p_step_direction=conflict']",
                     "=> CNN: {'total': 90., 'old': 0., 'new': 90.}"]
            log.write_text('\n'.join(lines))
            runs = parse_logs([log])
            self.assertEqual(summary(runs[0])['average'], 85.)
            self.assertEqual(summary(runs[0])['new_mean_t1_on'], 85.)
            self.assertTrue(summary(runs[0])['complete'])
            self.assertFalse(summary(runs[1])['complete'])
            render(runs, directory / 'plots')
            manifest = json.loads((directory / 'plots/manifest.json').read_text())
            self.assertEqual(len(manifest['figures']), 5)
            self.assertTrue((directory / 'plots/steps.csv').is_file())

    def test_outer_log_smoke_is_not_used_as_full_result(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            log = directory / 'outer.log'
            log.write_text("=> overrides: ['prefix=smoke', 'max_tasks=2', 'epochs=1']\n"
                           "=> CNN: {'total': 91, 'old': 0, 'new': 91}\n"
                           "=> CNN: {'total': 88, 'old': 89, 'new': 87}\n"
                           "=> Average Accuracy: 89.5\n=> Last Accuracy: 88\n"
                           "=> overrides: ['prefix=full_incomplete', 'max_tasks=10', 'epochs=20']\n"
                           "=> CNN: {'total': 96, 'old': 0, 'new': 96}\n")
            render(parse_logs([log]), directory / 'plots')
            manifest = json.loads((directory / 'plots/manifest.json').read_text())
            self.assertEqual(manifest['plotted_prefixes'], ['full_incomplete'])
            self.assertEqual(manifest['figures'], [])

    def test_two_machine_specs_are_matched_and_launchers_forward_overrides(self):
        specs = []
        for gpu in ('3090', '5090'):
            spec = json.loads((ROOT / f'scripts/sweeps/imgr10_p_direction_{gpu}.json').read_text())
            self.assertEqual(spec['seeds'], [1993])
            common = spec['common_overrides'].copy()
            common.pop('wandb_group')
            self.assertNotIn('data_path', common)
            self.assertEqual(common['ca_epochs'], 5)
            self.assertEqual(common['max_tasks'], 10)
            self.assertEqual(common['dual_mask_anchor_reg_weight'], 10)
            self.assertEqual([v['overrides'] for v in spec['variants']],
                             [{'p_step_direction': m} for m in ('baseline', 'norm', 'conflict')])
            specs.append(common)
            script = ROOT / f'scripts/9_25_imgr10_p_direction_{gpu}.sh'
            subprocess.run(['bash', '-n', str(script)], check=True)
            dry = subprocess.check_output(['bash', str(script), '--dry-run'], cwd='/tmp', text=True)
            commands = dry.replace('\\\n', '').splitlines()
            self.assertEqual(len(commands), 3)
            for command, mode in zip(commands, ('baseline', 'norm', 'conflict')):
                tokens = shlex.split(command)
                settings = dict(tokens[i + 1].split('=', 1) for i, t in enumerate(tokens) if t == '--set')
                self.assertEqual(settings['p_step_direction'], mode)
                self.assertEqual(settings['ca_epochs'], '5')
                self.assertIn('$@', tokens)
                self.assertIn('${TIMESTAMP}', settings['prefix'])
        self.assertEqual(*specs)


if __name__ == '__main__':
    unittest.main()
