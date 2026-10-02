import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
module_spec = importlib.util.spec_from_file_location('basis_queue', ROOT / 'scripts/run_plora_weight_basis.py')
queue = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(queue)


class BasisQueueTests(unittest.TestCase):
    def test_matched_three_groups_only_change_a_construction(self):
        base = queue.settings_for('random')
        for mode in queue.MODES:
            settings = queue.settings_for(mode)
            self.assertEqual(settings.pop('plora_a_init_mode'), mode)
            self.assertEqual(settings, {key: value for key, value in base.items() if key != 'plora_a_init_mode'})
        expected = dict(max_tasks=3, total_sessions=10, init_epoch=20, epochs=20,
            ca_epochs=5, dual_mask_anchor_reg_weight=2.5, disable_fused_sdpa=True,
            memory_size=0, save_task_weights=False, plora_train_a=False,
            p_old_gradient_oracle=False, branch_choice_mode='off',
            dual_mask_s_conflict_enabled=True, dual_mask_p_conflict_enabled=True,
            dual_mask_conflict_score_mode='conflict', p_direction_score='off',
            p_score_counterfactual_report=False, plora_a_init_batches=4)
        for key, value in expected.items():
            self.assertEqual(base[key], value, key)
        self.assertEqual(base['seed'], [1993])
        self.assertNotIn('data_path', base)
        self.assertNotIn('pretrained_path', base)
        functional = json.loads((ROOT / 'scripts/sweeps/imgr10_p_functional_score_3090.json').read_text())['common_overrides']
        intentional = {'max_tasks': 3, 'p_score_counterfactual_report': False,
                       'p_direction_score': 'off', 'plora_a_init_mode': 'random', 'wandb_group': base['wandb_group']}
        self.assertEqual(base, {**functional, **intentional})

    def test_smokes_before_all_three_t3_runs_and_no_output_for_dry_run(self):
        script = ROOT / 'scripts/10_02_imgr10_plora_weight_basis_t3_3090.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)
        output = subprocess.check_output(['bash', str(script), '--mode', 'dry-run'], cwd='/tmp', text=True)
        commands = [shlex.split(line.split('Command:', 1)[1]) for line in output.splitlines() if line.startswith('Command:')]
        self.assertEqual(len(commands), 6)
        for index, command in enumerate(commands):
            settings = dict(token.split('=', 1) for token in command if token.startswith(('max_tasks=', 'epochs=', 'plora_a_init_mode=', 'save_task_weights=')))
            self.assertEqual(settings['plora_a_init_mode'], queue.MODES[index % 3])
            self.assertEqual(settings['max_tasks'], '2' if index < 3 else '3')
            self.assertEqual(settings['epochs'], '1' if index < 3 else '20')
            self.assertEqual(settings['save_task_weights'], 'false')
        path = output.split('Outputs:', 1)[1].splitlines()[0].strip()
        self.assertFalse(Path(path).exists())

    def test_summary_reports_measured_initial_and_actual_update_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'gradient').mkdir()
            initial = dict(task=1, layer=0, mode='gradient', a_gram_relative_error=1e-6,
                gradient_energy_fraction=.8, wpre_energy_fraction=.2, history_energy_fraction=.1)
            actual = dict(task=1, layer=0, mode='gradient', raw_space_escape=1e-7,
                effective_space_escape=.3, effective_norm=1.2,
                wpre_overlap_effective=.02, history_overlap_effective=.01)
            (directory / 'gradient/training.log').write_text(
                "[trainer.py] => CNN: {'total': 89.3, 'old': 90.1, 'new': 87.7}\n" * 3 +
                '[trainer.py] => Average Accuracy: 92.7\n' +
                'PGradientAInit ' + json.dumps(initial) + '\n' +
                'PGradientAUpdate ' + json.dumps(actual) + '\n')
            queue.summarize(directory, [dict(mode='gradient', status='completed', exit_code=0, minutes=20)])
            row = json.loads((directory / 'results.json').read_text())[0]
            self.assertEqual(row['tasks_reported'], 3)
            self.assertEqual(row['Last'], 89.3)
            self.assertEqual(row['mean_effective_space_escape'], .3)
            self.assertEqual(row['max_a_gram_relative_error'], 1e-6)
            self.assertTrue((directory / 'basis_initializations.csv').is_file())
            self.assertTrue((directory / 'basis_actual_updates.csv').is_file())

    def test_algorithm_has_no_runtime_input_validators_or_checkpoint_writes(self):
        import ast
        for path in ('utils/plora_gradient_init.py',):
            tree = ast.parse((ROOT / path).read_text())
            self.assertFalse(any(isinstance(node, (ast.Assert, ast.Raise)) for node in ast.walk(tree)))
            self.assertNotIn('torch.save', (ROOT / path).read_text())

    def test_waiter_updates_only_after_original_pid_ends(self):
        path = ROOT / 'scripts/after_plora_weight_basis_3090.sh'
        subprocess.run(['bash', '-n', str(path)], check=True)
        source = path.read_text()
        self.assertLess(source.index('while kill -0'), source.index('git fetch'))
        self.assertLess(source.index('if [[ "$started"'), source.index('git fetch'))
        self.assertIn('git merge --ff-only', source)
        self.assertNotIn('git reset', source)
        self.assertNotIn('git switch', source)
        self.assertLess(source.index('python3 -m unittest'), source.index('python3 scripts/run_plora_weight_basis.py'))


if __name__ == '__main__':
    unittest.main()
