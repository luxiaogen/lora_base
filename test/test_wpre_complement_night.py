"""Predeclared nightly recipes, real training commands and time budgeting."""
import importlib.util
from pathlib import Path
import sys
import unittest


def runner():
    sys.path.insert(0, str(Path('scripts').resolve()))
    try:
        spec = importlib.util.spec_from_file_location('wpre_complement_runner', 'scripts/run_wpre_complement.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class NightRecipesTests(unittest.TestCase):
    def test_six_recipes_share_full_training_baseline_and_batch_denominator(self):
        module = runner()
        expected = [('complement_s', 'complement', 's'), ('random_s', 'random_matched', 's'),
                    ('teacher_correct_s', 'teacher_correct', 's'), ('complement_p', 'complement', 'p'),
                    ('random_p', 'random_matched', 'p'), ('complement_sp', 'complement', 'sp')]
        self.assertEqual(module.VARIANTS, expected)
        recipes = [module.settings_for(name, selection, scope, 't10', Path('run'))
                   for name, selection, scope in expected]
        for settings in recipes:
            for key, value in dict(seed=[1993], max_tasks=10, init_epoch=20, epochs=20,
                    ca_epochs=5, dual_mask_anchor_reg_weight=2.5, disable_fused_sdpa=True,
                    wpre_distill_weight=1., wpre_distill_normalization='batch', save_task_weights=False,
                    plora_train_a=False, dual_mask_composed_conflict='off',
                    two_expert_calibration_holdout_mod=0, incremental_holdout=False,
                    ridge_fusion_enabled=False, two_expert_oracle_diagnostic=False).items():
                self.assertEqual(settings[key], value, key)
        ignored = ('wpre_distill_selection', 'wpre_distill_scope', 'prefix')
        reference = {k: v for k, v in recipes[0].items() if k not in ignored}
        for recipe in recipes[1:]:
            self.assertEqual({k: v for k, v in recipe.items() if k not in ignored}, reference)

    def test_smoke_does_not_replace_full_training_and_command_runs_main(self):
        module = runner()
        smoke = module.settings_for('complement_s', 'complement', 's', 'smoke', Path('run'))
        self.assertEqual((smoke['max_tasks'], smoke['init_epoch'], smoke['epochs'], smoke['ca_epochs']),
                         (2, 1, 1, 1))
        command = module.training_command(smoke)
        self.assertEqual(command[1:4], ['main.py', '--config', 'exps/dlora/imgr10.json'])
        self.assertIn('wpre_distill_selection=complement', command)
        self.assertIn('save_task_weights=false', command)

    def test_budget_reserves_time_for_the_next_whole_run(self):
        module = runner()
        self.assertTrue(module.fits_budget(0., 480., []))
        self.assertTrue(module.fits_budget(390., 480., [70., 72.]))
        self.assertFalse(module.fits_budget(410., 480., [70., 72.]))
        self.assertFalse(module.fits_budget(479., 480., []))


if __name__ == '__main__':
    unittest.main()
