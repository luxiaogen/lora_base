import ast
from pathlib import Path
import shlex
import subprocess
import unittest

import torch


def average_function():
    tree = ast.parse(Path('methods/dlora.py').read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_apply_epoch_average')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<late-average>', 'exec'), namespace)
    return namespace[function.name]


class LateWeightAverageTests(unittest.TestCase):
    def test_final_five_epochs_average_only_selected_parameters(self):
        apply_average = average_function()
        shared = torch.nn.Parameter(torch.zeros(2, 2))
        private = torch.nn.Parameter(torch.zeros(2, 2))
        head = torch.nn.Parameter(torch.zeros(2, 2))
        old_head = torch.nn.Parameter(torch.full((2, 2), 9.0))
        params = [shared, private, head]
        totals = [torch.zeros_like(param) for param in params]
        random_state = torch.get_rng_state().clone()

        for epoch in range(1, 21):
            with torch.no_grad():
                shared.fill_(epoch)
                private.fill_(epoch * 2)
                head.fill_(epoch * 3)
                if epoch >= 16:
                    for running_sum, param in zip(totals, params):
                        running_sum.add_(param)

        shift = apply_average(params, totals, 5)
        self.assertAlmostEqual(shared[0, 0].item(), 18.0)
        self.assertAlmostEqual(private[0, 0].item(), 36.0)
        self.assertAlmostEqual(head[0, 0].item(), 54.0)
        self.assertEqual(old_head[0, 0].item(), 9.0)
        self.assertGreater(shift, 0.0)
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))

    def test_script_uses_baseline_recipe_and_only_one_candidate(self):
        output = subprocess.check_output(
            ['bash', 'scripts/9_29_imgr10_late_average_3090.sh', '--dry-run'], text=True)
        command = shlex.split(next(line.removeprefix('Command: ')
                                   for line in output.splitlines() if line.startswith('Command: ')))
        overrides = dict(command[i + 1].split('=', 1) for i, arg in enumerate(command) if arg == '--set')
        self.assertEqual(overrides['seed'], '[1993]')
        self.assertEqual(overrides['max_tasks'], '3')
        self.assertEqual(overrides['late_weight_average_epochs'], '5')
        self.assertEqual(overrides['dual_mask_anchor_reg_weight'], '2.5')
        self.assertEqual(overrides['ca_epochs'], '5')
        self.assertEqual(overrides['disable_fused_sdpa'], 'true')
        self.assertEqual(overrides['save_task_weights'], 'true')
        self.assertNotIn('data_path', overrides)


if __name__ == '__main__':
    unittest.main()
