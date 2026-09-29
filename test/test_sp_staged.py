import ast
import logging
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


def phase_hook():
    tree = ast.parse(Path('methods/dlora.py').read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_set_branch_training_phase')
    namespace = {'logging': logging}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<phase-hook>', 'exec'), namespace)
    return namespace[method.name]


class StagedTrainingTests(unittest.TestCase):
    def fixture(self, task=1, split=5):
        from test.test_global_conflict_budget import GlobalBudgetSelectionTests
        attention = GlobalBudgetSelectionTests._make_attention('layer')
        attention.before_task(1)
        attention.set_task_and_stage(1, 0)
        for unit in (attention.S_lora[1], attention.P_lora[1]):
            unit.A_weight.requires_grad_(False)
        learner = SimpleNamespace(args={'sp_staged_s_epochs': split}, _cur_task=task,
                                  _iter_lora_modules=lambda: iter([attention]))
        return learner, attention

    def test_disabled_and_task0_are_noops(self):
        hook = phase_hook()
        for task, split in ((0, 5), (1, 0)):
            learner, attn = self.fixture(task, split)
            flags = [p.requires_grad for p in attn.parameters()]
            rng = torch.get_rng_state().clone()
            hook(learner, 0)
            self.assertEqual(flags, [p.requires_grad for p in attn.parameters()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_real_gates_freeze_s_and_optimizer_keeps_both_branches(self):
        hook = phase_hook()
        learner, attn = self.fixture()
        s, p = attn.S_lora[1], attn.P_lora[1]
        head = torch.nn.Linear(12, 2)
        optimizer = torch.optim.SGD([s.B_weight, p.B_weight, *head.parameters()],
                                    lr=.02, momentum=.9, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        x = torch.randn(6, 4)
        frozen_s = None
        frozen_output = None
        for epoch in range(20):
            hook(learner, epoch)
            before_s = s.B_weight.detach().clone()
            before_p = p.B_weight.detach().clone()
            before_head = head.weight.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(attn._contrib_from_units(x, 1)),
                                                      torch.tensor([0, 1, 0, 1, 0, 1]))
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.assertFalse(torch.equal(before_head, head.weight))
            self.assertFalse(s.A_weight.requires_grad or p.A_weight.requires_grad)
            if epoch < 5:
                self.assertTrue(torch.equal(before_p, p.B_weight))
                self.assertEqual(torch.count_nonzero(p.B_weight).item(), 0)
                self.assertFalse(torch.equal(before_s, s.B_weight))
                frozen_s = s.B_weight.detach().clone()
                frozen_output = attn._masked_unit_forward(x, s, isolated=False).detach().clone()
            else:
                self.assertIsNone(s.B_weight.grad)
                self.assertTrue(torch.equal(frozen_s, s.B_weight))
                self.assertTrue(torch.equal(frozen_output, attn._masked_unit_forward(x, s, isolated=False)))
                self.assertFalse(torch.equal(before_p, p.B_weight))
        self.assertEqual(scheduler.last_epoch, 20)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0)

    def test_hook_is_in_real_epoch_loop(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        train = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'train_function')
        calls = [n for n in ast.walk(train) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == '_set_branch_training_phase']
        self.assertEqual(len(calls), 1)

    def test_queue_matches_single_candidate(self):
        import shlex
        spec = json.loads(Path('scripts/sweeps/imgr10_sp_staged_3090.json').read_text())
        output = subprocess.check_output(['bash', 'scripts/9_29_imgr10_sp_staged_3090.sh', '--dry-run'], text=True)
        commands = [shlex.split(line[len('Command: '):]) for line in output.splitlines()
                    if line.startswith('Command: ')]
        self.assertEqual(len(commands), 1)
        command = commands[0]
        actual = dict(command[i + 1].split('=', 1) for i, arg in enumerate(command) if arg == '--set')
        expected = dict(spec['common_overrides'], **spec['variants'][0]['overrides'])
        for key, value in expected.items():
            self.assertEqual(actual[key], str(value).lower() if isinstance(value, bool)
                             else json.dumps(value, separators=(',', ':')) if isinstance(value, list) else str(value), key)
        self.assertEqual(actual['sp_staged_s_epochs'], '5')
        self.assertEqual(actual['save_task_weights'], 'false')
        self.assertNotIn('data_path', actual)


if __name__ == '__main__':
    unittest.main()
