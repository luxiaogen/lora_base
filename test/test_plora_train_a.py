import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest

import torch

from models.attention import Attention_LoRA
from test.test_dual_mask_core import make_args
from test.test_plora_lr import grouping_method, legacy_groups


def module_for(enabled=False):
    module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=3)
    module._init_params(make_args(plora_train_a=enabled))
    return module


class PrivateATests(unittest.TestCase):
    def test_queue_changes_only_private_a_flag(self):
        root = Path(__file__).resolve().parents[1]
        output = subprocess.check_output(
            ['bash', 'scripts/9_27_imgr10_plora_train_a_3090.sh', '--dry-run'],
            cwd=root, text=True)
        commands = [shlex.split(line) for line in output.splitlines() if 'main.py --config' in line]
        self.assertEqual(len(commands), 2)
        settings = [dict(token.split('=', 1) for token in command if '=' in token) for command in commands]
        self.assertEqual(settings[0].pop('plora_train_a'), 'false')
        self.assertEqual(settings[1].pop('plora_train_a'), 'true')
        for run in settings:
            run.pop('prefix')
            self.assertNotIn('data_path', run)
            self.assertEqual(json.loads(run['seed']), [1993])
            self.assertEqual(run['dual_mask_anchor_reg_weight'], '10')
            self.assertEqual(run['ca_epochs'], '5')
            self.assertEqual(run['max_tasks'], '10')
            self.assertEqual(run['disable_fused_sdpa'], 'true')
        self.assertEqual(settings[0], settings[1])

    def test_task0_identical_and_incremental_only_private_a_changes(self):
        modules = []
        for enabled in (False, True):
            torch.manual_seed(1993)
            m = module_for(enabled)
            m.before_task(0)
            m.set_task_and_stage(0, 0)
            self.assertTrue(m.S_lora[0].A.weight.requires_grad)
            self.assertFalse(m.P_lora[0].A.weight.requires_grad)
            m.before_task(1)
            m.set_task_and_stage(1, 0)
            self.assertFalse(m.S_lora[1].A.weight.requires_grad)
            self.assertEqual(m.P_lora[1].A.weight.requires_grad, enabled)
            self.assertFalse(m.S_lora[0].A.weight.requires_grad)
            self.assertTrue(m.P_lora[1].B.weight.requires_grad)
            modules.append(m)
        for name, value in modules[0].state_dict().items():
            self.assertTrue(torch.equal(value, modules[1].state_dict()[name]), name)
        counts = [sum(p.numel() for p in m.parameters() if p.requires_grad) for m in modules]
        self.assertEqual(counts[1] - counts[0], modules[1].P_lora[1].A.weight.numel())

    def test_masked_forward_updates_a_and_merge_preserves_output(self):
        torch.manual_seed(7)
        m = module_for(True)
        m.before_task(1)
        m.set_task_and_stage(1, 0)
        m.eval()
        x = torch.randn(3, 5, 4)
        a0 = m.P_lora[1].A.weight.detach().clone()
        sa0 = m.S_lora[1].A.weight.detach().clone()
        legacy = legacy_groups(m)
        groups = grouping_method()(SimpleNamespace(_network=m, _cur_task=1, args={}),
                                   legacy[0]['params'], legacy[1]['params'], .1, 0.)
        self.assertIn(id(m.P_lora[1].A.weight), [id(p) for p in groups[0]['params']])
        optimizer = torch.optim.SGD(groups)
        for _ in range(3):
            optimizer.zero_grad()
            m(x, task=1).square().mean().backward()
            optimizer.step()
        self.assertFalse(torch.equal(a0, m.P_lora[1].A.weight))
        self.assertTrue(torch.equal(sa0, m.S_lora[1].A.weight))
        before = m(x, task=1).detach()
        m.after_task(1)
        self.assertTrue(torch.allclose(before, m(x, task=1), atol=1e-5, rtol=1e-5))
        self.assertIsNone(m.P_lora[1])

    def test_disabled_private_branch_remains_frozen(self):
        m = module_for(True)
        m.use_plora = False
        m.before_task(1)
        m.set_task_and_stage(1, 0)
        self.assertFalse(m.P_lora[1].A.weight.requires_grad)
