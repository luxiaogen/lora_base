"""旧B独立配方：固定初始化容量不再由NCM控制。"""
import json
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.attention import Attention_LoRA


ROOT = Path(__file__).resolve().parents[1]


class FixedBRecipeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1993)

    def test_task0_uses_configured_inactive_P_rank(self):
        # 若忽略配置或再次根据NCM调整秩，实际P形状会与这里不同。
        args = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
        args['task0_p_init_rank'] = 39
        layer = Attention_LoRA(64, num_heads=4, r=64, n_tasks=2)
        layer._init_params(args)
        layer.before_task(0)
        layer.set_task_and_stage(0, 0)
        self.assertEqual(tuple(layer.S_lora[0].A_weight.shape), (64, 64))
        self.assertEqual(tuple(layer.P_lora[0].A_weight.shape), (39, 64))
        self.assertTrue(layer.S_lora[0].A_weight.requires_grad)
        self.assertFalse(layer.P_lora[0].A_weight.requires_grad)
        self.assertFalse(layer.P_lora[0].B_weight.requires_grad)

    def test_incremental_P_rank_comes_from_rank_config(self):
        # 防止代码将增量P rank硬编码为64，而不是读取配置。
        args = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
        args['task0_p_init_rank'] = 39
        layer = Attention_LoRA(64, num_heads=4, r=16, n_tasks=2)
        layer._init_params(args)
        layer.before_task(1)
        layer.set_task_and_stage(1, 0)
        self.assertEqual(tuple(layer.P_lora[1].A_weight.shape), (16, 64))
        self.assertFalse(layer.P_lora[1].A_weight.requires_grad)
        self.assertTrue(layer.P_lora[1].B_weight.requires_grad)

    def test_SVD_uses_configured_energy_instead_of_a_hidden_constant(self):
        # 奇异值3/2/1的平方能量为9/4/1；0.90需两项，0.95需三项。
        weight = torch.zeros(24, 8)
        weight[:3, :3] = torch.diag(torch.tensor([3., 2., 1.]))
        for coverage, expected in ((.90, 2), (.95, 3)):
            with self.subTest(coverage=coverage):
                layer = Attention_LoRA(8, num_heads=2, r=4)
                layer._init_params({'task0_p_init_rank': 39, 'svd_energy_coverage': coverage})
                layer._svd_importance(weight)
                self.assertEqual(layer.last_svd_rank, expected)

    def test_epoch_log_keeps_classification_and_anchor_terms(self):
        # 若只打印总Loss，无法再核实Anchor是否实际加入。
        from methods.dlora import Learner

        class MiniNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = Attention_LoRA(64, num_heads=4, r=64, n_tasks=2)
                self.attention._init_params({'task0_p_init_rank': 39})
                self.head = nn.Linear(64, 20, bias=False)
                self.attention.before_task(0)
                self.attention.set_task_and_stage(0, 0)

            def forward(self, inputs):
                features = self.attention(inputs, 0).mean(1)
                logits = F.linear(F.normalize(features, dim=1), F.normalize(self.head.weight, dim=1))
                return {'logits': logits}

            def interface(self, inputs):
                return self(inputs)['logits']

        learner = object.__new__(Learner)
        learner._network, learner._device = MiniNetwork(), torch.device('cpu')
        learner._cur_task, learner._known_classes = 0, 0
        learner.scale, learner.margin, learner.run_epoch = 20., .1, 1
        learner.args = {'task0_anchor_weight': 2.5}
        inputs = torch.randn(6, 3, 64)
        loader = DataLoader(TensorDataset(torch.arange(6), inputs, torch.arange(6)), batch_size=3)
        optimizer = torch.optim.SGD([p for p in learner._network.parameters() if p.requires_grad], lr=.02)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        with self.assertLogs(level='INFO') as captured:
            learner.train_function(loader, loader, optimizer, scheduler)
        lines = [line for line in captured.output if 'Task 0, Epoch' in line]
        self.assertEqual(len(lines), 1)
        self.assertIn('CE ', lines[0])
        self.assertIn('anchor_reg ', lines[0])
        self.assertIn('anchor_reg_weighted ', lines[0])


if __name__ == '__main__':
    unittest.main()
