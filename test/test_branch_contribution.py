import copy
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from test import test_private_rank as fixtures


class BranchContributionTests(unittest.TestCase):
    def test_switch_changes_only_current_branch_and_restores_default(self):
        torch.manual_seed(19)
        module = fixtures.PrivateRankTests().make_module()
        module.before_task(1)
        module.set_task_and_stage(1, 0)
        with torch.no_grad():
            module.S_lora[1].B.weight.normal_(std=.1)
            module.P_lora[1].B.weight.normal_(std=.1)
        module.eval()
        inputs = torch.randn(2, 3, 4)
        with torch.no_grad():
            baseline = module._contrib_from_units(inputs, 1)
            module._diagnostic_branch_mode = "s_only"
            shared = module._contrib_from_units(inputs, 1)
            module._diagnostic_branch_mode = "p_only"
            private = module._contrib_from_units(inputs, 1)
            del module._diagnostic_branch_mode
            restored = module._contrib_from_units(inputs, 1)
        self.assertTrue(torch.allclose(baseline, shared + private, atol=1e-6))
        self.assertTrue(torch.equal(baseline, restored))
        self.assertGreater(shared.norm().item(), 0)
        self.assertGreater(private.norm().item(), 0)

    def test_real_learner_diagnostic_is_read_only_and_training_continues(self):
        from methods.dlora import Learner
        from models.network import ViT

        outcomes = []
        for enabled in (False, True):
            torch.manual_seed(19)
            args = json.loads(Path('exps/dlora/imgr10.json').read_text())
            args.update(device=[torch.device('cpu')], embd_dim=8, init_cls=2, increment=2,
                        total_sessions=2, rank=2, num_heads=2, init_epoch=1, epochs=1,
                        num_workers=0, dual_mask_svd_rank=2,
                        dual_mask_competence_adaptive=False,
                        dual_mask_conflict_reg_enabled=False,
                        dual_mask_branch_contribution_diagnostic=enabled)
            encoder = ViT(img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2, rank=2, n_tasks=2)
            with patch('models.network._create_vision_transformer', return_value=encoder):
                learner = Learner(args)
            module = next(learner._iter_lora_modules())
            images = torch.randn(4, 3, 8, 8)
            for task in range(2):
                learner._cur_task = task
                learner._known_classes, learner._total_classes = task * 2, (task + 1) * 2
                learner._network.numtask = task + 1
                train_data = TensorDataset(torch.arange(2), images[2 * task:2 * task + 2], torch.tensor([0, 1]) + task * 2)
                test_data = TensorDataset(torch.arange(2 * task + 2), images[:2 * task + 2],
                                          torch.arange(2 * task + 2))
                learner._train(DataLoader(train_data, batch_size=2), DataLoader(test_data, batch_size=2))
                self.assertIsNone(module.S_lora[task])
                self.assertIsNone(module.P_lora[task])
            if enabled:
                scores = learner._last_branch_contribution
                self.assertEqual(set(scores), {'both', 's_only', 'p_only'})
                for branch_scores in scores.values():
                    self.assertEqual(set(branch_scores), {'total', 'old', 'new', 'task_prediction'})
                    self.assertTrue(all(np.isfinite(value) for value in branch_scores.values()))
            else:
                self.assertFalse(hasattr(learner, '_last_branch_contribution'))
            self.assertFalse(hasattr(module, '_diagnostic_branch_mode'))
            outcomes.append((copy.deepcopy(learner._network.state_dict()), torch.get_rng_state()))
        for name, value in outcomes[0][0].items():
            self.assertTrue(torch.equal(value, outcomes[1][0][name]), name)
        self.assertTrue(torch.equal(outcomes[0][1], outcomes[1][1]))

    def test_diagnostic_restores_weights_modes_and_rng(self):
        from methods.dlora import Learner
        from models.network import ViT

        torch.manual_seed(23)
        args = json.loads(Path('exps/dlora/imgr10.json').read_text())
        args.update(device=[torch.device('cpu')], embd_dim=8, init_cls=2, increment=2,
                    total_sessions=2, rank=2, num_heads=2, num_workers=0,
                    dual_mask_svd_rank=2, dual_mask_competence_adaptive=False)
        encoder = ViT(img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2, rank=2, n_tasks=2)
        with patch('models.network._create_vision_transformer', return_value=encoder):
            learner = Learner(args)
        module = next(learner._iter_lora_modules())
        learner._cur_task = 1
        learner._known_classes, learner._total_classes = 2, 4
        learner._network.numtask = 2
        module.before_task(1)
        module.set_task_and_stage(1, 0)
        with torch.no_grad():
            module.S_lora[1].B.weight.normal_(std=.1)
            module.P_lora[1].B.weight.normal_(std=.1)
        data = TensorDataset(torch.arange(4), torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 3]))
        loader = DataLoader(data, batch_size=4, num_workers=0)
        learner._network.train()
        parameters_before = copy.deepcopy(learner._network.state_dict())
        modes_before = [layer.training for layer in learner._network.modules()]
        python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        learner._measure_branch_contribution(loader, [module])
        for name, value in parameters_before.items():
            self.assertTrue(torch.equal(value, learner._network.state_dict()[name]), name)
        self.assertEqual(modes_before, [layer.training for layer in learner._network.modules()])
        self.assertEqual(python_before, random.getstate())
        self.assertEqual(numpy_before[0], np.random.get_state()[0])
        self.assertTrue(np.array_equal(numpy_before[1], np.random.get_state()[1]))
        self.assertEqual(numpy_before[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertFalse(hasattr(module, '_diagnostic_branch_mode'))


if __name__ == '__main__':
    unittest.main()
