import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from utils.old_model_distillation import frozen_teacher, old_output_loss
from models.attention import Attention_LoRA
from test.test_dual_mask_core import make_args


def term():
    tree = ast.parse(Path('methods/dlora.py').read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_old_model_distillation_term')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<distillation-term>', 'exec'), ns)
    return ns[method.name]


class OldModelDistillationTests(unittest.TestCase):
    def test_teacher_lifecycle_order(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == 'incremental_train')
        source = ast.unparse(method)
        self.assertLess(source.index('frozen_teacher(self._network)'), source.index('self._network.update_fc'))
        self.assertIn("self._cur_task > 0", source)
        release = source.index('self._old_teacher = None', source.index('self._train('))
        self.assertLess(release, source.index('self._compute_class_mean'))

    def test_disabled_path_no_forward(self):
        self.assertEqual(term()(SimpleNamespace(), {}, None), (None, {}))

    def test_real_merged_attention_can_be_copied_and_student_receives_gradient(self):
        module = Attention_LoRA(dim=4, num_heads=1, r=2, n_tasks=3)
        module._init_params(make_args())
        module.before_task(0)
        module.set_task_and_stage(0, 0)
        x = torch.randn(3, 5, 4)
        module(x, task=0).sum().backward()
        module.after_task(0)
        teacher = frozen_teacher(module)
        with torch.no_grad():
            expected = module(x, task=0)
            self.assertTrue(torch.allclose(teacher(x, task=0), expected))
        module.before_task(1)
        module.set_task_and_stage(1, 0)
        old_head = nn.Linear(4, 3, bias=False)
        teacher_features = teacher(x, task=0).mean(1).detach()
        teacher_logits = F.normalize(teacher_features, dim=1) @ F.normalize(old_head.weight.detach(), dim=1).T
        fake_teacher = SimpleNamespace(interface=lambda inputs: teacher_logits)
        learner = SimpleNamespace(_network=SimpleNamespace(classifier_pool=[old_head]),
                                  _old_teacher=fake_teacher, _cur_task=1, scale=20.,
                                  args={'old_model_distill_weight': 1., 'old_model_distill_temperature': 2.})
        with torch.no_grad():
            module.P_lora[1].B.weight.normal_(std=.1)
        loss, metrics = term()(learner, {'features': module(x, task=1).mean(1)}, x)
        loss.backward()
        self.assertGreater(module.P_lora[1].B.weight.grad.norm().item(), 0.)
        self.assertIsNone(old_head.weight.grad)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(not v.requires_grad for v in metrics.values()))

    def test_one_candidate_recipe(self):
        spec = json.loads(Path('scripts/sweeps/imgr10_old_model_distill_3090.json').read_text())
        self.assertEqual(len(spec['variants']), 1)
        common = spec['common_overrides']
        self.assertFalse(common['plora_train_a'])
        self.assertEqual(common['dual_mask_anchor_reg_weight'], 10)
        self.assertEqual(common['ca_epochs'], 5)
        self.assertNotIn('data_path', common)
        self.assertEqual(spec['variants'][0]['overrides'],
                         {'old_model_distill_weight': 1, 'old_model_distill_temperature': 2})
        output = subprocess.check_output(['bash', 'scripts/9_27_imgr10_old_model_distill_3090.sh', '--dry-run'], text=True)
        self.assertEqual(output.count('main.py --config'), 1)
        self.assertIn('--set old_model_distill_weight=1', output)

    def test_identical_outputs_zero_loss(self):
        x, w = torch.randn(5, 4), torch.randn(7, 4)
        logits = F.normalize(x, dim=1) @ F.normalize(w, dim=1).T
        self.assertAlmostEqual(old_output_loss(x, w, logits, 20., 2.).item(), 0., places=5)

    def test_gradient_only_reaches_student_features(self):
        x = torch.randn(5, 4, requires_grad=True)
        w = torch.randn(7, 4, requires_grad=True)
        teacher = torch.randn(5, 7, requires_grad=True)
        loss = old_output_loss(x, w, teacher, 20., 2.)
        loss.backward()
        self.assertGreater(x.grad.norm().item(), 0.)
        self.assertIsNone(w.grad)
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(loss))

    def test_teacher_independent_frozen_and_rng_preserved(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.5))
        state = torch.get_rng_state().clone()
        teacher = frozen_teacher(model)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        before = teacher[0].weight.clone()
        with torch.no_grad():
            model[0].weight.add_(1)
        self.assertTrue(torch.equal(before, teacher[0].weight))
