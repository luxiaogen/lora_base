import ast
import json
import copy
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from utils.old_competition import old_competition_loss
from utils.stage_audit import stage_metrics


class OldCompetitionTests(unittest.TestCase):
    def test_old_head_frozen_but_features_receive_gradient(self):
        features = torch.tensor([[1., .3]], requires_grad=True)
        old = torch.nn.Parameter(torch.tensor([[1., 0.]]))
        current = torch.tensor([[.2, .1]], requires_grad=True)
        loss, active = old_competition_loss(features, current, torch.tensor([0]), old, 20.)
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertGreater(float(features.grad.norm()), 0)
        self.assertEqual(float(current.grad[0, 0]), -20.)
        self.assertEqual(float(active), 1.)

    def test_no_penalty_if_true_new_class_beats_old(self):
        features = torch.tensor([[0., 1.]], requires_grad=True)
        logits = torch.tensor([[.8, .2]], requires_grad=True)
        loss, active = old_competition_loss(features, logits, torch.tensor([0]), torch.tensor([[1., 0.]]), 20.)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(float(active), 0)

    def test_loss_matches_all_seen_cosine_scores(self):
        torch.manual_seed(2)
        features, old, new = torch.randn(4, 6), torch.randn(3, 6), torch.randn(2, 6)
        logits = F.linear(F.normalize(features, dim=1), F.normalize(new, dim=1))
        targets = torch.tensor([0, 1, 0, 1])
        loss, _ = old_competition_loss(features, logits, targets, old, 20.)
        all_logits = F.linear(F.normalize(features, dim=1), F.normalize(torch.cat([old, new]), dim=1))
        expected = (all_logits[:, :3].max(1).values - all_logits.gather(1, (targets+3)[:, None]).squeeze(1)).relu().mean()*20
        torch.testing.assert_close(loss, expected)

    def test_task0_and_zero_weight_skip_extra_work(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_old_competition_term')
        ns = {'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<hook>', 'exec'), ns)
        for task, weight in [(0, .1), (1, 0)]:
            learner = SimpleNamespace(_cur_task=task, args={'old_competition_weight': weight})
            self.assertEqual(ns['_old_competition_term'](learner, {}, None), (None, {}))

    def test_training_term_reuses_features_and_keeps_old_head_unchanged(self):
        tree = ast.parse(Path('methods/dlora.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_old_competition_term')
        ns = {'torch': torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<hook>', 'exec'), ns)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(17)
        encoder = torch.nn.Linear(3, 4).to(device)
        heads = torch.nn.ModuleList([torch.nn.Linear(4, 2, bias=False).to(device) for _ in range(2)])
        initial_encoder, initial_heads = copy.deepcopy(encoder.state_dict()), copy.deepcopy(heads.state_dict())
        inputs = torch.randn(8, 3, device=device)
        targets = torch.arange(8, device=device) % 2
        final = []
        for weight in (0., .1):
            encoder.load_state_dict(initial_encoder)
            heads.load_state_dict(initial_heads)
            heads[0].requires_grad_(False)
            opt = torch.optim.SGD(list(encoder.parameters()) + list(heads[1].parameters()), lr=.02)
            features = encoder(inputs)
            logits = F.linear(F.normalize(features, dim=1), F.normalize(heads[1].weight, dim=1))
            learner = SimpleNamespace(_cur_task=1, _network=SimpleNamespace(classifier_pool=heads),
                                      args={'old_competition_weight': weight, 'scale': 20})
            extra, metrics = ns['_old_competition_term'](learner, {'features': features, 'logits': logits}, targets)
            loss = F.cross_entropy(logits * 20, targets)
            if extra is not None:
                self.assertGreater(float(metrics['old_competition_weighted']), 0)
                loss = loss + extra
            opt.zero_grad(); loss.backward(); opt.step()
            self.assertIsNone(heads[0].weight.grad)
            self.assertTrue(torch.equal(heads[0].weight, initial_heads['0.weight']))
            final.append(encoder.weight.detach().clone())
        self.assertFalse(torch.equal(final[0], final[1]))

    def test_error_partition_and_oracle_recovery(self):
        targets = torch.tensor([0, 1, 2, 3, 2])
        logits = torch.tensor([[0., 0., 2., 0.], [2., 0., 0., 0.],
                               [2., 0., 1., 0.], [0., 0., 2., 1.], [0., 0., 2., 1.]])
        metrics = stage_metrics(logits, targets, 2)
        self.assertEqual(metrics['old']['cross_partition_errors'], 1)
        self.assertEqual(metrics['old']['within_partition_errors'], 1)
        self.assertEqual(metrics['new']['cross_partition_errors'], 1)
        self.assertEqual(metrics['new']['within_partition_errors'], 1)
        self.assertEqual(metrics['new']['partition_oracle_recovered'], 1)
        self.assertEqual(metrics['total']['cross_partition_errors'], 2)

    def test_paired_specs_only_loss_weight_changes(self):
        for gpu in ('3090', '5090'):
            spec = json.loads(Path(f'scripts/sweeps/imgr10_old_competition_{gpu}.json').read_text())
            self.assertEqual(spec['seeds'], [1993])
            self.assertEqual(spec['common_overrides']['dual_mask_anchor_reg_weight'], 5)
            self.assertNotIn('data_path', spec['common_overrides'])
            self.assertEqual([v['overrides'] for v in spec['variants']],
                             [{'old_competition_weight': 0}, {'old_competition_weight': .1}])
            script = f'scripts/9_26_imgr10_old_competition_{gpu}.sh'
            subprocess.run(['bash', '-n', script], check=True)
            commands = subprocess.check_output(['bash', script, '--dry-run'], text=True).split('    python main.py')[1:]
            self.assertEqual(len(commands), 2)
            for command, weight in zip(commands, ('0', '0.1')):
                tokens = shlex.split(command.replace('\\\n', ' '))
                settings = dict(tokens[i+1].split('=', 1) for i, token in enumerate(tokens) if token == '--set')
                self.assertEqual(settings['old_competition_weight'], weight)
                self.assertEqual(settings['dual_mask_anchor_reg_weight'], '5')
                self.assertEqual(settings['p_step_direction'], 'off')
                self.assertNotIn('data_path', settings)


if __name__ == '__main__':
    unittest.main()
