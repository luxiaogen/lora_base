from contextlib import contextmanager
import ast
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.two_expert_oracle import (
    collect_experts, expert_report, expert_rows, rank_auc, save_report,
)
from scripts.analyze_two_expert_oracle import summarize


class TwoExpertTests(unittest.TestCase):
    def example(self):
        # Base correct: 0,2. NCM rescues: 1. NCM breaks: 2. Both wrong: 3.
        base = torch.tensor([[.9, .1, 0, 0], [.1, .9, 0, 0],
                             [0, 0, .6, .4], [.8, .1, .1, 0]])
        anchor = torch.tensor([[.8, .2, 0, 0], [.9, .1, 0, 0],
                               [0, 0, .1, .9], [.1, .8, .1, 0]])
        return base, anchor, torch.tensor([0, 0, 2, 3])

    def test_upper_bound_and_legal_switch_count_harmed_and_neutral(self):
        base, anchor, targets = self.example()
        rows = expert_rows(base, anchor, targets, torch.arange(4), torch.arange(4), 2)
        report = expert_report(rows, known_classes=2)
        total = report['groups']['total']
        self.assertEqual((total['base_accuracy'], total['anchor_accuracy'],
                          total['oracle_accuracy']), (50., 50., 75.))
        self.assertEqual((total['rescued'], total['harmed'], total['neutral_disagreement']), (1, 1, 1))
        rule = total['signals']['margin_advantage']
        self.assertEqual((rule['selected_rescued'], rule['selected_harmed'],
                          rule['selected_neutral']), (0, 1, 0))
        self.assertEqual(rule['fixed_rule_accuracy'], 25.)
        self.assertEqual(rule['net_gain_pp'], -25.)
        self.assertEqual(report['groups']['old']['oracle_gain_pp'], 50.)
        self.assertEqual(report['groups']['new']['oracle_gain_pp'], 0.)

    def test_signals_and_decisions_do_not_depend_on_true_labels(self):
        base, anchor, targets = self.example()
        args = (torch.arange(4), torch.arange(4), 2)
        rows = expert_rows(base, anchor, targets, *args)
        changed = expert_rows(base, anchor, targets.flip(0), *args)
        for key in ('base_prediction', 'anchor_prediction', 'margin_advantage',
                    'proposal_advantage', 'base_uncertainty'):
            self.assertTrue(torch.equal(rows[key], changed[key]), key)

    def test_task_id_ceiling_is_report_only_and_not_a_switch_signal(self):
        base, anchor, targets = self.example()
        rows = expert_rows(base, anchor, targets, torch.arange(4), torch.arange(4), 2)
        self.assertEqual(rows['task_id_prediction'].tolist(), [0, 1, 2, 2])
        self.assertNotIn('task_id_prediction', expert_report(rows, 2)['groups']['total']['signals'])

    def test_auc_ties_reverse_and_empty(self):
        self.assertEqual(rank_auc(torch.tensor([0., 1.]), torch.tensor([False, True])), 1.)
        self.assertEqual(rank_auc(torch.tensor([1., 0.]), torch.tensor([False, True])), 0.)
        self.assertEqual(rank_auc(torch.ones(2), torch.tensor([False, True])), .5)
        self.assertIsNone(rank_auc(torch.ones(2), torch.ones(2, dtype=torch.bool)))
        self.assertIsNone(rank_auc(torch.empty(0), torch.empty(0, dtype=torch.bool)))

    def test_collection_restores_weights_modes_gradients_rng_and_loader(self):
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.eye(4))
                self.child = nn.Dropout()

            def interface(self, inputs):
                random.random()
                np.random.rand()
                torch.rand(1)
                return inputs @ self.weight

            def extract_vector(self, inputs):
                return inputs @ self.weight

            @contextmanager
            def anchor(self):
                before = self.weight.detach().clone()
                try:
                    self.weight.copy_(torch.eye(4).flip(1))
                    yield
                finally:
                    self.weight.copy_(before)

        network = Network().train()
        network.child.eval()
        network.weight.grad = torch.ones_like(network.weight)
        base, _, targets = self.example()
        loader = DataLoader(TensorDataset(torch.arange(4), base, targets), batch_size=2,
                            generator=torch.Generator().manual_seed(1993))
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 loader.generator.get_state())
        rows = collect_experts(network, loader, 'cpu', network.anchor,
                               torch.eye(4), torch.arange(4), 2)
        self.assertEqual(rows['index'].tolist(), list(range(4)))
        self.assertTrue(torch.equal(network.weight, torch.eye(4)))
        self.assertTrue(torch.equal(network.weight.grad, torch.ones(4, 4)))
        self.assertTrue(network.training)
        self.assertFalse(network.child.training)
        self.assertEqual(random.getstate(), state[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], state[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), state[2]))
        self.assertTrue(torch.equal(loader.generator.get_state(), state[3]))

    def test_save_only_small_sample_scalars_and_metadata(self):
        base, anchor, targets = self.example()
        rows = expert_rows(base, anchor, targets, torch.arange(4), torch.arange(4), 2)
        with tempfile.TemporaryDirectory() as folder:
            report = save_report(Path(folder), 1, 'test_report_only', rows, 2)
            saved = json.loads((Path(folder) / 'task_01_test_report_only.json').read_text())
            self.assertEqual(saved, report)
            self.assertFalse(saved['used_for_training'])
            self.assertFalse(saved['old_train_images_accessed'])
            self.assertTrue(saved['oracle_uses_true_class_labels'])
            self.assertEqual(saved['class_scope'], 'all_seen')
            csv = (Path(folder) / 'task_01_test_report_only.csv').read_text()
            self.assertEqual(len(csv.splitlines()), 5)
            self.assertNotIn('feature', csv.splitlines()[0])
            self.assertNotIn('logits', csv.splitlines()[0])

    def test_actual_attention_hook_on_off_matches_rng_weights_and_logits(self):
        from test.test_global_conflict_budget import GlobalBudgetSelectionTests
        from torch.nn import functional as F

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = GlobalBudgetSelectionTests._make_attention('layer')
                self.attention.before_task(1)
                self.attention.set_task_and_stage(1, 0)
                with torch.no_grad():
                    self.attention.S_lora[1].B_weight.fill_(.03)
                    self.attention.P_lora[1].B_weight.fill_(.02)
                self.attention.after_task(1)
                self.head = nn.Linear(4, 4, bias=False)

            def extract_vector(self, inputs):
                return self.attention(inputs[:, None], task=1)[:, 0]

            def interface(self, inputs):
                return F.linear(F.normalize(self.extract_vector(inputs), dim=1),
                                F.normalize(self.head.weight, dim=1))

        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'methods/dlora.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == 'eval_w0_task')
        namespace = dict(torch=torch, F=F, DataLoader=DataLoader)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<eval-hook>', 'exec'), namespace)
        inputs = torch.randn(4, 4)
        targets = torch.arange(4)
        original = Network().eval()
        rng = torch.get_rng_state().clone()
        ends = []
        with tempfile.TemporaryDirectory() as folder:
            for enabled in (False, True):
                net = copy.deepcopy(original)
                torch.set_rng_state(rng)
                loader = DataLoader(TensorDataset(torch.arange(4), inputs, targets), batch_size=2)
                train = DataLoader(TensorDataset(torch.arange(2), inputs[2:], targets[2:]), batch_size=2)
                learner = SimpleNamespace(args=dict(two_expert_oracle_diagnostic=enabled,
                                          two_expert_oracle_dir=folder), _network=net,
                                          _device='cpu', _w0_class_means={i: torch.eye(4)[i] for i in range(4)},
                                          _w0_accuracy_curve=[], _cur_task=1, _known_classes=2,
                                          class_num=2, batch_size=2, w0_loader=train, test_loader=loader,
                                          _pretrained_anchor_context=net.attention.use_pretrained_anchor)
                accuracy = namespace['eval_w0_task'](learner)
                ends.append((accuracy, torch.get_rng_state().clone(), net.interface(inputs)))
                self.assertEqual(learner._w0_accuracy_curve, [accuracy])
                self.assertFalse(net.attention.pretrained_anchor_mode)
                for key, value in original.state_dict().items():
                    self.assertTrue(torch.equal(net.state_dict()[key], value), key)
            self.assertEqual(ends[0][0], ends[1][0])
            self.assertTrue(torch.equal(ends[0][1], ends[1][1]))
            self.assertTrue(torch.equal(ends[0][2], ends[1][2]))
            train_report = json.loads((Path(folder) / 'task_01_current_train_seen_probe.json').read_text())
            self.assertEqual(train_report['groups']['old']['n'], 0)

    def test_launcher_single_full_run_and_no_machine_path_overrides(self):
        import subprocess
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root / 'scripts/sweeps/imgr10_two_expert_oracle_3090.json').read_text())
        config = spec['common_overrides']
        self.assertEqual(spec['seeds'], [1993])
        self.assertEqual(len(spec['variants']), 1)
        self.assertEqual((config['init_epoch'], config['epochs'], config['ca_epochs']), (20, 20, 5))
        self.assertEqual(config['dual_mask_anchor_reg_weight'], 2.5)
        self.assertEqual(config['branch_choice_mode'], 'off')
        self.assertFalse(config['p_old_gradient_oracle'])
        self.assertFalse(config['save_task_weights'])
        self.assertTrue(config['disable_fused_sdpa'])
        for key in ('data_path', 'device', 'pretrained_path'):
            self.assertNotIn(key, config)
        output = subprocess.check_output(['bash', 'scripts/10_01_imgr10_two_expert_oracle_3090.sh',
                                          '--dry-run'], cwd=root, text=True)
        self.assertEqual(output.count('Command: '), 2)
        self.assertEqual(output.count('--set two_expert_oracle_diagnostic=true'), 2)
        self.assertNotIn('--set data_path=', output)

    def test_summary_does_not_call_short_smoke_a_complete_t10(self):
        base, anchor, targets = self.example()
        rows = expert_rows(base, anchor, targets, torch.arange(4), torch.arange(4), 2)
        with tempfile.TemporaryDirectory() as folder:
            save_report(Path(folder), 0, 'test_report_only', rows, 0)
            summary = summarize(folder)
            self.assertFalse(summary['full_t10_completed'])
            self.assertEqual(summary['average']['oracle_accuracy'], 75.)

    def test_incomplete_reports_do_not_pass_smoke_gate(self):
        from contextlib import redirect_stdout
        import io
        base, anchor, targets = self.example()
        rows = expert_rows(base, anchor, targets, torch.arange(4), torch.arange(4), 2)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            self.assertIsNone(summarize(folder, 2))
            save_report(Path(folder), 0, 'test_report_only', rows, 0)
            self.assertFalse(summarize(folder, 2)['expected_reports_complete'])
            save_report(Path(folder), 1, 'test_report_only', rows, 2)
            self.assertFalse(summarize(folder, 2)['expected_reports_complete'])
            for task in range(2):
                save_report(Path(folder), task, 'current_train_seen_probe', rows, task * 2)
            self.assertTrue(summarize(folder, 2)['expected_reports_complete'])
            (Path(folder) / 'task_01_test_report_only.csv').unlink()
            self.assertFalse(summarize(folder, 2)['expected_reports_complete'])

    def test_misspelled_mode_cannot_start_training(self):
        import subprocess
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(['bash', 'scripts/10_01_imgr10_two_expert_oracle_3090.sh', '--dryrun'],
                                cwd=root, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('Command:', result.stdout)


if __name__ == '__main__':
    unittest.main()
