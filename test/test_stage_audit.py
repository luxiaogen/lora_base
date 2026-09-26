import ast
import copy
import json
import random
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.stage_audit import StageAudit, collect_logits, stage_metrics
from scripts.summarize_stage_audit import parse


class Network(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4).to(device)
        self.dropout = torch.nn.Dropout(.5)

    def interface(self, inputs):
        return self.linear(self.dropout(inputs))


class StageAuditTests(unittest.TestCase):
    def test_metrics_and_transitions(self):
        targets = torch.tensor([0, 1, 2, 3])
        before = torch.eye(4)
        before[0] = torch.tensor([0., 2., 0., 0.])
        after = torch.eye(4)
        after[2] = torch.tensor([2., 0., 0., 0.])
        result = stage_metrics(after, targets, 2, before)
        self.assertEqual(result['old']['corrected'], 1)
        self.assertEqual(result['new']['broken'], 1)
        self.assertEqual(result['total']['accuracy'], 75)
        self.assertIsNone(stage_metrics(after, targets, 0)['old']['accuracy'])

    def test_readonly_rng_modes_weights_and_training_trajectory(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(13)
        network = Network(device)
        initial = copy.deepcopy(network.state_dict())
        data = TensorDataset(torch.arange(8), torch.randn(8, 3), torch.arange(8) % 4)
        loader = DataLoader(data, batch_size=4, generator=torch.Generator().manual_seed(23))
        states = []
        for enabled in (False, True):
            network.load_state_dict(initial)
            network.train()
            torch.manual_seed(51)
            optimizer = torch.optim.SGD(network.parameters(), lr=.01, momentum=.9)
            for _ in range(2):
                if enabled:
                    generator_state = loader.generator.get_state().clone()
                    py_state, np_state = random.getstate(), np.random.get_state()
                    collect_logits(network, loader, device)
                    self.assertTrue(network.training and network.dropout.training)
                    self.assertTrue(torch.equal(generator_state, loader.generator.get_state()))
                    self.assertEqual(py_state, random.getstate())
                    self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
                optimizer.zero_grad()
                network.interface(data.tensors[1].to(device)).square().mean().backward()
                optimizer.step()
            states.append(copy.deepcopy(network.state_dict()))
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]), name)

    def test_record_and_parser(self):
        network = Network('cpu')
        loader = DataLoader(TensorDataset(torch.arange(4), torch.randn(4, 3), torch.arange(4)), batch_size=2)
        audit = StageAudit(0, 0)
        with self.assertLogs(level='INFO'):
            rows = [audit.record(stage, network, loader, 'cpu')
                    for stage in ('pre_merge', 'post_merge', 'post_ca')]
        self.assertEqual(rows[1]['metrics']['total']['logit_max_abs_change'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.log'
            path.write_text('\n'.join('StageAudit ' + json.dumps(r) for r in rows) + '\n=> Last Accuracy: 25')
            self.assertEqual(len(parse(path, 1)), 9)
            rows[1]['sample_sha256'] = 'different'
            path.write_text('\n'.join('StageAudit ' + json.dumps(r) for r in rows) + '\n=> Last Accuracy: 25')
            with self.assertRaises(AssertionError):
                parse(path, 1)

    def test_hook_order_and_specs(self):
        source = Path('methods/dlora.py').read_text()
        ast.parse(source)
        train = source[source.index('    def _train('):source.index('    def train_function(')]
        self.assertLess(train.index("record('pre_merge'"), train.index('module.after_task('))
        self.assertLess(train.index('module.after_task('), train.index("record('post_merge'"))
        incremental = source[source.index('    def incremental_train('):source.index('    def _lora_optimizer_groups(')]
        self.assertLess(incremental.index('_stage2_compact_classifier('), incremental.index("record('post_ca'"))
        for gpu, weights in [('3090', [10]), ('5090', [10, 5])]:
            spec = json.loads(Path(f'scripts/sweeps/imgr10_stage_audit_{gpu}.json').read_text())
            self.assertEqual(spec['seeds'], [1993])
            self.assertEqual(spec['common_overrides']['ca_epochs'], 5)
            self.assertTrue(spec['common_overrides']['stage_audit'])
            self.assertNotIn('data_path', spec['common_overrides'])
            self.assertEqual([v['overrides'] for v in spec['variants']],
                             [{'dual_mask_anchor_reg_weight': w} for w in weights])
            script = f'scripts/9_26_imgr10_stage_audit_{gpu}.sh'
            subprocess.run(['bash', '-n', script], check=True)
            dry = subprocess.check_output(['bash', script, '--dry-run'], text=True)
            commands = dry.split('    python main.py')[1:]
            self.assertEqual(len(commands), len(weights))
            for command, weight in zip(commands, weights):
                tokens = shlex.split(command.replace('\\\n', ' '))
                settings = dict(tokens[i+1].split('=', 1) for i, token in enumerate(tokens) if token == '--set')
                self.assertEqual(settings['p_step_direction'], 'off')
                self.assertEqual(settings['dual_mask_anchor_reg_weight'], str(weight))
                self.assertEqual(settings['ca_epochs'], '5')
                self.assertNotIn('data_path', settings)


if __name__ == '__main__':
    unittest.main()
