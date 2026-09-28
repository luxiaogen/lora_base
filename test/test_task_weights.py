import ast
import copy
import json
from pathlib import Path
import random
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from models.attention import Attention_LoRA
from utils.task_weights import checkpoint_directory, save_task_weights, load_task_weights

ROOT = Path(__file__).resolve().parents[1]


class TinyMergedNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention_LoRA(8, num_heads=2, r=2, n_tasks=3)
        self.classifier_pool = torch.nn.ModuleList([torch.nn.Linear(8, 2) for _ in range(3)])
        self.numtask = 2

    def forward(self, x):
        features = self.attention(x, self.numtask - 1).mean(1)
        return torch.cat([head(features) for head in self.classifier_pool[:self.numtask]], dim=1)


class TaskWeightsTests(unittest.TestCase):
    def test_saving_does_not_change_training_trajectory(self):
        initial = TinyMergedNetwork()
        trajectories = []
        with tempfile.TemporaryDirectory() as tmp:
            for enabled in (False, True):
                net = copy.deepcopy(initial).train()
                opt = torch.optim.SGD(net.parameters(), lr=.01, momentum=.9)
                torch.manual_seed(19)
                learner = SimpleNamespace(_network=net, _cur_task=0, _known_classes=2, _total_classes=2)
                manager = SimpleNamespace(_class_order=[0, 1, 2, 3], _increments=[2, 2])
                for task in range(2):
                    opt.zero_grad()
                    net(torch.randn(2, 3, 8)).square().mean().backward()
                    opt.step()
                    learner._cur_task = task
                    if enabled:
                        save_task_weights(tmp, learner, manager, {}, [{'task': task, 'top1': 80.}])
                trajectories.append((copy.deepcopy(net.state_dict()), torch.get_rng_state().clone()))
            self.assertEqual(len(list(Path(tmp).glob('task_*.pt'))), 2)
        for name, tensor in trajectories[0][0].items():
            self.assertTrue(torch.equal(tensor, trajectories[1][0][name]), name)
        self.assertTrue(torch.equal(trajectories[0][1], trajectories[1][1]))

    def test_merged_attention_roundtrip_and_readonly_save(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = TinyMergedNetwork().to(device).eval()
        with torch.no_grad():
            network.attention.qkv.weight.add_(.03)
        learner = SimpleNamespace(_network=network, _cur_task=1, _known_classes=4, _total_classes=4)
        manager = SimpleNamespace(_class_order=np.array([3, 1, 0, 2, 5, 4]), _increments=[2, 2, 2])
        x = torch.randn(2, 3, 8, device=device)
        expected = network(x).detach().cpu()
        params = [(id(p), p.device, p.requires_grad, p.detach().clone()) for p in network.parameters()]
        modes = [m.training for m in network.modules()]
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        with tempfile.TemporaryDirectory() as tmp:
            args = dict(logdir=tmp, device=[device], seed=1993, save_task_weights=True)
            directory = checkpoint_directory(args)
            self.assertNotEqual(directory, checkpoint_directory(args))
            path = save_task_weights(directory, learner, manager, args, [{'task': 0, 'top1': 90.}, {'task': 1, 'top1': 80.}])
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix('.pt.tmp').exists())
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(random.getstate(), python_rng)
            self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_rng[1]))
            for a,b in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else []):
                self.assertTrue(torch.equal(a,b))
            self.assertEqual(modes, [m.training for m in network.modules()])
            for before, p in zip(params, network.parameters()):
                self.assertEqual(before[:3], (id(p), p.device, p.requires_grad))
                self.assertTrue(torch.equal(before[3], p))
            payload = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(payload['class_order'], [3, 1, 0, 2, 5, 4])
            self.assertEqual(payload['average_accuracy'], 85.)
            self.assertEqual(payload['last_accuracy'], 80.)
            self.assertTrue(all(v.device.type=='cpu' for v in payload['model_state_dict'].values()))
            restored = TinyMergedNetwork().eval()
            restored.numtask = 0
            load_task_weights(restored, path)
            torch.testing.assert_close(expected, restored(x.cpu()).detach(), rtol=1e-5, atol=1e-6)
            wrapped = torch.nn.DataParallel(network)
            learner._network = wrapped
            path = save_task_weights(directory, learner, manager, args, [{'task': 1, 'top1': 80.}])
            self.assertFalse(any(k.startswith('module.') for k in torch.load(path, weights_only=True)['model_state_dict']))

    def test_opt_in_hook_after_evaluation_and_results(self):
        tree = ast.parse((ROOT/'trainer.py').read_text())
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='_train')
        guards = [n for n in ast.walk(train) if isinstance(n, ast.If) and "save_task_weights" in ast.unparse(n.test)]
        self.assertEqual(len(guards), 2)
        self.assertTrue(all('False' in ast.unparse(n.test) for n in guards))
        text = ast.unparse(train)
        self.assertLess(text.index('model.eval_task()'), text.index('save_task_weights('))
        self.assertLess(text.index('recorded_tasks.append('), text.index('save_task_weights('))
        self.assertNotIn('model.save_checkpoint(', text)

    def test_rerun_only_enables_saving_and_preserves_recipe(self):
        old=json.loads((ROOT/'scripts/sweeps/imgr10_anchor2p5_t10_3090.json').read_text())
        new=json.loads((ROOT/'scripts/sweeps/imgr10_anchor2p5_save_t10_3090.json').read_text())
        a=dict(old['common_overrides']); b=dict(new['common_overrides'])
        a.pop('wandb_group'); b.pop('wandb_group')
        self.assertEqual(b.pop('save_task_weights'), True)
        self.assertEqual(a,b)
        self.assertEqual(old['variants'],new['variants'])
        script=ROOT/'scripts/9_28_imgr10_anchor2p5_save_t10_3090.sh'
        subprocess.run(['bash','-n',str(script)],check=True)
        output=subprocess.check_output(['bash',str(script),'--dry-run'],cwd='/tmp',text=True)
        commands=[line for line in output.splitlines() if 'main.py --config' in line]
        self.assertEqual(len(commands),1)
        tokens=shlex.split(commands[0])
        settings=dict(tokens[i+1].split('=',1) for i,t in enumerate(tokens) if t=='--set')
        self.assertEqual(settings['save_task_weights'],'true')
        self.assertEqual(settings['dual_mask_anchor_reg_weight'],'2.5')
        self.assertEqual(settings['max_tasks'],'10')
        self.assertEqual(settings['seed'],'[1993]')


if __name__=='__main__':
    unittest.main()
