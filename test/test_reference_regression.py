"""对照真实旧B源码：初始化、RNG、前向/损失/梯度/SGD/合并。"""
import importlib.util
import ast
import copy
from contextlib import nullcontext
import logging
import json
import os
from pathlib import Path
import sys

import unittest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(os.environ.get('DUALMASK_REFERENCE', str(ROOT / '_reference_archive')))
sys.path.insert(0, str(ROOT))
from models.attention import Attention_LoRA
from models.losses import AngularPenaltySMLoss
from utils.depth_permissions import calibrate_center, install_depth_permissions


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference():
    if not REFERENCE.is_dir():
        raise unittest.SkipTest('旧B归档未提供；设置DUALMASK_REFERENCE后执行对照回归')
    cfg = json.loads((REFERENCE / 'reference_run.json').read_text())['effective_config']
    cfg['seed'] = 1993
    cfg['task0_p_init_rank'] = 39
    # 本测试只取消无RNG的日志/只读位置诊断，活动训练配置保持B。
    cfg['dual_mask_mechanism_audit'] = False
    cfg['dual_mask_applied_budget_log'] = False
    cfg['dual_mask_depth_audit'] = False
    return load('reference_attention', REFERENCE / 'models/attention.py'), cfg


def make_layers(cls, cfg):
    torch.manual_seed(1993)
    layers = [cls(64, num_heads=4, qkv_bias=True, r=64, n_tasks=2) for _ in range(4)]
    for i, layer in enumerate(layers):
        layer._init_params(cfg)
        layer.layer_idx = i
    return layers, torch.get_rng_state().clone()


def check_real_tensor_step(reference, task, device):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    old, cfg = reference
    old_layers, old_rng = make_layers(old.Attention_LoRA, cfg)
    new_layers, new_rng = make_layers(Attention_LoRA, cfg)
    for layer in old_layers + new_layers:
        layer.to(device)
    assert torch.equal(old_rng, new_rng)
    for a, b in zip(old_layers, new_layers):
        torch.testing.assert_close(a.qkv.weight, b.qkv.weight, rtol=0, atol=0)
    states = []
    for layers in (old_layers, new_layers):
        torch.set_rng_state(old_rng)
        if device == 'cuda':
            torch.cuda.manual_seed_all(1993)
        for layer in layers:
            if not isinstance(layer, Attention_LoRA):
                # .65*(1-.2)=.52，旧公式得到rank39；新模块读取固定配置。
                layer.set_pretrained_competence(.65, .2 if task == 0 else 0.)
            layer.set_pretrained_old_overlap_risk(.37)
            layer.before_task(task)
            layer.set_task_and_stage(task, layer.layer_idx)
        states.append(torch.get_rng_state().clone())
        if device == 'cuda':
            states.append(torch.cuda.get_rng_state().clone())
    if device == 'cuda':
        assert torch.equal(states[0], states[2]) and torch.equal(states[1], states[3])
    else:
        assert torch.equal(*states)
    old_depth = load('reference_depth', REFERENCE / 'utils/depth_permissions.py')
    old_center = old_depth.install_depth_permissions(old_layers, task, cfg)
    new_center = install_depth_permissions(new_layers, task, cfg)
    assert old_center == new_center
    x = torch.randn(2, 5, 64, generator=torch.Generator().manual_seed(8)).to(device)
    target = torch.randn(2, 5, 64, generator=torch.Generator().manual_seed(9)).to(device)
    for a, b in zip(old_layers, new_layers):
        # Task0的保护掩码不参与前向；清理其无效NCM覆盖赋值不改变训练。
        for name in (('general_mask', 'isolated_mask', 'w0_importance') if task > 0 else ('w0_importance',)):
            torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
        for name in ('S_lora', 'P_lora'):
            torch.testing.assert_close(getattr(a, name)[task].A_weight, getattr(b, name)[task].A_weight, rtol=0, atol=0)
        outcomes = []
        for layer in (a, b):
            layer.train()
            optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=.02, momentum=.9)
            for _ in range(2):
                output = layer(x, task)
                loss = (output - target).square().mean()
                if task == 0:
                    loss = loss + 2.5 * layer.anchor_regularization()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grads = {name: p.grad.clone() for name, p in layer.named_parameters() if p.grad is not None}
                optimizer.step()
            layer.eval()
            before_merge = layer(x, task).detach()
            layer.after_task(task)
            after_merge = layer(x, task).detach()
            torch.testing.assert_close(before_merge, after_merge, rtol=2e-6, atol=2e-6)
            merged = layer.qkv.weight.detach().clone()
            layer.after_task(task)
            assert torch.equal(merged, layer.qkv.weight)
            outcomes.append((loss.detach(), grads, merged, after_merge))
        for aa, bb in zip(outcomes[0][0:1] + outcomes[0][2:], outcomes[1][0:1] + outcomes[1][2:]):
            torch.testing.assert_close(aa, bb, rtol=0, atol=0)
        assert outcomes[0][1].keys() == outcomes[1][1].keys()
        for name in outcomes[0][1]:
            torch.testing.assert_close(outcomes[0][1][name], outcomes[1][1][name], rtol=0, atol=0)


def check_depth_calibration():
    old = load('reference_depth', REFERENCE / 'utils/depth_permissions.py')
    scores = [torch.rand(48, 16, generator=torch.Generator().manual_seed(i)) for i in range(12)]
    assert calibrate_center(scores) == old.calibrate_center(scores)


def check_CosFace():
    old = load('reference_losses', REFERENCE / 'models/losses.py')
    logits = torch.randn(8, 20, generator=torch.Generator().manual_seed(1993)).tanh()
    targets = torch.arange(8)
    outcomes = []
    for cls in (old.AngularPenaltySMLoss, AngularPenaltySMLoss):
        x = logits.clone().requires_grad_()
        loss = cls(loss_type='cosface', s=20., m=.1)(x, targets)
        loss.backward()
        outcomes.append((loss.detach(), x.grad))
    for a, b in zip(*outcomes):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def check_NME_reporting():
    from methods.dlora import Learner
    from torch.utils.data import DataLoader, TensorDataset
    class TinyNetwork(torch.nn.Module):
        def extract_vector(self, inputs):
            return inputs
        def interface(self, inputs):
            return inputs
    learner = object.__new__(Learner)
    learner._network = TinyNetwork()
    learner._device = torch.device('cpu')
    learner._cur_task, learner._known_classes, learner.class_num, learner.topk = 0, 0, 2, 1
    learner.acc_matrix = torch.zeros(1, 1).numpy()
    learner._class_means = torch.eye(2)
    labels = torch.tensor([0, 1, 0])
    learner.test_loader = DataLoader(TensorDataset(torch.arange(3), torch.eye(2)[labels], labels), batch_size=2)
    cnn, _, nme, _ = learner.eval_task()
    assert cnn['top1'] == 100.
    assert nme['top1'] == 100.


@unittest.skipUnless(REFERENCE.is_dir(), '旧B归档未提供；设置DUALMASK_REFERENCE后执行对照回归')
class ParityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cpu_task0(self):
        check_real_tensor_step(reference(), 0, 'cpu')

    def test_cpu_task1(self):
        check_real_tensor_step(reference(), 1, 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA回归在3090执行')
    def test_cuda_task0(self):
        check_real_tensor_step(reference(), 0, 'cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA回归在3090执行')
    def test_cuda_task1(self):
        check_real_tensor_step(reference(), 1, 'cuda')

    def test_depth_calibration(self):
        check_depth_calibration()

    def test_CosFace(self):
        check_CosFace()

    def test_NME_reporting(self):
        check_NME_reporting()

    def test_real_classification_training_and_CA_match_old_B(self):
        from methods.dlora import Learner
        from models.attention import Attention_LoRA as NewAttention
        from models.losses import AngularPenaltySMLoss
        from torch.utils.data import DataLoader, TensorDataset
        from torch.distributions.multivariate_normal import MultivariateNormal
        from tqdm import tqdm
        from utils.toolkit import tensor2numpy
        old_attention, cfg = reference()
        old_loss = load('old_training_loss', REFERENCE / 'models/losses.py')
        old_toolkit = load('old_training_toolkit', REFERENCE / 'utils/toolkit.py')
        tree = ast.parse((REFERENCE / 'methods/dlora.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Learner')
        # 只加载参照类，不加载失活辅助模块或预训练权重。
        cls.bases = []
        space = dict(torch=torch, optim=torch.optim, F=torch.nn.functional, np=np, copy=copy,
                     logging=logging, tqdm=tqdm, DataLoader=DataLoader,
                     MultivariateNormal=MultivariateNormal, tensor2numpy=tensor2numpy,
                     count_parameters=old_toolkit.count_parameters,
                     AngularPenaltySMLoss=old_loss.AngularPenaltySMLoss,
                     stage_cost=lambda *a, **k: nullcontext(), ExitStack=__import__('contextlib').ExitStack)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<old B Learner>', 'exec'), space)
        OldLearner = space['Learner']
        cfg = dict(cfg, init_epoch=2, epochs=2, ca_epochs=2)

        class MiniNetwork(torch.nn.Module):
            def __init__(self, attention_class):
                super().__init__()
                self.attention = attention_class(64, num_heads=4, qkv_bias=True, r=64, n_tasks=2)
                self.classifier_pool = torch.nn.ModuleList([torch.nn.Linear(64, 20, bias=False) for _ in range(2)])
                self.numtask = 1
            def extract_vector(self, x):
                return self.attention(x, self.numtask - 1).mean(1)
            def forward(self, x, fc_only=False):
                if fc_only:
                    return torch.cat([torch.nn.functional.linear(torch.nn.functional.normalize(x, dim=1),
                       torch.nn.functional.normalize(h.weight, dim=1)) for h in self.classifier_pool[:self.numtask]], 1)
                features = self.extract_vector(x)
                return dict(features=features, logits=torch.nn.functional.linear(
                    torch.nn.functional.normalize(features, dim=1),
                    torch.nn.functional.normalize(self.classifier_pool[self.numtask - 1].weight, dim=1)))
            def interface(self, x):
                return self(self.extract_vector(x), fc_only=True)

        for task in (0, 1):
            outcomes = []
            for learner_class, attention_class in ((OldLearner, old_attention.Attention_LoRA),
                                                   (Learner, NewAttention)):
                torch.manual_seed(1993)
                net = MiniNetwork(attention_class)
                net.attention._init_params(cfg)
                net.numtask = task + 1
                if attention_class is not NewAttention:
                    net.attention.set_pretrained_competence(.65, .2 if task == 0 else 0.)
                net.attention.before_task(task)
                net.attention.set_task_and_stage(task, 0)
                for i, head in enumerate(net.classifier_pool):
                    head.weight.requires_grad_(i == task)
                learner = object.__new__(learner_class)
                learner._network, learner.args = net, cfg
                learner._device, learner._multiple_gpus = torch.device('cpu'), [torch.device('cpu')]
                learner._cur_task, learner._known_classes, learner._total_classes = task, task * 20, (task + 1) * 20
                learner.class_num, learner.topk, learner.scale, learner.margin = 20, 1, 20., .1
                learner.run_epoch, learner.logit_norm, learner.task_sizes = 2, .1, [20] * (task + 1)
                learner._iter_lora_modules = lambda: iter([net.attention])
                x = torch.randn(12, 3, 64, generator=torch.Generator().manual_seed(9))
                labels = torch.arange(12) % 20 + task * 20
                loader = DataLoader(TensorDataset(torch.arange(12), x, labels), batch_size=4, shuffle=True)
                groups = [{'params': list(net.attention.S_lora[task].parameters()) + list(net.attention.P_lora[task].parameters()),
                           'lr': .02, 'momentum': .9, 'weight_decay': 0.},
                          {'params': list(net.classifier_pool[task].parameters()), 'lr': .02, 'momentum': .9, 'weight_decay': 0.}]
                optimizer = torch.optim.SGD(groups)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
                learner.train_function(loader, loader, optimizer, scheduler)
                # 同样的高斯特征采样与分类头更新，不使用外部数据或预训练下载。
                learner._class_means = torch.randn((task + 1) * 20, 64, generator=torch.Generator().manual_seed(22))
                learner._class_covs = torch.eye(64)[None].repeat((task + 1) * 20, 1, 1) * .02
                learner._stage2_compact_classifier(20)
                outcomes.append(({name: p.detach().clone() for name, p in net.named_parameters()},
                                 torch.get_rng_state().clone()))
            assert torch.equal(outcomes[0][1], outcomes[1][1])
            assert outcomes[0][0].keys() == outcomes[1][0].keys()
            for name in outcomes[0][0]:
                torch.testing.assert_close(outcomes[0][0][name], outcomes[1][0][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
