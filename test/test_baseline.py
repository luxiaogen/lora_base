import ast
import copy
import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from timm.models.layers import trunc_normal_

from models.attention import Attention_LoRA
from models.losses import AngularPenaltySMLoss
from models.network import MANet, ViT
from models.vit import default_cfgs
from main import apply_overrides
from methods.dlora import Learner
from utils.reproducibility import advance_loader_rng
from reference import load_reference, original_config, source


CONFIG = json.loads(Path("exps/dlora/imgr10.json").read_text())
REFERENCE = load_reference()


class ConfigurationTests(unittest.TestCase):
    def test_pretrained_source_is_unchanged(self):
        tree = ast.parse(source("models/vit.py"))
        cfgs = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "default_cfgs"
                            for target in node.targets))
        cfg = next(value for key, value in zip(cfgs.keys, cfgs.values)
                   if ast.literal_eval(key) == "vit_base_patch16_224_in21k")
        expected = {item.arg: ast.literal_eval(item.value) for item in cfg.keywords}
        self.assertEqual(default_cfgs["vit_base_patch16_224_in21k"]["url"], expected["url"])
        self.assertEqual(default_cfgs["vit_base_patch16_224_in21k"]["num_classes"],
                         expected["num_classes"])

    def test_numeric_overrides(self):
        config = copy.deepcopy(CONFIG)
        apply_overrides(config, ["dual_mask_anchor_reg_weight=5", "seed=[1993]",
                                 "data_path=/server/imagenet-r"])
        self.assertEqual(config["dual_mask_anchor_reg_weight"], 5)
        self.assertEqual(config["seed"], [1993])
        self.assertEqual(config["data_path"], "/server/imagenet-r")

    def test_removed_options_cannot_be_silently_enabled(self):
        for key in ("p_permission_release", "ca_real_new_features", "oracle", "lr_typo"):
            with self.assertRaisesRegex(ValueError, key):
                apply_overrides(copy.deepcopy(CONFIG), [key + "=true"])


def seeded(seed=1993):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_attention(cls, task=1):
    seeded()
    attention = cls(dim=12, num_heads=3, r=4, n_tasks=3, qkv_bias=True)
    attention._init_params(original_config(CONFIG))
    attention.set_pretrained_competence(.71, .24)
    attention.set_pretrained_old_overlap_risk(.13)
    attention.before_task(task)
    attention.set_task_and_stage(task, 0)
    return attention


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.old = make_attention(REFERENCE.attention.Attention_LoRA)
        self.new = make_attention(Attention_LoRA)

    def test_initialization_and_rng(self):
        for task in (0, 1, 2):
            old = make_attention(REFERENCE.attention.Attention_LoRA, task)
            old_rng = torch.get_rng_state()
            new = make_attention(Attention_LoRA, task)
            self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
            for branch in ("S_lora", "P_lora"):
                first, second = getattr(old, branch)[task], getattr(new, branch)[task]
                torch.testing.assert_close(first.A_weight, second.A_weight, rtol=0, atol=0)
                torch.testing.assert_close(first.B_weight, second.B_weight, rtol=0, atol=0)
                self.assertEqual(first.A_weight.requires_grad, second.A_weight.requires_grad)
                self.assertEqual(first.B_weight.requires_grad, second.B_weight.requires_grad)

    def test_controls_and_protection(self):
        for c, demand, overlap in ((0., 0., 0.), (.7, .25, .2), (1., 1., 1.)):
            for module in (self.old, self.new):
                module.set_pretrained_competence(c, demand)
                module.set_pretrained_old_overlap_risk(overlap)
                module.rebuild_dual_masks()
            for name in ("w0_importance", "general_mask", "isolated_mask"):
                torch.testing.assert_close(getattr(self.old, name), getattr(self.new, name), rtol=0, atol=0)
            self.assertEqual(self.old._conflict_parameters(), self.new._conflict_parameters())
            self.assertEqual(self.old.current_private_rank, self.new.current_private_rank)
            self.assertTrue(torch.equal(self.new.general_mask + self.new.isolated_mask,
                                        torch.ones_like(self.new.general_mask)))

    def test_effective_updates_and_gradients(self):
        seeded(4)
        data = torch.randn(36, 12)
        for isolated in (False, True):
            old_delta = data.clone().requires_grad_()
            new_delta = data.clone().requires_grad_()
            old_safe = self.old._safe_delta(old_delta, isolated)
            new_safe = self.new._safe_delta(new_delta, isolated)
            torch.testing.assert_close(old_safe, new_safe, rtol=0, atol=0)
            old_safe.square().sum().backward()
            new_safe.square().sum().backward()
            torch.testing.assert_close(old_delta.grad, new_delta.grad, rtol=0, atol=0)
            if isolated:
                self.assertTrue(torch.equal(new_safe[self.new.general_mask.bool()],
                                            torch.zeros_like(new_safe[self.new.general_mask.bool()])))

    def test_task0_unmasked(self):
        attention = make_attention(Attention_LoRA, 0)
        delta = torch.randn(36, 12, requires_grad=True)
        self.assertIs(attention._safe_delta(delta, False), delta)
        self.assertIs(attention._safe_delta(delta, True), delta)
        self.assertEqual(float(attention._joint_conflict_regularization(attention.S_lora[0], False).detach()), 0.)

    def test_regularization(self):
        for branch in ("S_lora", "P_lora"):
            seeded(16)
            value = torch.randn_like(getattr(self.old, branch)[1].B_weight)
            for module in (self.old, self.new):
                with torch.no_grad():
                    getattr(module, branch)[1].B_weight.copy_(value)
            isolated = branch == "P_lora"
            old_loss = self.old._joint_conflict_regularization(getattr(self.old, branch)[1], isolated)
            new_loss = self.new._joint_conflict_regularization(getattr(self.new, branch)[1], isolated)
            torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
            old_loss.backward()
            new_loss.backward()
            torch.testing.assert_close(getattr(self.old, branch)[1].B_weight.grad,
                                        getattr(self.new, branch)[1].B_weight.grad, rtol=0, atol=0)
        torch.testing.assert_close(self.old.anchor_regularization(), self.new.anchor_regularization(),
                                    rtol=0, atol=0)

    def test_zero_b_has_learning_gradient(self):
        seeded(7)
        inputs = torch.randn(2, 5, 12)
        output = self.new(inputs, 1)
        output.square().mean().backward()
        for unit in (self.new.S_lora[1], self.new.P_lora[1]):
            self.assertIsNotNone(unit.B_weight.grad)
            self.assertGreater(float(unit.B_weight.grad.norm()), 0.)
            self.assertIsNone(unit.A_weight.grad)

    def test_forward_merge_and_anchor_lifecycle(self):
        for task in (0, 1, 2):
            old = make_attention(REFERENCE.attention.Attention_LoRA, task)
            new = make_attention(Attention_LoRA, task)
            seeded(23)
            for branch in ("S_lora", "P_lora"):
                values = torch.randn_like(getattr(old, branch)[task].B_weight) * .025
                for module in (old, new):
                    with torch.no_grad():
                        getattr(module, branch)[task].B_weight.copy_(values)
            inputs = torch.randn(2, 5, 12)
            before = new(inputs, task)
            torch.testing.assert_close(before, old(inputs, task), rtol=0, atol=0)
            anchor = new.pretrained_weight.clone()
            old.after_task(task)
            new.after_task(task)
            torch.testing.assert_close(old.qkv.weight, new.qkv.weight, rtol=0, atol=0)
            torch.testing.assert_close(new(inputs, task), before, rtol=2e-5, atol=2e-7)
            weight = new.qkv.weight.clone()
            new.after_task(task)
            torch.testing.assert_close(new.qkv.weight, weight, rtol=0, atol=0)
            self.assertIsNone(new.S_lora[task])
            self.assertIsNone(new.P_lora[task])
            with new.use_pretrained_anchor():
                torch.testing.assert_close(new.qkv.weight, anchor, rtol=0, atol=0)
                with new.use_pretrained_anchor():
                    torch.testing.assert_close(new.qkv.weight, anchor, rtol=0, atol=0)
            torch.testing.assert_close(new.qkv.weight, weight, rtol=0, atol=0)
            torch.testing.assert_close(new.pretrained_weight, anchor, rtol=0, atol=0)

    def test_anchor_context_restores_on_exception(self):
        weight = self.new.qkv.weight.detach().clone()
        with self.assertRaises(ValueError):
            with self.new.use_pretrained_anchor():
                raise ValueError("probe")
        torch.testing.assert_close(self.new.qkv.weight, weight, rtol=0, atol=0)
        self.assertFalse(self.new.pretrained_anchor_mode)


class LoaderTests(unittest.TestCase):
    def test_compatibility_draw(self):
        for workers in (0,):
            loader = DataLoader(TensorDataset(torch.arange(8)), shuffle=False, num_workers=workers)
            seeded()
            list(loader)
            old_state = torch.get_rng_state()
            seeded()
            advance_loader_rng()
            self.assertTrue(torch.equal(old_state, torch.get_rng_state()))


class LossTests(unittest.TestCase):
    def test_cosface_value_and_gradients(self):
        seeded()
        scores = (torch.rand(8, 4) * 2 - 1).requires_grad_()
        clean = scores.detach().clone().requires_grad_()
        labels = torch.arange(8) % 4
        old_loss = REFERENCE.namespace["AngularPenaltySMLoss"](loss_type="cosface", s=20, m=.1)(scores, labels)
        new_loss = AngularPenaltySMLoss(s=20, m=.1)(clean, labels)
        torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
        old_loss.backward()
        new_loss.backward()
        torch.testing.assert_close(scores.grad, clean.grad, rtol=0, atol=0)


class Images(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return index, self.images[index], int(self.labels[index])


class SyntheticData:
    def __init__(self):
        generator = torch.Generator().manual_seed(1993)
        self.images = torch.randn(60, 3, 8, 8, generator=generator)
        self.labels = np.repeat(np.arange(6), 10)
    def get_task_size(self, task):
        return 2
    def get_dataset(self, classes, source, mode, ret_data=False):
        selected = np.isin(self.labels, classes)
        dataset = Images(self.images[selected], self.labels[selected])
        return (self.images[selected], self.labels[selected], dataset) if ret_data else dataset


def cpu_loader(*args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["pin_memory"] = False
    return DataLoader(*args, **kwargs)


class LearnerTests(unittest.TestCase):
    def tiny_network(self, args, old=False):
        namespace = REFERENCE.namespace if old else __import__("models.network", fromlist=["MANet"]).__dict__
        vit_class = REFERENCE.namespace["ViT"] if old else ViT
        def create(*_, **kwargs):
            return vit_class(img_size=8, patch_size=4, embed_dim=12, depth=2,
                             num_heads=3, n_tasks=3, rank=4, num_classes=5)
        with patch.dict(namespace, {"_create_vision_transformer": create}):
            return (REFERENCE.network if old else MANet)(args)

    def run_tasks(self, old=False):
        args = original_config(CONFIG)
        args.update(embd_dim=12, num_heads=3, rank=4, total_sessions=3, init_cls=2,
                    increment=2, batch_size=8, num_workers=0, init_epoch=2, epochs=2,
                    ca_epochs=1, device=[torch.device("cpu")], seed=1993)
        seeded()
        cls = REFERENCE.learner if old else Learner
        namespace = cls.__init__.__globals__
        with patch.dict(namespace, {
                "MANet": lambda config: self.tiny_network(config, old), "DataLoader": cpu_loader}):
            model = cls(args)
            data = SyntheticData()
            records = []
            for task in range(3):
                model.incremental_train(data)
                result = model.eval_task()
                if old:
                    model.eval_w0_task()
                    result = result[0]
                model.after_task()
                records.append((
                    result, {name: value.detach().clone() for name, value in model._network.state_dict().items()
                             if "cls_token_grow" not in name and "pos_embed_grow" not in name},
                    model._class_means.clone(), model._class_covs.clone(), torch.get_rng_state().clone()))
        return records

    def test_three_task_cpu_route_matches_a0(self):
        old = self.run_tasks(True)
        new = self.run_tasks(False)
        for task, (first, second) in enumerate(zip(old, new)):
            self.assertEqual(first[0], second[0], f"task {task} metrics")
            self.assertEqual(first[1].keys(), second[1].keys())
            for name in first[1]:
                torch.testing.assert_close(first[1][name], second[1][name], rtol=0, atol=0,
                                            msg=f"task {task} {name}")
            for index in (2, 3, 4):
                torch.testing.assert_close(first[index], second[index], rtol=0, atol=0,
                                            msg=f"task {task} state {index}")
