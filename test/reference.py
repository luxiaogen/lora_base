"""Pinned A0 source loaded from Git history, never from experiment switches."""
import ast
import copy
import logging
import subprocess
import types
from contextlib import ExitStack, nullcontext
from functools import partial
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.spatial.distance import cdist
from tqdm import tqdm
import timm.models.helpers as helpers
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

from utils.toolkit import tensor2numpy, count_parameters

SOURCE_REVISION = "e72bc41a0e6af502b46c62bc3e6242b28b851e8e"


def source(path):
    return subprocess.run(
        ["git", "show", SOURCE_REVISION + ":" + path],
        check=True, capture_output=True, text=True).stdout


def definitions(path, names, namespace):
    tree = ast.parse(source(path))
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(tree, path, "exec"), namespace)
    return namespace


def load_reference():
    attention = types.ModuleType("reference_attention")
    exec(compile(source("models/attention.py"), "models/attention.py", "exec"), attention.__dict__)
    namespace = dict(
        torch=torch, nn=nn, F=F, np=np, copy=copy, logging=logging, optim=optim,
        random=__import__("random"), DataLoader=DataLoader, ExitStack=ExitStack,
        MultivariateNormal=MultivariateNormal, tensor2numpy=tensor2numpy,
        count_parameters=count_parameters, cdist=cdist, tqdm=tqdm,
        stage_cost=lambda *_: nullcontext(), OrderedDict=OrderedDict, partial=partial, EPSILON=1e-8,
        PatchEmbed=PatchEmbed, Mlp=Mlp, DropPath=DropPath, trunc_normal_=trunc_normal_,
        lecun_normal_=lecun_normal_, named_apply=helpers.named_apply,
        Attention_LoRA=attention.Attention_LoRA, math=__import__("math"))
    definitions("models/losses.py", {"AngularPenaltySMLoss"}, namespace)
    definitions("methods/base.py", {"BaseLearner"}, namespace)
    definitions("models/vit.py", {
        "LayerScale", "Block_LoRA", "VisionTransformer", "init_weights_vit_timm",
        "init_weights_vit_jax", "init_weights_vit_moco", "get_init_weights_vit",
    }, namespace)
    backbone = namespace["VisionTransformer"]
    definitions("models/network.py", {"ViT", "MANet"}, namespace)
    network = namespace["MANet"]
    learner_namespace = namespace.copy()
    definitions("methods/dlora.py", {"Learner"}, learner_namespace)
    return types.SimpleNamespace(attention=attention, namespace=namespace,
                                 backbone=backbone, network=network, learner=learner_namespace["Learner"])


def original_config(config):
    result = dict(config)
    result.update(
        memory_size=0, memory_per_class=0, fixed_memory=True, optim="sgd",
        use_slora=True, use_plora=True, lora_A_init="kaiming",
        dual_mask_importance="svd", dual_mask_general_ratio=.4,
        dual_mask_competence_adaptive=True, dual_mask_plasticity_adaptive=True,
        dual_mask_protect_strength_mode="competence",
        dual_mask_conflict_energy_adaptive=True, dual_mask_conflict_energy_ratio_floor=True,
        dual_mask_conflict_old_overlap_adaptive=True, dual_mask_competence_all_seen=False,
        dual_mask_task0_gate_mode="unmasked", dual_mask_anchor_reg_enabled=True,
        dual_mask_anchor_reg_task0_only=True, dual_mask_conflict_reg_original_score=True,
        dual_mask_track_w0_metrics=True, ca=True)
    return result
