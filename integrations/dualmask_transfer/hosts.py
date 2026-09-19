"""Adapters for pinned official CL-LoRA and SD-LoRA releases."""
import copy
import logging

import torch
from torch import nn
from torch.nn import functional as F

from dualmask_transfer.core import DualMask


def cl_effective(adapter):
    raw = adapter.lora_A.weight @ adapter.lora_B.weight
    if not adapter.dm_active:
        return raw
    return adapter.dm_effective_start + adapter.dm_gate(raw - adapter.dm_raw_start)


def install_cl():
    from backbone.vit_cllora import Adapter_lora, VisionTransformer
    from models.cllora import Learner

    original_forward = Adapter_lora.forward
    original_init = Learner.__init__
    original_finish = VisionTransformer.add_adapter_to_list

    def configure(backbone, task):
        for layer in backbone.adapt_pos:
            adapters = backbone.cur_adapter[backbone.adapt_pos.index(layer)]
            role = "shared" if layer in backbone.general_pos else "private"
            for j, projection in enumerate(("q_proj", "k_proj", "v_proj")):
                adapter = adapters[j]
                if not isinstance(adapter, Adapter_lora):
                    continue
                if not hasattr(adapter, "dm_gate"):
                    weight = getattr(backbone.blocks[layer].attn, projection).weight
                    adapter.dm_gate = DualMask(weight, role).to(weight.device)
                    adapter.register_buffer("dm_raw_start", torch.zeros_like(weight))
                    adapter.register_buffer("dm_effective_start", torch.zeros_like(weight))
                adapter.dm_active = task > 0
        logging.info("DualMask CL task=%d: task0 unchanged; shared task increments, private plastic updates", task)

    def init(self, args):
        original_init(self, args)
        self._network.backbone.dm_task = 0
        configure(self._network.backbone, 0)

    def forward(self, x):
        if not hasattr(self, "dm_gate") or not self.dm_active:
            return original_forward(self, x)
        return F.linear(x, cl_effective(self))

    def finish(self):
        # Consolidate only shared adapters; private adapters are copied by the host.
        with torch.no_grad():
            for layer in self.general_pos:
                for adapter in self.cur_adapter[self.adapt_pos.index(layer)]:
                    if hasattr(adapter, "dm_gate"):
                        effective = cl_effective(adapter).detach().clone()
                        raw = adapter.lora_A.weight @ adapter.lora_B.weight
                        adapter.dm_raw_start.copy_(raw)
                        adapter.dm_effective_start.copy_(effective)
                        # Teachers copied by original_finish must retain effective weights.
                        adapter.dm_active = True
        original_finish(self)
        self.dm_task += 1
        configure(self, self.dm_task)

    Learner.__init__ = init
    Adapter_lora.forward = forward
    VisionTransformer.add_adapter_to_list = finish


def sd_kernel(a, b):
    raw = b.weight @ a.weight
    return b.dm_gate(raw) if hasattr(b, "dm_gate") else raw


def install_sd():
    from backbone.lora import _LoRA_qkv_timm_train

    original_init = _LoRA_qkv_timm_train.__init__
    original_forward = _LoRA_qkv_timm_train.forward
    cached = {}

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.task_id > 0:
            for name, part in (("q", slice(0, self.dim)), ("v", slice(-self.dim, None))):
                key = (self.t_layer_i, name)
                if key not in cached:
                    cached[key] = DualMask(self.qkv.weight[part], "single")
                # Saved B modules carry the same gate to all later tasks.
                getattr(self, "linear_b_" + name).dm_gate = copy.deepcopy(cached[key])
            # Old directions are frozen; only their host magnitudes keep learning.
            with torch.no_grad():
                for offset, name in enumerate(("q", "v")):
                    history = []
                    for task in range(self.task_id):
                        a = self.saved_A["saved_A_" + str(task)][self.t_layer_i * 2 + offset]
                        b = self.saved_B["saved_B_" + str(task)][self.t_layer_i * 2 + offset]
                        history.append(sd_kernel(a, b) / (a.weight.norm() * b.weight.norm()))
                    self.register_buffer("dm_history_" + name, torch.stack(history).to(self.qkv.weight))
        logging.info("DualMask SD task=%d layer=%d: new direction gated, host magnitude learning retained", self.task_id, self.t_layer_i)

    def forward(self, x):
        if self.task_id == 0:
            return original_forward(self, x)
        # Upstream constructs these four temporary modules every forward. Preserve
        # its CPU RNG consumption so this plugin does not alter the random stream.
        nn.Linear(self.dim, self.rank, bias=False)
        nn.Linear(self.rank, self.dim, bias=False)
        nn.Linear(self.dim, self.rank, bias=False)
        nn.Linear(self.rank, self.dim, bias=False)
        updates = []
        for name in ("q", "v"):
            a, b = getattr(self, "linear_a_" + name), getattr(self, "linear_b_" + name)
            kernel = self.scaling_factor[0](sd_kernel(a, b))
            for task in range(self.task_id):
                # Preserve the host's raw-factor normalization, not norm(masked BA).
                kernel = kernel + self.scaling_factor_prev[task](getattr(self, "dm_history_" + name)[task])
            updates.append(F.linear(x, kernel))
        qkv = self.qkv(x)
        return qkv + torch.cat((updates[0], torch.zeros_like(updates[0]), updates[1]), dim=-1)

    _LoRA_qkv_timm_train.__init__ = init
    _LoRA_qkv_timm_train.forward = forward
