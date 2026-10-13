"""固定分层B：W_pre谱权限、S/P独立幅度门及原位合并。"""
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class FrozenA_TrainableB(nn.Module):
    def __init__(self, dim_in, dim_out, r, A_init, B_init, device=None, dtype=None):
        super().__init__()
        self.dim_in, self.dim_out, self.r = dim_in, dim_out, r
        factory = dict(device=device if device is not None else A_init.device,
                       dtype=dtype if dtype is not None else A_init.dtype)
        # 保留原Linear构造的随机抽样，随后再安装实际初始化。
        self.A = nn.Linear(dim_in, r, bias=False, **factory)
        self.B = nn.Linear(r, dim_out, bias=False, **factory)
        with torch.no_grad():
            self.A.weight.copy_(A_init.to(self.A.weight))
            self.B.weight.copy_(B_init.to(self.B.weight))
        self.A.weight.requires_grad_(False)
        self.B.weight.requires_grad_(True)

    @property
    def A_weight(self):
        return self.A.weight

    @property
    def B_weight(self):
        return self.B.weight

    def forward(self, x):
        return self.B(self.A(x))


def _normalize_score(score):
    score = score.float()
    score = score - score.min()
    return score / score.max().clamp_min(1e-12)


def _exact_top_ratio_mask(score, ratio, valid_mask=None):
    flat = score.detach().flatten()
    indices = None if valid_mask is None else valid_mask.detach().bool().flatten().nonzero(as_tuple=True)[0]
    values = flat if indices is None else flat[indices]
    k = min(values.numel(), max(0, int(values.numel() * ratio)))
    selected = torch.zeros_like(flat)
    if k == values.numel():
        if indices is None:
            selected.fill_(1)
        else:
            selected[indices] = 1
    elif k > 0:
        top = torch.topk(values, k, largest=True, sorted=False).indices
        selected[top if indices is None else indices[top]] = 1
    return selected.reshape_as(score)


def _energy_coverage_mask(score, coverage):
    """取排序前缀，覆盖指定的重要性总量；不是坐标比例。"""
    coverage = min(max(float(coverage), 0.0), 1.0)
    flat = score.detach().float().flatten().clamp_min(0.0)
    valid = torch.ones_like(flat, dtype=torch.bool)
    selected = torch.zeros_like(flat)
    valid_indices = valid.nonzero(as_tuple=True)[0]
    valid_scores = flat[valid]
    total = valid_scores.sum()
    if coverage <= 0.0 or total <= 0.0:
        return torch.zeros_like(score)
    if coverage >= 1.0:
        selected[valid] = 1.0
    else:
        values, indices = torch.sort(valid_scores, descending=True)
        cumulative = torch.cumsum(values, dim=0)
        k = int(torch.searchsorted(cumulative, coverage * total).item()) + 1
        selected[valid_indices[indices[:k]]] = 1.0
    return selected.reshape_as(score).to(score)


def _select_svd_rank(singular_values, max_rank, energy_coverage):
    max_rank = max(1, min(int(max_rank), singular_values.numel()))
    energy = singular_values.detach().float().pow(2)
    total = energy.sum()
    if total <= 0.0:
        return 1
    k = int(torch.searchsorted(torch.cumsum(energy, dim=0), energy_coverage * total).item()) + 1
    return max(1, min(k, max_rank))


def _random_fixed_A_init(dim, r, device, dtype):
    M = torch.randn(dim, r, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(M, mode='reduced')
    return Q.T.contiguous()


def _kaiming_A_init(dim, r, device, dtype):
    A = torch.empty(r, dim, device=device, dtype=dtype)
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    return A


class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., r=64, n_tasks=10, eps=1e-12):
        super().__init__()
        self.dim, self.rank, self.n_tasks, self.num_heads = dim, r, n_tasks, num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.S_lora = nn.ModuleList([None for _ in range(n_tasks)])
        self.P_lora = nn.ModuleList([None for _ in range(n_tasks)])
        self.cur_task, self.layer_idx = 0, -1
        shape = (dim * 3, dim)
        self.register_buffer('w0_importance', torch.zeros(shape), persistent=False)
        self.register_buffer('general_mask', torch.ones(shape), persistent=False)
        self.register_buffer('isolated_mask', torch.ones(shape), persistent=False)
        self.register_buffer('pretrained_weight', torch.zeros(shape))
        self.register_buffer('pretrained_anchor_captured', torch.tensor(False, dtype=torch.bool))
        self.pretrained_anchor_mode = False
        self.pretrained_old_overlap_risk = 0.
        self.effective_energy_coverage = .9
        self.effective_protect_strength = .5
        self.slora_gamma, self.plora_gamma = .5, .75
        self.last_svd_rank, self.last_svd_energy_coverage = r, 0.

    def _init_params(self, args):
        self.slora_gamma = float(args.get('slora_gamma', .5))
        self.plora_gamma = float(args.get('plora_gamma', .75))
        self.task0_p_init_rank = int(args.get('task0_p_init_rank', self.rank))
        self.svd_energy_coverage = float(args.get('svd_energy_coverage', .95))
        self.protect_mass_coverage = float(args.get('protect_mass_coverage', .90))
        self.fixed_protect_strength = float(args.get('protect_strength', .5))
        self.conflict_top_fraction = float(args.get('conflict_top_fraction', .1))
        self.conflict_base_strength = float(args.get('conflict_base_strength', .5))
        self.effective_protect_strength = self.fixed_protect_strength
        self.capture_pretrained_anchor()

    def capture_pretrained_anchor(self, force=False):
        if bool(self.pretrained_anchor_captured.item()) and not force:
            return
        with torch.no_grad():
            self.pretrained_weight.copy_(self.qkv.weight.detach())
            self.pretrained_anchor_captured.fill_(True)

    def set_pretrained_anchor_mode(self, enabled):
        self.pretrained_anchor_mode = bool(enabled)

    @contextmanager
    def use_pretrained_anchor(self):
        already = self.pretrained_anchor_mode
        if not already:
            accumulated = self.qkv.weight.detach().clone()
            self.set_pretrained_anchor_mode(True)
            with torch.no_grad():
                self.qkv.weight.copy_(self.pretrained_weight)
        try:
            yield
        finally:
            if not already:
                with torch.no_grad():
                    self.qkv.weight.copy_(accumulated)
                self.set_pretrained_anchor_mode(False)

    def relative_weight_drift(self):
        anchor = self.pretrained_weight.detach().float()
        delta = self.qkv.weight.detach().float() - anchor
        return float((delta.norm() / anchor.norm().clamp_min(1e-12)).item())

    def set_pretrained_old_overlap_risk(self, risk):
        self.pretrained_old_overlap_risk = min(max(float(risk), 0.), 1.)

    def before_task(self, task):
        t = self.cur_task = int(task)
        if t > 0:
            self.effective_protect_strength = self.fixed_protect_strength
        device, dtype = self.qkv.weight.device, self.qkv.weight.dtype
        rs = self.rank
        A = _random_fixed_A_init(self.dim, rs, device, dtype)
        B = torch.zeros(self.dim * 3, rs, device=device, dtype=dtype)
        self.S_lora[t] = FrozenA_TrainableB(self.dim, self.dim * 3, rs, A, B, device, dtype)
        for p in self.qkv.parameters():
            p.requires_grad_(False)
        for p in self.proj.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            self.S_lora[t].A.weight.copy_(_kaiming_A_init(self.dim, rs, device, dtype))
            self.S_lora[t].B.weight.zero_()
        # Task0的P不生效；其配置秩只恢复旧B的随机抽样顺序。
        rp = self.task0_p_init_rank if t == 0 else self.rank
        A = _kaiming_A_init(self.dim, rp, device, dtype)
        B = torch.zeros(self.dim * 3, rp, device=device, dtype=dtype)
        self.P_lora[t] = FrozenA_TrainableB(self.dim, self.dim * 3, rp, A, B, device, dtype)
        self.rebuild_dual_masks()

    def set_task_and_stage(self, task, layer_idx, stage=0):
        self.cur_task, self.layer_idx = int(task), int(layer_idx)
        for p in list(self.qkv.parameters()) + list(self.proj.parameters()):
            p.requires_grad_(False)
        for unit in list(self.S_lora) + list(self.P_lora):
            if unit is not None:
                unit.A.weight.requires_grad_(False)
                unit.B.weight.requires_grad_(False)
        self.S_lora[task].A.weight.requires_grad_(task == 0)
        self.S_lora[task].B.weight.requires_grad_(True)
        if task > 0:
            self.P_lora[task].B.weight.requires_grad_(True)

    def _svd_importance(self, weight):
        u, s, vh = torch.linalg.svd(weight.detach().float(), full_matrices=False)
        k = _select_svd_rank(s, 768, self.svd_energy_coverage)
        self.last_svd_rank = int(k)
        energy = s.detach().float().pow(2)
        self.last_svd_energy_coverage = float((energy[:k].sum() / energy.sum().clamp_min(1e-12)).item())
        s_top = s[:k].clamp_min(0.)
        row = (u[:, :k].pow(2) * s_top.unsqueeze(0)).sum(dim=1)
        col = (vh[:k, :].t().pow(2) * s_top.unsqueeze(0)).sum(dim=1)
        return (row.unsqueeze(1) * col.unsqueeze(0)).to(weight)

    def rebuild_dual_masks(self, coverage_override=None):
        with torch.no_grad():
            reuse = self.cur_task > 0 and bool(torch.count_nonzero(self.w0_importance).item())
            score = self.w0_importance.detach().clone() if reuse else _normalize_score(self._svd_importance(self.pretrained_weight.detach()))
            coverage = self.protect_mass_coverage
            if coverage_override is not None:
                coverage = coverage_override
            self.effective_energy_coverage = coverage
            protect = _energy_coverage_mask(score, coverage)
            self.w0_importance.copy_(score.to(self.w0_importance))
            self.general_mask.copy_(protect.to(self.general_mask))
            self.isolated_mask.copy_((1.0 - protect).to(self.isolated_mask))

    def _conflict_parameters(self):
        return self.conflict_top_fraction, min(self.conflict_base_strength * (1.0 + self.pretrained_old_overlap_risk), 1.0)

    def _joint_conflict(self, delta, conflict_ratio=None, valid_mask=None):
        # 保持归一化与torch.topk的原顺序，包含并列值的原有选择行为。
        score = _normalize_score(delta.detach().abs())
        ratio = self.conflict_top_fraction if conflict_ratio is None else conflict_ratio
        return score, _exact_top_ratio_mask(score, ratio, valid_mask)

    def _safe_delta(self, delta, isolated, conflict_ratio=None, conflict_strength=None, return_details=False):
        if self.cur_task == 0:
            return (delta, torch.ones_like(delta), torch.zeros_like(delta)) if return_details else delta
        protect = self.general_mask.to(delta)
        permission = 1.0 - protect if isolated else 1.0 - self.effective_protect_strength * protect
        if conflict_strength is None:
            conflict_strength = self._conflict_parameters()[1]
        _, mask = self._joint_conflict(delta, conflict_ratio)
        gate = permission * (1.0 - conflict_strength * mask.to(delta.dtype))
        safe = delta * gate
        return (safe, gate, mask) if return_details else safe

    def _masked_unit_forward(self, x, unit, isolated, residual_scale=1.):
        return F.linear(x, self._safe_delta(unit.B_weight @ unit.A_weight, isolated))

    def anchor_regularization(self):
        unit = self.S_lora[self.cur_task]
        current = torch.zeros_like(self.qkv.weight)
        if unit is not None:
            raw = self.slora_gamma * (unit.B_weight @ unit.A_weight)
            current = current + self._safe_delta(raw, False)
        anchor = self.pretrained_weight.detach().float()
        effective = self.qkv.weight.detach().float() + current.float()
        return (effective - anchor).pow(2).sum() / anchor.pow(2).sum().clamp_min(1e-12)

    def _contrib_from_units(self, x, t_idx):
        out = x.new_zeros((*x.shape[:-1], self.dim * 3))
        if self.pretrained_anchor_mode:
            return out
        s, p = self.S_lora[t_idx], self.P_lora[t_idx]
        if s is not None:
            out = out + self.slora_gamma * self._masked_unit_forward(x, s, False)
        if t_idx > 0 and p is not None:
            out = out + self.plora_gamma * self._masked_unit_forward(x, p, True)
        return out

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        b, n, c = x.shape
        qkv = self.qkv(x) + self._contrib_from_units(x, task)
        qkv = qkv.reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))

    def after_task(self, task):
        t = int(task)
        if self.S_lora[t] is None and self.P_lora[t] is None:
            return
        updates = []
        with torch.no_grad():
            for unit, gamma, isolated in ((self.S_lora[t], self.slora_gamma, False),
                                          (self.P_lora[t] if t > 0 else None, self.plora_gamma, True)):
                if unit is not None:
                    # 旧B合并顺序先乘gamma，再计算同一幅度门；回归测试检查其行为。
                    raw = gamma * (unit.B_weight.detach() @ unit.A_weight.detach())
                    updates.append(self._safe_delta(raw, isolated))
            delta = torch.stack(updates).sum(dim=0)
            self.qkv.weight.add_(delta.to(self.qkv.weight))
        self.S_lora[t] = self.P_lora[t] = None
