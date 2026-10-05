"""Original DualMask attention: fixed W_pre, asymmetric S/P gates, one merge."""
import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenA_TrainableB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, r: int, A_init: torch.Tensor, B_init: torch.Tensor, device=None, dtype=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.r = r
        factory = dict(device=device if device is not None else A_init.device,
                       dtype=dtype if dtype is not None else A_init.dtype)
        self.A = nn.Linear(dim_in, r, bias=False, **factory)
        self.B = nn.Linear(r, dim_out, bias=False, **factory)
        with torch.no_grad():
            self.A.weight.copy_(A_init.to(self.A.weight.device, dtype=self.A.weight.dtype))
            self.B.weight.copy_(B_init.to(self.B.weight.device, dtype=self.B.weight.dtype))
        for p in self.A.parameters():
            p.requires_grad_(False)
        for p in self.B.parameters():
            p.requires_grad_(True)

    @property
    def A_weight(self): return self.A.weight

    @property
    def B_weight(self): return self.B.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


def _random_fixed_A_init(dim: int, r: int, device, dtype) -> torch.Tensor:
    M = torch.randn(dim, r, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q.T.contiguous()

def _kaiming_A_init(dim: int, r: int, device, dtype) -> torch.Tensor:
    A = torch.empty(r, dim, device=device, dtype=dtype)
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    return A

def _zero_B_init(dim: int, r: int, device, dtype) -> torch.Tensor:
    return torch.zeros(dim, r, device=device, dtype=dtype)


def _normalize_score(score: torch.Tensor) -> torch.Tensor:
    score = score.float()
    score = score - score.min()
    denom = score.max().clamp_min(1e-12)
    return score / denom


def _energy_coverage_mask(
        score: torch.Tensor,
        coverage: float,
        valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """根据传入的参数重要性分数 score[i,j], 选取数值最大的coverage权重"""
    coverage = min(max(float(coverage), 0.0), 1.0)
    flat = score.detach().float().flatten().clamp_min(0.0)
    if valid_mask is None:
        valid = torch.ones_like(flat, dtype=torch.bool)
    else:
        valid = valid_mask.detach().bool().flatten()
    selected = torch.zeros_like(flat)
    valid_indices = valid.nonzero(as_tuple=True)[0]
    if valid_indices.numel() == 0:
        return selected.reshape_as(score).to(dtype=score.dtype, device=score.device)

    valid_scores = flat[valid]
    total = valid_scores.sum()

    if coverage <= 0.0 or total <= 0.0:
        return torch.zeros_like(score)
    if coverage >= 1.0:
        selected[valid] = 1.0
        return selected.reshape_as(score).to(dtype=score.dtype, device=score.device)

    values, indices = torch.sort(valid_scores, descending=True)

    cumulative = torch.cumsum(values, dim=0)
    k = int(torch.searchsorted(cumulative, coverage * total).item()) + 1
    selected[valid_indices[indices[:k]]] = 1.0
    return selected.reshape_as(score).to(dtype=score.dtype, device=score.device)


def _energy_coverage_with_ratio_floor_mask(
        score: torch.Tensor,
        ratio: float,
        coverage: float,
        valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select enough coordinates for both the ratio floor and score coverage."""
    ratio = min(max(float(ratio), 0.0), 1.0)
    coverage = min(max(float(coverage), 0.0), 1.0)
    selected = torch.zeros_like(score).flatten()
    if ratio <= 0.0:
        return selected.reshape_as(score)

    if valid_mask is None:
        valid = torch.ones_like(selected, dtype=torch.bool)
    else:
        valid = valid_mask.detach().bool().flatten()
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return selected.reshape_as(score)

    valid_scores = score.detach().float().flatten()[valid].clamp_min(0.0)
    total = valid_scores.sum()
    if total <= 0.0:
        return selected.reshape_as(score)

    values, _ = torch.sort(valid_scores, descending=True)
    ratio_k = valid_count if ratio >= 1.0 else max(1, int(valid_count * ratio))
    if coverage <= 0.0:
        coverage_k = 0
    elif coverage >= 1.0:
        coverage_k = valid_count
    else:
        cumulative = torch.cumsum(values, dim=0)
        coverage_k = int(torch.searchsorted(cumulative, coverage * total).item()) + 1
    k = max(ratio_k, coverage_k)
    threshold = values[k - 1]
    selected[valid] = (valid_scores >= threshold).to(selected.dtype)
    return selected.reshape_as(score).to(dtype=score.dtype, device=score.device)


def _select_svd_rank(
        singular_values: torch.Tensor,
        max_rank: int,
        energy_coverage: float,
) -> int:
    max_rank = max(1, min(int(max_rank), singular_values.numel()))
    coverage = min(max(float(energy_coverage), 0.0), 1.0)
    if coverage <= 0.0:
        return max_rank

    energy = singular_values.detach().float().pow(2)
    total = energy.sum()
    if total <= 0.0:
        return 1
    cumulative = torch.cumsum(energy, dim=0)
    k = int(torch.searchsorted(cumulative, coverage * total).item()) + 1
    return max(1, min(k, max_rank))


class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, r=64, n_tasks=10, eps=1e-12):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.rank = r
        self.n_tasks = n_tasks
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.S_lora = nn.ModuleList([None for _ in range(n_tasks)])
        self.P_lora = nn.ModuleList([None for _ in range(n_tasks)])
        self.cur_task = 0
        self.layer_idx = -1
        shape = (dim * 3, dim)
        self.register_buffer("w0_importance", torch.zeros(shape), persistent=False)
        self.register_buffer("general_mask", torch.ones(shape), persistent=False)
        self.register_buffer("isolated_mask", torch.ones(shape), persistent=False)
        self.register_buffer("pretrained_weight", torch.zeros(shape), persistent=True)
        self.register_buffer("pretrained_anchor_captured", torch.tensor(False), persistent=True)
        self.pretrained_anchor_mode = False

    def _init_params(self, args):
        self.args = args
        self.slora_gamma = float(args["slora_gamma"])
        self.plora_gamma = float(args["plora_gamma"])
        self.dual_mask_svd_rank = int(args["dual_mask_svd_rank"])
        self.dual_mask_svd_energy_coverage = float(args["dual_mask_svd_energy_coverage"])
        self.dual_mask_conflict_ratio = float(args["dual_mask_conflict_ratio"])
        self.dual_mask_conflict_strength = float(args["dual_mask_conflict_strength"])
        self.capture_pretrained_anchor()
        self.set_pretrained_competence(0.0)

    def capture_pretrained_anchor(self):
        if bool(self.pretrained_anchor_captured.item()):
            return
        with torch.no_grad():
            self.pretrained_weight.copy_(self.qkv.weight.detach())
            self.pretrained_anchor_captured.fill_(True)

    def set_pretrained_anchor_mode(self, enabled):
        self.pretrained_anchor_mode = bool(enabled)

    @contextmanager
    def use_pretrained_anchor(self):
        already_in_anchor = self.pretrained_anchor_mode
        if not already_in_anchor:
            accumulated_weight = self.qkv.weight.detach().clone()

            self.set_pretrained_anchor_mode(True)
            with torch.no_grad():
                self.qkv.weight.copy_(self.pretrained_weight)
        try:
            yield
        finally:
            if not already_in_anchor:
                with torch.no_grad():
                    self.qkv.weight.copy_(accumulated_weight)
                self.set_pretrained_anchor_mode(False)

    def set_pretrained_competence(self, competence, plasticity_demand=0.0):
        competence = min(max(float(competence), 0.0), 1.0)
        plasticity_demand = min(max(float(plasticity_demand), 0.0), 1.0)
        control = competence * (1.0 - plasticity_demand)
        self.pretrained_competence = competence
        self.pretrained_plasticity_demand = plasticity_demand
        self.pretrained_control_competence = control
        self.effective_energy_coverage = 0.70 + 0.25 * control
        self.effective_protect_strength = control
        self.current_private_rank = max(1, int(round(self.rank * (1.0 - control * 0.75))))
        self.pretrained_old_overlap_risk = 0.0

    def set_pretrained_old_overlap_risk(self, risk):
        self.pretrained_old_overlap_risk = min(max(float(risk), 0.0), 1.0)

    def before_task(self, task):
        self.cur_task = int(task)
        device = next(self.parameters()).device
        dtype = self.qkv.weight.dtype
        # Keep the original draws, including the initial QR and unused Task0 P.
        a = _random_fixed_A_init(self.dim, self.rank, device, dtype)
        b = _zero_B_init(self.dim * 3, self.rank, device, dtype)
        self.S_lora[task] = FrozenA_TrainableB(
            self.dim, self.dim * 3, self.rank, a, b, device=device, dtype=dtype)
        for p in self.qkv.parameters():
            p.requires_grad_(False)
        for p in self.proj.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            self.S_lora[task].A_weight.copy_(_kaiming_A_init(self.dim, self.rank, device, dtype))
            self.S_lora[task].B_weight.zero_()
        p_rank = self.current_private_rank
        a = _kaiming_A_init(self.dim, p_rank, device, dtype)
        b = _zero_B_init(self.dim * 3, p_rank, device, dtype)
        self.P_lora[task] = FrozenA_TrainableB(
            self.dim, self.dim * 3, p_rank, a, b, device=device, dtype=dtype)
        self.rebuild_dual_masks()

    def set_task_and_stage(self, task, layer_idx):
        self.cur_task = int(task)
        self.layer_idx = int(layer_idx)
        for p in self.qkv.parameters():
            p.requires_grad_(False)
        for p in self.proj.parameters():
            p.requires_grad_(False)
        for unit in list(self.S_lora) + list(self.P_lora):
            if unit is not None:
                unit.A_weight.requires_grad_(False)
                unit.B_weight.requires_grad_(False)
        self.S_lora[task].A_weight.requires_grad_(task == 0)
        self.S_lora[task].B_weight.requires_grad_(True)
        if task > 0:
            self.P_lora[task].B_weight.requires_grad_(True)

    def _svd_importance(self, weight: torch.Tensor) -> torch.Tensor:
        weight_f = weight.detach().float()
        u, s, vh = torch.linalg.svd(weight_f, full_matrices=False)
        k = _select_svd_rank(
            s,
            max_rank=self.dual_mask_svd_rank,
            energy_coverage=(self.dual_mask_svd_energy_coverage),
        )
        self.last_svd_rank = int(k)
        energy = s.detach().float().pow(2)
        self.last_svd_energy_coverage = float((energy[:k].sum() / energy.sum().clamp_min(1e-12)).item())
        s_top = s[:k].clamp_min(0.0)

        row_score = (u[:, :k].pow(2) * s_top.unsqueeze(0)).sum(dim=1)
        col_score = (vh[:k, :].t().pow(2) * s_top.unsqueeze(0)).sum(dim=1)

        score = row_score.unsqueeze(1) * col_score.unsqueeze(0)
        return score.to(device=weight.device, dtype=weight.dtype)

    def rebuild_dual_masks(self):
        with torch.no_grad():
            if self.cur_task > 0 and bool(torch.count_nonzero(self.w0_importance).item()):
                score = self.w0_importance.detach().clone()
            else:
                score = _normalize_score(self._svd_importance(self.pretrained_weight.detach()))
            protect = _energy_coverage_mask(score, self.effective_energy_coverage)
            self.w0_importance.copy_(score.to(self.w0_importance))
            self.general_mask.copy_(protect.to(self.general_mask))
            self.isolated_mask.copy_((1.0 - protect).to(self.isolated_mask))

    def _conflict_parameters(self):
        ratio = min(max(self.dual_mask_conflict_ratio, 0.0), 1.0)
        strength = min(max(self.dual_mask_conflict_strength, 0.0), 1.0)
        strength = min(strength * (1.0 + self.pretrained_old_overlap_risk), 1.0)
        return ratio, strength

    def _joint_conflict(self, delta, conflict_ratio=None):
        importance = self.w0_importance.to(delta)
        score = _normalize_score(importance * _normalize_score(delta.detach().abs()))
        ratio = self.dual_mask_conflict_ratio if conflict_ratio is None else conflict_ratio
        mask = _energy_coverage_with_ratio_floor_mask(score, ratio, coverage=0.50)
        return score, mask

    def _safe_delta(self, delta, isolated, conflict_ratio=None, conflict_strength=None):
        if self.cur_task == 0:
            return delta
        protect = self.general_mask.to(delta)
        alpha = min(max(self.effective_protect_strength, 0.0), 1.0)
        if isolated:
            protect_gate = torch.ones_like(protect)
        else:
            protect_gate = 1.0 - alpha * protect
        if conflict_strength is None:
            _, conflict_strength = self._conflict_parameters()
        beta = min(max(conflict_strength, 0.0), 1.0)
        _, conflict = self._joint_conflict(delta, conflict_ratio)
        conflict_gate = 1.0 - beta * conflict.to(delta.dtype)
        gate = (1.0 - protect) * conflict_gate if isolated else protect_gate * conflict_gate
        return delta * gate

    def anchor_regularization(self):
        task = self.cur_task
        delta = torch.zeros_like(self.qkv.weight)
        s = self.S_lora[task]
        if s is not None:
            raw = self.slora_gamma * (s.B_weight @ s.A_weight)
            delta = delta + self._safe_delta(raw, isolated=False)
        p = self.P_lora[task]
        if task > 0 and p is not None:
            raw = self.plora_gamma * (p.B_weight @ p.A_weight)
            delta = delta + self._safe_delta(raw, isolated=True)
        anchor = self.pretrained_weight.detach().float()
        effective = self.qkv.weight.detach().float() + delta.float()
        return (effective - anchor).pow(2).sum() / anchor.pow(2).sum().clamp_min(1e-12)

    def _joint_conflict_regularization(self, unit, isolated):
        delta = unit.B_weight @ unit.A_weight
        if self.cur_task == 0:
            return delta.sum() * 0.0
        safe = self._safe_delta(delta, isolated)
        importance = self.w0_importance.to(delta)
        protection = (importance * safe.pow(2)).mean()
        score = _normalize_score(importance * _normalize_score(delta.detach().abs()))
        conflict = (score.detach() * safe.pow(2)).mean()
        return protection + conflict

    def _masked_unit_forward(self, x, unit, isolated):
        delta = unit.B_weight @ unit.A_weight
        return F.linear(x, self._safe_delta(delta, isolated))

    def _contrib_from_units(self, x, t_idx):
        output = x.new_zeros((*x.shape[:-1], self.dim * 3))
        if self.pretrained_anchor_mode:
            return output
        s, p = self.S_lora[t_idx], self.P_lora[t_idx]
        if s is not None:
            output = output + self.slora_gamma * self._masked_unit_forward(x, s, False)
        if t_idx > 0 and p is not None:
            output = output + self.plora_gamma * self._masked_unit_forward(x, p, True)
        return output

    def forward(self, x: torch.Tensor, task: int):

        Bsz, N, C = x.shape
        qkv:torch.Tensor = self.qkv(x) + self._contrib_from_units(x, task)
        qkv:torch.Tensor = qkv.reshape(Bsz, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        x = x.transpose(1, 2).reshape(Bsz, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def after_task(self, task):
        deltas = []
        for unit, gamma, isolated in (
                (self.S_lora[task], self.slora_gamma, False),
                (self.P_lora[task], self.plora_gamma, True)):
            if unit is not None and (not isolated or task > 0):
                raw = gamma * (unit.B_weight.detach() @ unit.A_weight.detach())
                deltas.append(self._safe_delta(raw, isolated))
        if deltas:
            with torch.no_grad():
                delta = torch.stack(deltas).sum(dim=0)
                self.qkv.weight.add_(delta.to(self.qkv.weight))
        self.S_lora[task] = None
        self.P_lora[task] = None
