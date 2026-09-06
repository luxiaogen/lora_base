import logging
import math
import os
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenA_TrainableB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, r: int, A_init: torch.Tensor, B_init: torch.Tensor, device=None, dtype=None):
        super().__init__()
        assert A_init.shape == (r, dim_in)   # 64,768
        assert B_init.shape == (dim_out, r) # 2304,64
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.r = r
        # ** 表示对字典进行关键字解包（Unpacking）。这一行代码完全等价于nn.Linear(dim_in, r, bias=False, device=device, dtype=dtype)
        factory = dict(device=device if device is not None else A_init.device,
                       dtype=dtype if dtype is not None else A_init.dtype) # {'device': device(type='cuda', index=0), 'dtype': torch.float32}
        self.A = nn.Linear(dim_in, r, bias=False, **factory)
        self.B = nn.Linear(r, dim_out, bias=False, **factory)
        # 所有对 param.data 或 param.copy_() 进行初始化赋值的操作，内部都会强制包裹在 with torch.no_grad(): 下。这是标准且最安全的参数初始化写法
        with torch.no_grad(): # 参数赋值不是推理/前向传播
            self.A.weight.copy_(A_init.to(self.A.weight.device, dtype=self.A.weight.dtype))
            self.B.weight.copy_(B_init.to(self.B.weight.device, dtype=self.B.weight.dtype))
        # A B都训练
        for p in self.A.parameters():
            p.requires_grad_(False)
        for p in self.B.parameters():
            p.requires_grad_(True)

    @property
    def A_weight(self): return self.A.weight

    @property
    def B_weight(self): return self.B.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))  # (..., dim)



def _random_fixed_A_init(dim: int, r: int, device, dtype) -> torch.Tensor:
    M = torch.randn(dim, r, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q.T.contiguous()  # (r, dim)

def _kaiming_A_init(dim: int, r: int, device, dtype) -> torch.Tensor:
    A = torch.empty(r, dim, device=device, dtype=dtype)
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    return A

def _zero_B_init(dim: int, r: int, device, dtype) -> torch.Tensor:
    return torch.zeros(dim, r, device=device, dtype=dtype)

# 固定的跨数据集平衡策略。它复现旧版默认控制器，但不再暴露为独立超参数。
_BALANCED_COVERAGE_BASE = 0.70
_BALANCED_COVERAGE_SPAN = 0.25
_BALANCED_STRENGTH_BASE = 0.30
_BALANCED_STRENGTH_SPAN = 0.60
_BALANCED_PRIVATE_MIN_RATIO = 0.25
_BALANCED_STATIC_STRENGTH = 0.70
_CONFLICT_ENERGY_COVERAGE = 0.50



def _normalize_score(score: torch.Tensor) -> torch.Tensor:
    score = score.float()
    score = score - score.min()
    denom = score.max().clamp_min(1e-12)
    return score / denom


def _top_ratio_mask(score: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = min(max(float(ratio), 0.0), 1.0)  # 0.5
    flat = score.flatten()  # 2304*768
    if flat.max() <= flat.min():
        return torch.zeros_like(score)
    if ratio <= 0.0:
        return torch.zeros_like(score)
    if ratio >= 1.0:
        return torch.ones_like(score)
    k = max(1, int(flat.numel() * ratio))  # 2304*768*0.5
    threshold = torch.topk(flat, k, largest=True).values.min()  # tensor(0.0032) | 选这50%中要保护的区域的最小值
    return (score >= threshold).to(score.dtype)  # 大于这个阈值的就是要保护的区域


def _masked_top_ratio_mask(
        score: torch.Tensor,
        valid_mask: torch.Tensor,
        ratio: float,
) -> torch.Tensor:
    """Select Top-r only among valid coordinates."""
    ratio = min(max(float(ratio), 0.0), 1.0)
    valid = valid_mask.detach().bool().flatten()
    selected = torch.zeros_like(score).flatten()
    valid_count = int(valid.sum().item())
    if ratio <= 0.0 or valid_count == 0:
        return selected.reshape_as(score)
    if ratio >= 1.0:
        return valid.reshape_as(score).to(dtype=score.dtype, device=score.device)

    valid_scores = score.flatten()[valid]
    k = max(1, int(valid_count * ratio))
    threshold = torch.topk(valid_scores, k, largest=True).values.min()
    selected[valid] = (valid_scores >= threshold).to(selected.dtype)
    return selected.reshape_as(score)


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
    # 选取能够覆盖 95% 奇异值平方能量的最小 rank k | 预定义的谱覆盖率
    coverage = min(max(float(energy_coverage), 0.0), 1.0)  #
    if coverage <= 0.0:
        return max_rank

    energy = singular_values.detach().float().pow(2)
    total = energy.sum()
    if total <= 0.0:
        return 1
    cumulative = torch.cumsum(energy, dim=0)
    k = int(torch.searchsorted(cumulative, coverage * total).item()) + 1   # 一般都是大于32
    return max(1, min(k, max_rank))

class Attention_LoRA(nn.Module):
    """
        带W0保护与BA冲突控制的双掩码LoRA分支
        第一个掩码标记了重要的预训练权重 W₀ 方向，这些方向在训练中应受到保护。
        第二个掩码标记了重要性较低的塑性区域，在这些区域中，独立的 BA 更新可以更自由地移动。
        两者之间的交互作用通过一个轻量级的联合得分 normalize(I_W₀) × normalize(I_BA) 进行近似估计
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, r=64, n_tasks=10, eps=1e-12):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.rank = r
        self.n_tasks = n_tasks # 20 任务数

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.S_lora = nn.ModuleList([None for _ in range(n_tasks)])
        self.cur_task = 0
        shape = (self.dim * 3, self.dim)

        self.P_lora = torch.nn.ModuleList([None for _ in range(self.n_tasks)])
        # W0 重要性生成保护区 general_mask；其补集 isolated_mask 供 P 分支使用。
        self.register_buffer("w0_importance", torch.zeros(shape), persistent=False)
        # general_mask[i, j] = 1  表示这个 W0 位置重要，要保护 | general_mask[i, j] = 0  表示这个位置相对不重要，可以改   W_pre/W_0 保护区域
        self.register_buffer("general_mask", torch.ones(shape), persistent=False)
        # isolated_mask[i, j] = 1  表示这个位置不太重要，可以给 isolated branch 改 | isolated_mask[i, j] = 0  表示这个位置重要，不给 P_lora 改  可塑区域
        self.register_buffer("isolated_mask", torch.ones(shape), persistent=False)

        # Final per-task conflict-distribution diagnostics.  They are populated
        # at merge time and never participate in forward/backward computation.
        self.register_buffer("last_conflict_entropy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_conflict_top10_energy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_conflict_energy50_ratio", torch.tensor(0.0), persistent=False)

        self.register_buffer("last_conflict_gate_suppression", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_safe_suppression", torch.tensor(0.0), persistent=False)

        self.register_buffer("last_effective_conflict_ratio", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_effective_conflict_strength", torch.tensor(0.0), persistent=False)

        self.register_buffer("last_private_conflict_mask_overlap", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_private_conflict_energy_overlap", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_private_conflict_gate_suppression", torch.tensor(0.0), persistent=False)

        self.register_buffer("last_relocation_target_energy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_relocation_recovered_energy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_relocation_activation_error", torch.tensor(0.0), persistent=False)

        self.register_buffer("pretrained_weight", torch.zeros(shape), persistent=True)
        self.register_buffer("pretrained_anchor_captured", torch.tensor(False, dtype=torch.bool), persistent=True)


        # 先判断 W0 哪里重要，再决定 LoRA 的 BA 哪里能加、哪里不能加。
        self.dual_mask_importance = "svd"
        self.dual_mask_general_ratio = 0.5  # 决定 W0 保护区多大
        self.dual_mask_layerwise_ratio_mode = "none"

        self.dual_mask_svd_rank = self.rank  # 决定 SVD 重要性用多少主方向
        self.last_svd_rank = self.rank

        self.last_svd_energy_coverage = 0.0

        self.dual_mask_conflict_ratio = 0.25  # 决定 BA-W0 冲突区多大
        self.dual_mask_conflict_strength = 1.0  # 决定冲突区压制多强

        self.dual_mask_conflict_reg_enabled = True

        self.dual_mask_conflict_energy_adaptive = False

        # 冲突区域的重叠是否自适应
        self.dual_mask_conflict_old_overlap_adaptive = False

        self.dual_mask_private_conflict_mode = "global"

        # task0的学习方式 放开学/没有冲突部分/正常
        self.dual_mask_task0_gate_mode = "full"
        self.dual_mask_s_protect_enabled = True

        self.dual_mask_conflict_merge_mode = "suppress"

        self._functional_merge_strength_override = None
        self.last_functional_merge_strength = float("nan")

        self.dual_mask_relocation_steps = 20
        self.dual_mask_relocation_lr = 0.1
        self.dual_mask_relocation_vectors = 64
        self._pending_relocations = {}
        self._relocation_input_collection = None

        self.dual_mask_safe_residual_enabled = False
        self.dual_mask_safe_residual_vectors = 64
        self._pending_safe_residual_deltas = []
        self._last_safe_residual_loss = None

        self.dual_mask_svd_energy_coverage = 0.0
        self.dual_mask_competence_adaptive = False

        self.dual_mask_plasticity_adaptive = False
        self.dual_mask_protect_strength_mode = "legacy_linear"

        self.pretrained_competence = 0.0

        self.pretrained_plasticity_demand = 0.0
        self.pretrained_control_competence = 0.0

        self.pretrained_old_overlap_risk = 0.0

        self.effective_energy_coverage = 0.0
        self.effective_protect_strength = _BALANCED_STATIC_STRENGTH
        self.current_private_rank = self.rank
        self.pretrained_anchor_mode = False
        # 掩码可视化配置。
        self.dual_mask_vis = False
        self.dual_mask_vis_dir = "visualizations/dual_mask_snapshots"
        self.dual_mask_vis_layers = {0, 5, 11}
        self.dual_mask_vis_tasks = {0, 1}
        self.dual_mask_vis_save_weight = False
        self.lora_A_init = "kaiming"
        self.layer_idx = -1

    def _init_params(self, args):
        self.args = args
        self.use_slora: bool = args["use_slora"]
        self.use_plora: bool = args["use_plora"]
        msg = f'Use slora:{self.use_slora} and Use plora:{self.use_plora}'
        print(msg)
        logging.info(msg)

        # slora_gamma: S_lora 分支的缩放系数  0.5
        self.slora_gamma = float(args.get("slora_gamma", 1.0))
        # plora_gamma: P_lora 分支的缩放系数  0.75
        self.plora_gamma = float(args.get("plora_gamma", 1.0))
        # LoRA_output = slora_gamma * S_lora(x) + plora_gamma * P_lora(x)
        if self.use_slora and self.use_plora and args.get("avg", False):
            self.slora_gamma *= 0.5
            self.plora_gamma *= 0.5
        # svd 使用截断方向；soft_svd 对全部奇异方向连续加权。
        self.dual_mask_importance = str(args.get("dual_mask_importance", "svd")).lower()
        supported_importance_modes = {"svd", "soft_svd"}
        if self.dual_mask_importance not in supported_importance_modes:
            raise ValueError(
                "Unsupported dual_mask_importance={!r}. Choose from: {}.".format(
                    self.dual_mask_importance,
                    ", ".join(sorted(supported_importance_modes)),
                )
            )
        self.dual_mask_general_ratio = float(args.get("dual_mask_general_ratio", 0.5))  # 0.4

        self.dual_mask_layerwise_ratio_mode = str(
            args.get("dual_mask_layerwise_ratio_mode", "none")
        ).lower()
        if self.dual_mask_layerwise_ratio_mode not in {
            "none",
            "shallow_high",
            "deep_high",
        }:
            raise ValueError(
                "Unsupported dual_mask_layerwise_ratio_mode={!r}. Choose from: none, shallow_high, deep_high.".format(
                    self.dual_mask_layerwise_ratio_mode
                )
            )

        self.dual_mask_svd_rank = int(args.get("dual_mask_svd_rank", self.rank))  # 32
        self.dual_mask_svd_energy_coverage = float(args.get("dual_mask_svd_energy_coverage", 0.0))

        self.dual_mask_conflict_ratio = float(args.get("dual_mask_conflict_ratio", 0.25)) # Top-k 的比例参数  0.1
        self.dual_mask_conflict_strength = float(args.get("dual_mask_conflict_strength", 1.0))  # 冲突区压制多强  也就是beta

        self.dual_mask_conflict_reg_enabled = bool(args.get("dual_mask_conflict_reg_enabled", True))

        self.dual_mask_conflict_energy_adaptive = bool(args.get("dual_mask_conflict_energy_adaptive", False))

        self.dual_mask_conflict_energy_ratio_floor = bool(
            args.get("dual_mask_conflict_energy_ratio_floor", True)
        )

        self.dual_mask_conflict_old_overlap_adaptive = bool(args.get("dual_mask_conflict_old_overlap_adaptive", False))

        self.dual_mask_private_conflict_mode = str(args.get("dual_mask_private_conflict_mode", "global")).lower()
        supported_private_conflict_modes = {"global", "none", "plastic"}
        if self.dual_mask_private_conflict_mode not in supported_private_conflict_modes:
            raise ValueError("Unsupported dual_mask_private_conflict_mode={!r}. Choose from: {}.".format(self.dual_mask_private_conflict_mode,", ".join(sorted(supported_private_conflict_modes)),)
            )

        self.dual_mask_task0_gate_mode = str(args.get("dual_mask_task0_gate_mode", "full")).lower()
        self.dual_mask_s_protect_enabled = bool(args.get("dual_mask_s_protect_enabled", True))
        supported_task0_gate_modes = {"full", "protect_only", "unmasked"}
        if self.dual_mask_task0_gate_mode not in supported_task0_gate_modes:
            raise ValueError("Unsupported dual_mask_task0_gate_mode={!r}. Choose from: {}.".format(self.dual_mask_task0_gate_mode,", ".join(sorted(supported_task0_gate_modes)),))

        self.dual_mask_conflict_merge_mode = str(
            args.get("dual_mask_conflict_merge_mode", "suppress")
        ).lower()
        supported_conflict_merge_modes = {
            "none",
            "suppress",
            "relocate",
            "suppress_relocate",
        }
        if self.dual_mask_conflict_merge_mode not in supported_conflict_merge_modes:
            raise ValueError(
                "Unsupported dual_mask_conflict_merge_mode={!r}. Choose from: {}.".format(
                    self.dual_mask_conflict_merge_mode,
                    ", ".join(sorted(supported_conflict_merge_modes)),
                )
            )
        self.dual_mask_relocation_steps = max(
            1, int(args.get("dual_mask_relocation_steps", 20))
        )
        self.dual_mask_relocation_lr = float(
            args.get("dual_mask_relocation_lr", 0.1)
        )
        self.dual_mask_relocation_vectors = max(
            1, int(args.get("dual_mask_relocation_vectors", 64))
        )

        self.dual_mask_safe_residual_enabled = bool(
            args.get("dual_mask_safe_residual_enabled", False)
        )
        self.dual_mask_safe_residual_vectors = max(
            1, int(args.get("dual_mask_safe_residual_vectors", 64))
        )

        # 保护区的强度是否根据W0_competence自适应
        self.dual_mask_competence_adaptive = bool(args.get("dual_mask_competence_adaptive", False))
        self.dual_mask_plasticity_adaptive = bool(args.get("dual_mask_plasticity_adaptive", False))

        self.dual_mask_protect_strength_mode = str(args.get("dual_mask_protect_strength_mode", "legacy_linear")).lower()
        supported_protect_strength_modes = {"legacy_linear", "competence"}

        if self.dual_mask_protect_strength_mode not in supported_protect_strength_modes:
            raise ValueError(
                "Unsupported dual_mask_protect_strength_mode={!r}. Choose from: {}.".format(self.dual_mask_protect_strength_mode,", ".join(sorted(supported_protect_strength_modes)),)
            )
        # 固定保存初始预训练权重；自适应模式只改变覆盖率、强度和 private rank。
        self.capture_pretrained_anchor()
        self.set_pretrained_competence(0.0)

        # 可视化对应的超参数。
        self.dual_mask_vis = bool(args.get("dual_mask_vis", False))
        self.dual_mask_vis_dir = str(args.get("dual_mask_vis_dir", self.dual_mask_vis_dir))
        self.dual_mask_vis_layers = self._parse_vis_indices(args.get("dual_mask_vis_layers", [0, 5, 11]))
        self.dual_mask_vis_tasks = self._parse_vis_indices(args.get("dual_mask_vis_tasks", [0, 1]))
        self.dual_mask_vis_save_weight = bool(args.get("dual_mask_vis_save_weight", False))
        self.lora_A_init = str(args.get("lora_A_init", "orthogonal")).lower()
        logging.info(
            "Dual-mask branch: importance=%(importance)s, "
            "protect_ratio=%(protect_ratio).3f, svd_rank=%(svd_rank)s, "
            "conflict_strength=%(conflict_strength).3f, "
            "conflict_energy_adaptive=%(conflict_energy_adaptive)s, "
            "conflict_energy_ratio_floor=%(conflict_energy_ratio_floor)s, "
            "task0_gate_mode=%(task0_gate_mode)s, "
            "private_conflict_mode=%(private_conflict_mode)s, "
            "old_overlap_conflict_adaptive=%(old_overlap_conflict_adaptive)s, "
            "plasticity_adaptive=%(plasticity_adaptive)s, "
            "protect_strength_mode=%(protect_strength_mode)s, "
            "conflict_merge_mode=%(conflict_merge_mode)s, "
            "A_init=%(a_init)s",
            {
                "importance": self.dual_mask_importance,
                "protect_ratio": self.dual_mask_general_ratio,
                "svd_rank": self.dual_mask_svd_rank,
                "conflict_ratio": self.dual_mask_conflict_ratio,
                "conflict_strength": self.dual_mask_conflict_strength,
                "conflict_energy_adaptive": self.dual_mask_conflict_energy_adaptive,
                "conflict_energy_ratio_floor": self.dual_mask_conflict_energy_ratio_floor,
                "private_conflict_mode": self.dual_mask_private_conflict_mode,
                "task0_gate_mode": self.dual_mask_task0_gate_mode,
                "s_protect_enabled": self.dual_mask_s_protect_enabled,
                "layerwise_ratio_mode": self.dual_mask_layerwise_ratio_mode,
                "old_overlap_conflict_adaptive": self.dual_mask_conflict_old_overlap_adaptive,
                "plasticity_adaptive": self.dual_mask_plasticity_adaptive,
                "protect_strength_mode": self.dual_mask_protect_strength_mode,
                "conflict_merge_mode": self.dual_mask_conflict_merge_mode,
                "a_init": self.lora_A_init,
            },
        )

    def capture_pretrained_anchor(self, force: bool = False):
        if bool(self.pretrained_anchor_captured.item()) and not force:
            return  # 已经保存后面不再保存
        with torch.no_grad():
            self.pretrained_weight.copy_(self.qkv.weight.detach())  # W_pre = W0.clone() 永远不变的 W_pre
            self.pretrained_anchor_captured.fill_(True)

    def set_pretrained_anchor_mode(self, enabled: bool):
        self.pretrained_anchor_mode = bool(enabled)

    @contextmanager
    def use_pretrained_anchor(self):
        already_in_anchor = self.pretrained_anchor_mode
        if not already_in_anchor:  # 尚未处于 anchor 模式
            # 保存 W_current：W_pre + 历史安全增量
            accumulated_weight = self.qkv.weight.detach().clone()

            # 临时切回原始预训练权重 W_pre / W0
            self.set_pretrained_anchor_mode(True)
            with torch.no_grad():
                self.qkv.weight.copy_(self.pretrained_weight)
        try:
            yield  # 把控制权交给 with 里面的代码执行；等它执行完，再回来执行 finally
        finally:
            if not already_in_anchor:
                # 无论 with 内部正常结束还是报错，都恢复当前累计权重
                with torch.no_grad():
                    self.qkv.weight.copy_(accumulated_weight)
                self.set_pretrained_anchor_mode(False)

    def relative_weight_drift(self) -> float:
        anchor = self.pretrained_weight.detach().float()
        delta = self.qkv.weight.detach().float() - anchor
        # 衡量持续学习后 QKV 权重偏离原始预训练空间的相对幅度
        ## delta = W_current - W_pre  ,   drift = ||delta|| / ||W_pre||
        ## drift 小：当前模型仍接近 W_pre | drift 大：LoRA 合并后已明显偏离 W_pre
        return float((delta.norm() / anchor.norm().clamp_min(1e-12)).item())

    def set_pretrained_competence(
        self,
        competence: float,
        plasticity_demand: float = 0.0,
    ):
        competence = min(max(float(competence), 0.0), 1.0)

        plasticity_demand = min(max(float(plasticity_demand), 0.0), 1.0)
        control_competence = (
            competence * (1.0 - plasticity_demand)
            if self.dual_mask_plasticity_adaptive
            else competence
        )

        self.pretrained_competence = competence

        self.pretrained_plasticity_demand = plasticity_demand
        self.pretrained_control_competence = control_competence

        if self.dual_mask_competence_adaptive:  # 保护掩码相应的topk, 保护强度自适应
            ## W0 能力强：说明预训练空间已经适合当前任务，所以保护更多、压制更强  \ W0 能力弱：说明必须依赖 LoRA 学习，所以保护更少、压制更弱
            ### W0 competence 越高 → 越相信预训练空间 → 越偏向稳定性  \ W0 competence 越低 → 越需要任务学习 → 越偏向可塑性
            self.effective_energy_coverage = ( # 单调线性控制器
                _BALANCED_COVERAGE_BASE
                + _BALANCED_COVERAGE_SPAN * control_competence
            ) # 0.7 + 0.25 * ct
            if self.dual_mask_protect_strength_mode == "competence":
                self.effective_protect_strength = control_competence
            else:
                self.effective_protect_strength = (
                    _BALANCED_STRENGTH_BASE
                    + _BALANCED_STRENGTH_SPAN * control_competence
                )

            rank_ratio = 1.0 - control_competence * (
                1.0 - _BALANCED_PRIVATE_MIN_RATIO
            ) # 1 - ct*(1 - 0.25)
            self.current_private_rank = max(
                1,
                int(round(self.rank * rank_ratio)),
            )
        else:
            self.effective_energy_coverage = 0.0
            self.current_private_rank = self.rank

    def set_pretrained_old_overlap_risk(self, risk: float):
        self.pretrained_old_overlap_risk = min(max(float(risk), 0.0), 1.0)

    def set_functional_merge_strength(self, strength: Optional[float]):
        if strength is None:
            self._functional_merge_strength_override = None
            return
        self._functional_merge_strength_override = min(
            max(float(strength), 0.0),
            1.0,
        )

    @staticmethod
    def _parse_vis_indices(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in ("all", "*"):
                return None
            if not value:
                return set()
            return {int(item.strip()) for item in value.split(",") if item.strip()}
        if isinstance(value, int):
            return {int(value)}
        return {int(item) for item in value}

    def _should_save_dual_mask_snapshot(self, task: int) -> bool:
        if not self.dual_mask_vis:
            return False
        if self.dual_mask_vis_layers is not None and self.layer_idx not in self.dual_mask_vis_layers:
            return False
        if self.dual_mask_vis_tasks is not None and int(task) not in self.dual_mask_vis_tasks:
            return False
        return True

    def _save_dual_mask_snapshot(
            self,
            task: int,
            branch_deltas,
            conflict_ratio: Optional[float] = None,
            conflict_strength: Optional[float] = None,
    ):
        if not self._should_save_dual_mask_snapshot(task):
            return

        raw_delta = torch.stack([item["raw_delta"] for item in branch_deltas]).sum(dim=0)
        safe_delta = torch.stack([item["safe_delta"] for item in branch_deltas]).sum(dim=0)
        # conflict_score, conflict_mask = self._joint_conflict(raw_delta)
        conflict_score, conflict_mask = self._joint_conflict(
            raw_delta,
            conflict_ratio=conflict_ratio,
        )
        if conflict_ratio is None:
            conflict_ratio = self.dual_mask_conflict_ratio
        if conflict_strength is None:
            conflict_strength = self.dual_mask_conflict_strength

        seed = self.args.get("seed", "unknown")
        task_dir = os.path.join(
            self.dual_mask_vis_dir,
            "seed_{}".format(seed),
            "task_{:02d}".format(int(task)),
        )
        os.makedirs(task_dir, exist_ok=True)
        save_path = os.path.join(task_dir, "layer_{:02d}.pt".format(int(self.layer_idx)))

        payload = {
            "task": int(task),
            "layer": int(self.layer_idx),
            "seed": seed,
            "importance_mode": self.dual_mask_importance,
            "general_ratio": float(self.dual_mask_general_ratio),
            "conflict_ratio": float(conflict_ratio),
            "effective_conflict_ratio": float(conflict_mask.float().mean().item()),
            "protect_strength": float(self.effective_protect_strength),
            "conflict_strength": float(conflict_strength),

            "pretrained_competence": float(self.pretrained_competence),
            "pretrained_plasticity_demand": float(self.pretrained_plasticity_demand),
            "pretrained_control_competence": float(self.pretrained_control_competence),

            "private_rank": int(self.current_private_rank),
            "svd_rank": int(self.last_svd_rank),
            "svd_energy_coverage": float(self.last_svd_energy_coverage),
            "w0_importance": self.w0_importance.detach().cpu().float(),
            "general_mask": self.general_mask.detach().cpu().float(),
            "isolated_mask": self.isolated_mask.detach().cpu().float(),
            "conflict_score": conflict_score.detach().cpu().float(),
            "conflict_mask": conflict_mask.detach().cpu().float(),
            "raw_delta": raw_delta.detach().cpu().float(),
            "safe_delta": safe_delta.detach().cpu().float(),
            "branches": [
                {
                    "name": item["name"],
                    "isolated": bool(item["isolated"]),
                    "gamma": float(item["gamma"]),
                    "raw_delta": item["raw_delta"].detach().cpu().float(),
                    "safe_delta": item["safe_delta"].detach().cpu().float(),
                }
                for item in branch_deltas
            ],
        }
        if self.dual_mask_vis_save_weight:
            payload["pretrained_weight"] = self.pretrained_weight.detach().cpu().float()
            payload["accumulated_weight"] = self.qkv.weight.detach().cpu().float()

        torch.save(payload, save_path)
        logging.info("Saved dual-mask visualization snapshot: %s", save_path)

    def _init_A_weight(self, dim: int, rank: int, device, dtype) -> torch.Tensor:
        if self.lora_A_init in ("kaiming", "kaiming_uniform"):
            return _kaiming_A_init(dim, rank, device, dtype)
        if self.lora_A_init not in ("orthogonal", "qr"):
            logging.info("Unknown lora_A_init=%s; using orthogonal A init.", self.lora_A_init)
        return _random_fixed_A_init(dim, rank, device, dtype)

    def before_task(self, task: int):

        t = int(task)
        self.cur_task = t
        device = next(self.parameters()).device
        dtype = self.qkv.weight.dtype
        rs = self.rank # 64

        # init P_q / P_v
        A_rand_pq = _random_fixed_A_init(self.dim, rs, device, dtype)  # 随机QR分解正交初始化 A--Q  [rs,768]
        B_zero_pq = _zero_B_init(self.dim*3, rs, device, dtype) # 初始化 B 为全 0  # [768*3=2304,rs]
        self.S_lora[t] = FrozenA_TrainableB(self.dim, self.dim*3, rs, A_rand_pq, B_zero_pq, device=device, dtype=dtype)

        # freeze backbone  双重保险
        for p in self.qkv.parameters(): p.requires_grad_(False)
        for p in self.proj.parameters(): p.requires_grad_(False)
        if self.lora_A_init in ("kaiming", "kaiming_uniform"):
            with torch.no_grad():  # A 初始化正态分布
                self.S_lora[t].A.weight.copy_(
                    self._init_A_weight(self.dim, self.rank, device, dtype).to(
                        self.S_lora[t].A.weight.device,
                        dtype=self.S_lora[t].A.weight.dtype,
                    )
                )
                self.S_lora[t].B.weight.zero_()  # B初始化为0

        # 保留原始初始化顺序，避免改变同一 seed 后续任务的随机数轨迹
        p_rank = self.current_private_rank
        a_rand = self._init_A_weight(self.dim, p_rank, device, dtype)
        b_zero = _zero_B_init(self.dim * 3, p_rank, device, dtype)
        self.P_lora[t] = FrozenA_TrainableB(
            self.dim,
            self.dim * 3,
            p_rank,
            a_rand,
            b_zero,
            device=device,
            dtype=dtype,
        )

        self.rebuild_dual_masks()  # Dual masks rebuilt: W0 protect density 0.5000, plastic density 0.5000

    def _init_lora_weight(self, task, layer_idx:int=0):

        # Sequential baseline trains both A and B from a fresh initialization.
        if not self.use_plora and not self.use_slora: ## sequential tuning
            nn.init.kaiming_uniform_(self.S_lora[task].A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.S_lora[task].B.weight)

    def set_task_and_stage(self, task: int, layer_idx: int, stage: int = 0):
        task = int(task)
        ##################
        self.layer_idx = int(layer_idx)

        self.cur_task = task
        for p in self.qkv.parameters():
            p.requires_grad_(False)
        for p in self.proj.parameters():
            p.requires_grad_(False) # 把所有 attention head 的结果重新混合

        for unit in list(self.S_lora) + list(self.P_lora):
            if unit is None:
                continue
            unit.A.weight.requires_grad_(False)
            unit.B.weight.requires_grad_(False)

        if not self.use_slora and not self.use_plora:
            self.S_lora[task].A.weight.requires_grad_(True)
            self.S_lora[task].B.weight.requires_grad_(True)
            return

        if task == 0:
            self.S_lora[task].A.weight.requires_grad_(True)
            self.S_lora[task].B.weight.requires_grad_(True)
            return

        if self.use_slora:
            self.S_lora[task].B.weight.requires_grad_(True)
        if self.use_plora and self.P_lora[task] is not None:
            self.P_lora[task].B.weight.requires_grad_(True)

    def _soft_svd_importance(self, weight: torch.Tensor) -> torch.Tensor:
        """Use all singular directions with continuous energy weights."""
        weight_f = weight.detach().float()
        u, s, vh = torch.linalg.svd(weight_f, full_matrices=False)

        energy = s.clamp_min(0.0).pow(2)
        # 计算每个奇异方向的能量权重  p_l = s_l² / Σ_k s_k² 奇异值越大，该方向对 W0 越重要，权重 p_l 越大
        spectral_weights = energy / energy.sum().clamp_min(1e-12)
        row_score = (u.pow(2) * spectral_weights.unsqueeze(0)).sum(dim=1) # ^2:消除正负号
        col_score = (vh.t().pow(2) * spectral_weights.unsqueeze(0)).sum(dim=1)

        self.last_svd_rank = int(s.numel())
        self.last_svd_energy_coverage = 1.0
        # score[i,j] 越大，表示 W0[i,j] 所在行和列都更参与高能量奇异方向
        score = row_score.unsqueeze(1) * col_score.unsqueeze(0)
        return score.to(device=weight.device, dtype=weight.dtype)

    def _svd_importance(self, weight: torch.Tensor) -> torch.Tensor:
        weight_f = weight.detach().float()
        # s 里的值越大，说明对应方向越重要
        u, s, vh = torch.linalg.svd(weight_f, full_matrices=False)
        # 对 W0 的 QKV 权重做 SVD 时，保留多少个奇异方向来计算 W0 importance map
        k = _select_svd_rank(
            s,
            max_rank=self.dual_mask_svd_rank,
            energy_coverage=(self.dual_mask_svd_energy_coverage),
        )  # energy coverage 只能在配置的最大 rank 内选择
        self.last_svd_rank = int(k)
        energy = s.detach().float().pow(2)
        self.last_svd_energy_coverage = float((energy[:k].sum() / energy.sum().clamp_min(1e-12)).item())    # 0.53
        s_top = s[:k].clamp_min(0.0)

        row_score = (u[:, :k].pow(2) * s_top.unsqueeze(0)).sum(dim=1)
        col_score = (vh[:k, :].t().pow(2) * s_top.unsqueeze(0)).sum(dim=1)

        # 把行重要性和列重要性做外积，得到每个位置的综合重要性分数
        ## score[i, j] = row_score[i] * col_score[j]
        score = row_score.unsqueeze(1) * col_score.unsqueeze(0)
        return score.to(device=weight.device, dtype=weight.dtype)

    def _combined_importance(self) -> torch.Tensor:
        # 取当前 attention 层的 qkv.weight，detach() 表示不让这一步进入反向传播图。因为 mask 是一个启发式统计量，不需要通过它反传梯度
        weight = self.pretrained_weight.detach() # 永远不变的 W_pre

        mode = self.dual_mask_importance  # "dual_mask_importance": "svd"
        if mode == "soft_svd":
            svd_score = self._soft_svd_importance(weight) # 硬截断rank=32  覆盖率大概54%
        else:
            svd_score = self._svd_importance(weight) # 硬截断rank=32  覆盖率大概54%
        return svd_score

    def rebuild_dual_masks(self):
        with torch.no_grad():
            # SVD-only 的 W_pre 分数跨任务不变，可以复用；但每个任务仍按
            # 当前 competence 重新阈值化，使 adaptive coverage 真正生效
            reuse_w0_score = (
                self.cur_task > 0
                and self.dual_mask_importance in (
                    "svd",
                    "soft_svd",
                )
                and bool(torch.count_nonzero(self.w0_importance).item())
            )
            if reuse_w0_score:
                score = self.w0_importance.detach().clone()
            else:
                score = _normalize_score(self._combined_importance())

            # 根据重要性分数，取 top ratio 作为保护区
            ## score 最高的 50% 位置 -> protect = 1 | 1 表示这个位置是 W0 重要位置，不希望 LoRA 改
            ## score 剩下的 50% 位置 -> protect = 0 | 0 表示这个位置可以改
            coverage_mode = str(
                self.args.get("dual_mask_coverage_mode", "energy")
            ).strip().lower()
            use_adaptive_coverage = (
                self.dual_mask_competence_adaptive
                and coverage_mode == "energy"
            )
            if use_adaptive_coverage:
                # 使用能量覆盖，而不是固定 top ratio
                ## M_g = general_mask = Task 0 时 W_pre 的重要保护区
                mask_coverage = self.effective_energy_coverage
                protect = _energy_coverage_mask(score, mask_coverage)
            else:
                # 使用固定 top ratio
                mask_coverage = 0.0
                protect_ratio = self.dual_mask_general_ratio
                if self.dual_mask_layerwise_ratio_mode != "none":
                    # Keep the dataset-level ratio as the midpoint and move
                    # protection toward shallow/deep layers without adding
                    # another tunable endpoint.
                    depth = min(max(int(self.layer_idx), 0), 11) / 11.0
                    offset = 0.20 * (1.0 - 2.0 * depth)
                    if self.dual_mask_layerwise_ratio_mode == "deep_high":
                        offset = -offset
                    protect_ratio = min(max(protect_ratio + offset, 0.0), 1.0)
                protect = _top_ratio_mask(score, protect_ratio)  # mask
            # plastic[i, j] = 1 表示这个位置可以给 P_lora 使用
            # plastic[i, j] = 0 表示这个位置是保护区
            plastic = 1.0 - protect  # 可塑性区域

            self.w0_importance.copy_(score.to(device=self.w0_importance.device, dtype=self.w0_importance.dtype))
            self.general_mask.copy_(protect.to(device=self.general_mask.device, dtype=self.general_mask.dtype))  # 50% 位置被保护
            self.isolated_mask.copy_(plastic.to(device=self.isolated_mask.device, dtype=self.isolated_mask.dtype))  # 50% 位置可塑
            logging.info(
                "Dual masks rebuilt: layer %s, mask_coverage %.4f, "
                "W0 protect density %.4f, plastic density %.4f, "
                "protect_strength %.4f, protected_importance_mean %.4f, "
                "svd_rank %s, achieved_svd_energy %.4f",
                self.layer_idx,
                mask_coverage,
                self.general_mask.float().mean().item(),
                self.isolated_mask.float().mean().item(),
                self.effective_protect_strength,
                (self.w0_importance * self.general_mask).float().mean().item(),
                self.last_svd_rank,
                self.last_svd_energy_coverage,
            )

    def _conflict_parameters(self):
        """Return the fixed conflict range and optional old-overlap strength."""
        base_ratio = min(max(self.dual_mask_conflict_ratio, 0.0), 1.0)

        if self._functional_merge_strength_override is not None:
            return base_ratio, self._functional_merge_strength_override

        base_strength = min(max(self.dual_mask_conflict_strength, 0.0), 1.0)
        if self.dual_mask_conflict_old_overlap_adaptive:
            base_strength = min(base_strength * (1.0 + self.pretrained_old_overlap_risk),1.0,)
        return base_ratio, base_strength

    def _joint_conflict(
            self,
            delta: torch.Tensor,
            conflict_ratio: Optional[float] = None,
            valid_mask: Optional[torch.Tensor] = None,
    ):
        ## abs(delta[i, j]) 越大，说明 LoRA 越想修改这个位置
        ## ba_importance[i, j] 越大，表示 BA 在这个位置的改动越强   _normalize_score 归一化到大概 [0, 1]
        ba_importance = _normalize_score(delta.detach().abs())
        w0_importance = self.w0_importance.to(device=delta.device, dtype=delta.dtype)

        # 如果 W0 在这个位置很重要，并且 LoRA 也想大幅修改这个位置，那么这个位置就是高冲突位置
        # 联合分数只在 W_pre 重要且当前 BA 改动大的位置取高值。
        conflict_score = _normalize_score(w0_importance * ba_importance)
        # 把冲突分数最高的一部分位置标出来
        ## 比如配置 dual_mask_conflict_ratio = 0.25，就把冲突分数最高的 25% 位置标为 1，其他位置标为 0
        ## conflict_mask[i, j] = 1  表示这个位置是高冲突区域 conflict_mask[i, j] = 0  表示这个位置冲突不高
        ratio = self.dual_mask_conflict_ratio if conflict_ratio is None else conflict_ratio
        if self.dual_mask_conflict_energy_adaptive and float(ratio) > 0.0:
            if self.dual_mask_conflict_energy_ratio_floor:
                conflict_mask = _energy_coverage_with_ratio_floor_mask(
                    conflict_score,
                    ratio=ratio,
                    coverage=_CONFLICT_ENERGY_COVERAGE,
                    valid_mask=valid_mask,
                )
            else:
                conflict_mask = _energy_coverage_mask(
                    conflict_score,
                    coverage=_CONFLICT_ENERGY_COVERAGE,
                    valid_mask=valid_mask,
                )

        elif valid_mask is None:
            conflict_mask = _top_ratio_mask(conflict_score, ratio)
        else:
            conflict_mask = _masked_top_ratio_mask(
                conflict_score,
                valid_mask,
                ratio,
            )
        return conflict_score, conflict_mask

    def _effective_gate_mode(self) -> str:
        if self.cur_task == 0:
            return self.dual_mask_task0_gate_mode
        return "full"

    def _safe_delta(
            self,
            delta: torch.Tensor,
            isolated: bool,
            conflict_ratio: Optional[float] = None,
            conflict_strength: Optional[float] = None,
    ) -> torch.Tensor:

        gate_mode = self._effective_gate_mode()
        if gate_mode == "unmasked":
            return delta

        # general_mask/protect_mask: W0 重要区域，应该保护
        protect_mask = self.general_mask.to(device=delta.device, dtype=delta.dtype)
        # isolated_mask/plastic_mask: W0 非重要区域，允许 P_lora 使用
        plastic_mask = 1.0 - protect_mask
        protect_strength = min(max(self.effective_protect_strength, 0.0), 1.0)

        # This ablation disables only S-LoRA protection. P-LoRA still uses
        # plastic_mask below, so its behavior is unchanged.
        if isolated or not self.dual_mask_s_protect_enabled:
            protect_gate = torch.ones_like(protect_mask)
        else:
            # 保护区统一使用 competence-adaptive 强度，复现旧版平衡控制器。
            protect_gate = 1.0 - protect_strength * protect_mask

        private_conflict_disabled = (
            isolated and self.dual_mask_private_conflict_mode == "none"
        )
        if gate_mode == "protect_only" or private_conflict_disabled:
            conflict_gate = torch.ones_like(protect_gate)
        else:
            # conflict 高表示,W0 很重要，而且 LoRA 也想大幅修改这个位置
            _, conflict_mask = self._joint_conflict(
                delta,
                conflict_ratio=conflict_ratio,
                valid_mask=(
                    plastic_mask
                    if isolated and self.dual_mask_private_conflict_mode == "plastic"
                    else None
                ),
            )
            if conflict_strength is None:
                _, conflict_strength = self._conflict_parameters()
            conflict_strength = min(max(conflict_strength, 0.0), 1.0)
            # 高冲突区域按 conflict_strength 压制；其余区域保持不变。
            conflict_gate = 1.0 - conflict_strength * conflict_mask.to(delta.dtype)
        # Private LoRA 只使用非保护区；在二值互补 mask 下，
        # protect_gate * plastic_mask 恒等于 plastic_mask。

        if isolated:
            gate = plastic_mask * conflict_gate
        else:
            gate = protect_gate * conflict_gate  # [2304,768]
        return delta * gate

    def _merge_base_and_conflict(
            self,
            delta: torch.Tensor,
            isolated: bool,
            conflict_ratio: float,
            compute_conflict: bool = True,
    ):
        """Return the pre-conflict update and its conflict mask."""
        gate_mode = self._effective_gate_mode()
        if gate_mode == "unmasked":
            return delta, torch.zeros_like(delta)

        protect_mask = self.general_mask.to(device=delta.device, dtype=delta.dtype)
        plastic_mask = 1.0 - protect_mask
        protect_strength = min(max(self.effective_protect_strength, 0.0), 1.0)
        if isolated:
            base_delta = delta * plastic_mask
        elif self.dual_mask_s_protect_enabled:
            base_delta = delta * (1.0 - protect_strength * protect_mask)
        else:
            base_delta = delta

        private_conflict_disabled = (
            isolated and self.dual_mask_private_conflict_mode == "none"
        )
        if gate_mode == "protect_only" or private_conflict_disabled:
            return base_delta, torch.zeros_like(delta)

        _, conflict_mask = self._joint_conflict(
            delta,
            conflict_ratio=conflict_ratio,
            valid_mask=(
                plastic_mask
                if isolated and self.dual_mask_private_conflict_mode == "plastic"
                else None
            ),
        )
        return base_delta, conflict_mask.to(dtype=delta.dtype)

    def _compose_merge_delta(
            self,
            raw_delta: torch.Tensor,
            isolated: bool,
            conflict_ratio: float,
            conflict_strength: float,
            relocation_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compose one branch update according to the merge-only ablation."""
        mode = self.dual_mask_conflict_merge_mode
        if mode == "suppress":
            return self._safe_delta(
                raw_delta,
                isolated=isolated,
                conflict_ratio=conflict_ratio,
                conflict_strength=conflict_strength,
            )

        base_delta, conflict_mask = self._merge_base_and_conflict(
            raw_delta,
            isolated=isolated,
            conflict_ratio=conflict_ratio,
        )
        if mode == "none":
            return base_delta

        if relocation_delta is None:
            raise RuntimeError(
                "Conflict relocation must be prepared before after_task()."
            )
        if mode == "relocate":
            return base_delta * (1.0 - conflict_mask) + relocation_delta
        if mode == "suppress_relocate":
            suppressed = self._safe_delta(
                raw_delta,
                isolated=isolated,
                conflict_ratio=conflict_ratio,
                conflict_strength=conflict_strength,
            )
            return suppressed + relocation_delta
        raise RuntimeError(f"Unexpected conflict merge mode: {mode}")

    def _fit_low_rank_relocation(
            self,
            unit,
            gamma: float,
            target_delta: torch.Tensor,
            safe_support: torch.Tensor,
            inputs: torch.Tensor,
    ):
        """Fit rank-wise coefficients that reproduce target activations safely."""
        device = target_delta.device
        fit_dtype = torch.float32
        x = inputs.detach().reshape(-1, inputs.shape[-1]).to(device=device, dtype=fit_dtype)
        if x.shape[0] > self.dual_mask_relocation_vectors:
            indices = torch.linspace(
                0,
                x.shape[0] - 1,
                steps=self.dual_mask_relocation_vectors,
                device=device,
            ).long()
            x = x.index_select(0, indices)

        target = target_delta.detach().to(dtype=fit_dtype)
        support = safe_support.detach().to(device=device, dtype=fit_dtype)
        target_output = F.linear(x, target)
        target_energy = target_output.pow(2).mean()

        zero = torch.zeros_like(target_delta)
        if target_energy <= 1e-12 or support.count_nonzero() == 0:
            return zero, {
                "target_energy": float(target_energy.item()),
                "recovered_energy": 0.0,
                "activation_error": 0.0 if target_energy <= 1e-12 else 1.0,
            }

        a = unit.A_weight.detach().to(device=device, dtype=fit_dtype)
        b = unit.B_weight.detach().to(device=device, dtype=fit_dtype)
        coefficients = torch.zeros(a.shape[0], device=device, dtype=fit_dtype, requires_grad=True)
        optimizer = torch.optim.Adam(
            [coefficients],
            lr=self.dual_mask_relocation_lr,
        )
        with torch.enable_grad():
            for _ in range(self.dual_mask_relocation_steps):
                candidate = (
                    float(gamma) * ((b * coefficients.unsqueeze(0)) @ a)
                    * support
                )
                # Optimize relative functional error. The real conflict residual
                # is often only 1e-10--1e-7 in activation energy; raw MSE makes
                # Adam's epsilon dominate and leaves the relocation at zero.
                loss = (
                    F.mse_loss(F.linear(x, candidate), target_output)
                    / target_energy.detach().clamp_min(1e-12)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            relocated = (
                float(gamma)
                * ((b * coefficients.detach().unsqueeze(0)) @ a)
                * support
            )
            residual_energy = F.mse_loss(F.linear(x, relocated), target_output)
            relative_error = torch.sqrt(
                residual_energy / target_energy.clamp_min(1e-12)
            )
            recovered = (
                1.0 - residual_energy / target_energy.clamp_min(1e-12)
            ).clamp(0.0, 1.0)
        return relocated.to(dtype=target_delta.dtype), {
            "target_energy": float(target_energy.item()),
            "recovered_energy": float(recovered.item()),
            "activation_error": float(relative_error.item()),
        }

    def begin_relocation_input_collection(self) -> None:
        self._relocation_input_collection = []

    def _record_relocation_inputs(self, inputs: torch.Tensor) -> None:
        if self._relocation_input_collection is None:
            return
        collected = sum(item.shape[0] for item in self._relocation_input_collection)
        remaining = self.dual_mask_relocation_vectors - collected
        if remaining <= 0:
            return
        flattened = inputs.detach().reshape(-1, inputs.shape[-1])
        if flattened.shape[0] > remaining:
            indices = torch.linspace(
                0,
                flattened.shape[0] - 1,
                steps=remaining,
                device=flattened.device,
            ).long()
            flattened = flattened.index_select(0, indices)
        self._relocation_input_collection.append(flattened)

    def end_relocation_input_collection(self) -> torch.Tensor:
        collected = self._relocation_input_collection
        self._relocation_input_collection = None
        if not collected:
            return self.qkv.weight.new_zeros((0, self.dim))
        return torch.cat(collected, dim=0)

    def prepare_conflict_relocation(
            self,
            task: int,
            inputs: torch.Tensor,
    ) -> None:
        """Fit the merge-time relocation using current-task train-only inputs."""
        self._pending_relocations = {}
        mode = self.dual_mask_conflict_merge_mode
        if mode not in {"relocate", "suppress_relocate"}:
            return

        t = int(task)
        branches = []
        if not self.use_slora and not self.use_plora:
            branches.append(("S", self.S_lora[t], 1.0, False))
        else:
            if self.use_slora or t == 0:
                branches.append(("S", self.S_lora[t], float(self.slora_gamma), False))
            if t > 0 and self.use_plora and self.P_lora[t] is not None:
                branches.append(("P", self.P_lora[t], float(self.plora_gamma), True))

        conflict_ratio, conflict_strength = self._conflict_parameters()
        plastic_mask = (1.0 - self.general_mask).to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        metrics = []
        for name, unit, gamma, isolated in branches:
            raw_delta = gamma * (unit.B_weight.detach() @ unit.A_weight.detach())
            base_delta, conflict_mask = self._merge_base_and_conflict(
                raw_delta,
                isolated=isolated,
                conflict_ratio=conflict_ratio,

            )
            residual_scale = 1.0 if mode == "relocate" else conflict_strength
            target_delta = base_delta * conflict_mask * residual_scale
            safe_support = plastic_mask * (1.0 - conflict_mask)
            relocated, branch_metrics = self._fit_low_rank_relocation(
                unit,
                gamma=gamma,
                target_delta=target_delta,
                safe_support=safe_support,
                inputs=inputs,
            )
            self._pending_relocations[name] = relocated.detach()
            metrics.append(branch_metrics)

        if metrics:
            target_energy = sum(item["target_energy"] for item in metrics)
            recovered_energy = sum(item["recovered_energy"] for item in metrics) / len(metrics)
            activation_error = sum(item["activation_error"] for item in metrics) / len(metrics)
        else:
            target_energy = recovered_energy = activation_error = 0.0
        with torch.no_grad():
            self.last_relocation_target_energy.fill_(target_energy)
            self.last_relocation_recovered_energy.fill_(recovered_energy)
            self.last_relocation_activation_error.fill_(activation_error)
        logging.info(
            "Task %s layer %s conflict relocation: mode=%s, target_energy=%.6e, "
            "recovered_energy=%.4f, activation_error=%.4f",
            t,
            self.layer_idx,
            mode,
            target_energy,
            recovered_energy,
            activation_error,
        )

    def _masked_unit_forward(
            self,
            x: torch.Tensor,
            unit,
            isolated: bool,
            residual_scale: float = 1.0,
    ) -> torch.Tensor:
        raw_delta = unit.B_weight @ unit.A_weight
        ## isolated=True: safe_delta_p = BA_p * plastic_mask * conflict_gate
        ## isolated=False: safe_delta_s = BA_s * protect_gate * conflict_gate
        safe_delta = self._safe_delta(raw_delta, isolated=isolated)
        if (
                self.dual_mask_safe_residual_enabled
                and self.training
                and torch.is_grad_enabled()
        ):
            base_delta, _ = self._merge_base_and_conflict(
                raw_delta,
                isolated=isolated,
                conflict_ratio=self._conflict_parameters()[0],
                compute_conflict=False,
            )
            self._pending_safe_residual_deltas.append(
                residual_scale * (base_delta - safe_delta)
            )
        return F.linear(x, safe_delta)  # 输出 = x @ safe_delta.T

    def _finalize_safe_residual(self, x: torch.Tensor):
        if not self._pending_safe_residual_deltas:
            return
        sampled_inputs = x.detach().reshape(-1, x.shape[-1])
        if sampled_inputs.shape[0] > self.dual_mask_safe_residual_vectors:
            indices = torch.linspace(
                0,
                sampled_inputs.shape[0] - 1,
                steps=self.dual_mask_safe_residual_vectors,
                device=sampled_inputs.device,
            ).long()
            sampled_inputs = sampled_inputs.index_select(0, indices)
        residual_delta = torch.stack(
            self._pending_safe_residual_deltas
        ).sum(dim=0)
        residual_output = F.linear(sampled_inputs, residual_delta)
        input_energy = (
            sampled_inputs.float().pow(2).sum(dim=-1).mean()
            .detach()
            .clamp_min(1e-12)
        )
        self._last_safe_residual_loss = (
            residual_output.float().pow(2).sum(dim=-1).mean()
            / input_energy
        )
        self._pending_safe_residual_deltas = []

    def safe_residual_regularization(self):
        loss = self._last_safe_residual_loss
        self._last_safe_residual_loss = None
        return loss

    def anchor_regularization(self) -> torch.Tensor:
        """Penalize the effective current QKV weight drifting from W_pre."""
        task = int(self.cur_task)
        current_delta = torch.zeros_like(self.qkv.weight)

        if not self.use_slora and not self.use_plora:
            unit = self.S_lora[task]
            if unit is not None:
                raw_delta = unit.B_weight @ unit.A_weight
                current_delta = current_delta + self._safe_delta(
                    raw_delta,
                    isolated=False,
                )
        else:
            unit_s = self.S_lora[task]
            if unit_s is not None and (self.use_slora or task == 0):
                raw_delta_s = self.slora_gamma * (
                    unit_s.B_weight @ unit_s.A_weight
                )
                current_delta = current_delta + self._safe_delta(
                    raw_delta_s,
                    isolated=False,
                )

            unit_p = self.P_lora[task]
            if task > 0 and self.use_plora and unit_p is not None:
                raw_delta_p = self.plora_gamma * (
                    unit_p.B_weight @ unit_p.A_weight
                )
                current_delta = current_delta + self._safe_delta(
                    raw_delta_p,
                    isolated=True,
                )

        anchor = self.pretrained_weight.detach().float()
        effective_weight = self.qkv.weight.detach().float() + current_delta.float()
        drift = effective_weight - anchor
        return drift.pow(2).sum() / anchor.pow(2).sum().clamp_min(1e-12)

    def _joint_conflict_regularization(self, unit, isolated: bool) -> torch.Tensor:
        delta = unit.B_weight @ unit.A_weight # ΔW = 0 × A = 0

        gate_mode = self._effective_gate_mode()
        if gate_mode == "unmasked":
            return delta.sum() * 0.0

        safe_delta = self._safe_delta(delta, isolated=isolated)
        w0_importance = self.w0_importance.to(device=delta.device, dtype=delta.dtype)
        # 如果某个位置 W0 很重要，那么 safe_delta 在这个位置越大，惩罚越大
        protection = (w0_importance * safe_delta.pow(2)).mean()
        if gate_mode == "protect_only":
            return protection

        if isolated and self.dual_mask_private_conflict_mode == "none":
            return protection

        conflict_score, _ = self._joint_conflict(
            delta,
            valid_mask=(
                self.isolated_mask
                if isolated and self.dual_mask_private_conflict_mode == "plastic"
                else None
            ),
        )

        # 如果某个位置 conflict_score 高，那么 safe_delta 在这个位置越大，惩罚越大
        conflict = (conflict_score.detach() * safe_delta.pow(2)).mean()
        if self.dual_mask_conflict_reg_enabled:
            return protection + conflict
        return protection

    @staticmethod
    def _delta_stats(raw_delta: torch.Tensor, safe_delta: torch.Tensor):
        raw = raw_delta.detach().float()
        safe = safe_delta.detach().float()
        raw_norm = raw.norm().item()
        safe_norm = safe.norm().item()
        if raw_norm <= 1e-12:
            suppressed_ratio = 0.0
        else:
            suppressed_ratio = 1.0 - safe_norm / raw_norm
        q_safe, k_safe, v_safe = safe.chunk(3, dim=0)
        return {
            "raw_norm": raw_norm,
            "safe_norm": safe_norm,
            "suppressed_ratio": suppressed_ratio,
            "raw_abs_mean": raw.abs().mean().item(),
            "safe_abs_mean": safe.abs().mean().item(),
            "max_abs": raw.abs().max().item(),
            "q_safe_norm": q_safe.norm().item(),
            "k_safe_norm": k_safe.norm().item(),
            "v_safe_norm": v_safe.norm().item(),
        }

    @staticmethod
    def _format_delta_stats(name: str, stats) -> str:
        return (
            "{} raw_norm={:.6f}, safe_norm={:.6f}, suppressed={:.2%}, "
            "raw_abs_mean={:.3e}, safe_abs_mean={:.3e}, max_abs={:.3e}"
        ).format(
            name,
            stats["raw_norm"],
            stats["safe_norm"],
            stats["suppressed_ratio"],
            stats["raw_abs_mean"],
            stats["safe_abs_mean"],
            stats["max_abs"],
        )

    @staticmethod
    def _conflict_distribution_stats(
            conflict_score: torch.Tensor,
            conflict_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized conflict entropy and the selected-score energy."""
        flat_score = conflict_score.detach().float().flatten().clamp_min(0.0)
        total = flat_score.sum()
        if total <= 0.0:
            zero = flat_score.new_zeros(())
            return zero, zero

        probability = flat_score / total
        positive = probability > 0.0
        entropy = -(probability[positive] * probability[positive].log()).sum()
        if flat_score.numel() > 1:
            entropy = entropy / torch.log(
                flat_score.new_tensor(float(flat_score.numel()))
            )
        else:
            entropy = flat_score.new_zeros(())
        top_energy = (
                flat_score[conflict_mask.detach().bool().flatten()].sum() / total
        )
        return entropy, top_energy

    @staticmethod
    def _conflict_gate_suppression(
            raw_delta: torch.Tensor,
            conflict_mask: torch.Tensor,
            conflict_strength: float,
    ) -> float:
        """Measure suppression caused by the current conflict gate alone."""
        conflict_gate = 1.0 - conflict_strength * conflict_mask.to(raw_delta.dtype)
        return Attention_LoRA._delta_stats(
            raw_delta,
            raw_delta * conflict_gate,
        )["suppressed_ratio"]

    @staticmethod
    def _conflict_energy50_ratio(conflict_score: torch.Tensor) -> torch.Tensor:
        """Return the smallest coordinate ratio covering 50% conflict energy."""
        flat_score = conflict_score.detach().float().flatten().clamp_min(0.0)
        total = flat_score.sum()
        if total <= 0.0:
            return flat_score.new_zeros(())

        values, _ = torch.sort(flat_score, descending=True)
        cumulative = torch.cumsum(values, dim=0)
        k = int(torch.searchsorted(cumulative, 0.5 * total).item()) + 1
        return flat_score.new_tensor(k / flat_score.numel())

    def _log_merge_stats(
            self,
            task: int,
            branch_deltas,
            conflict_ratio: Optional[float] = None,
            conflict_strength: Optional[float] = None,
    ):
        raw_total = torch.stack([item["raw_delta"] for item in branch_deltas]).sum(dim=0)
        safe_total = torch.stack([item["safe_delta"] for item in branch_deltas]).sum(dim=0)
        total_stats = self._delta_stats(raw_total, safe_total)
        # _, conflict_mask = self._joint_conflict(raw_total)
        # conflict_score, conflict_mask = self._joint_conflict(raw_total)
        conflict_score, conflict_mask = self._joint_conflict(
            raw_total,
            conflict_ratio=conflict_ratio,
        )

        effective_conflict_ratio = conflict_mask.detach().float().mean().item()

        protect_mask = self.general_mask.detach().float()

        fixed_conflict_mask = _top_ratio_mask(conflict_score,self.dual_mask_conflict_ratio if conflict_ratio is None else conflict_ratio,)

        conflict_entropy, conflict_top10_energy = self._conflict_distribution_stats(
            conflict_score,
            fixed_conflict_mask,
        )

        conflict_energy50_ratio = self._conflict_energy50_ratio(conflict_score)

        if conflict_ratio is None:
            conflict_ratio = self.dual_mask_conflict_ratio
        if conflict_strength is None:
            conflict_strength = self.dual_mask_conflict_strength
        conflict_strength = min(max(conflict_strength, 0.0), 1.0)

        conflict_gate_suppression = self._conflict_gate_suppression(
            raw_total,
            conflict_mask,
            conflict_strength,
        )

        private_mask_overlap = raw_total.new_zeros(())
        private_energy_overlap = raw_total.new_zeros(())
        private_gate_suppression = 0.0
        private_item = next(
            (item for item in branch_deltas if item["isolated"]),
            None,
        )
        if private_item is not None:
            private_raw = private_item["raw_delta"]
            plastic_mask = (1.0 - protect_mask).to(
                device=private_raw.device,
                dtype=private_raw.dtype,
            )
            private_score, global_private_mask = self._joint_conflict(
                private_raw,
                conflict_ratio=conflict_ratio,
            )
            selected_count = global_private_mask.detach().float().sum()
            if selected_count > 0.0:
                private_mask_overlap = (
                    global_private_mask.detach().float() * plastic_mask.float()
                ).sum() / selected_count
            selected_energy = (
                private_score.detach().float()
                * global_private_mask.detach().float()
            )
            if selected_energy.sum() > 0.0:
                private_energy_overlap = (
                    selected_energy * plastic_mask.float()
                ).sum() / selected_energy.sum()

            if self.dual_mask_private_conflict_mode == "none":
                actual_private_mask = torch.zeros_like(global_private_mask)
            elif self.dual_mask_private_conflict_mode == "plastic":
                _, actual_private_mask = self._joint_conflict(
                    private_raw,
                    conflict_ratio=conflict_ratio,
                    valid_mask=plastic_mask,
                )
            else:
                actual_private_mask = global_private_mask
            private_plastic_delta = private_raw * plastic_mask
            private_gate_suppression = self._conflict_gate_suppression(
                private_plastic_delta,
                actual_private_mask,
                conflict_strength,
            )


        with torch.no_grad():
            self.last_conflict_entropy.copy_(conflict_entropy)
            self.last_conflict_top10_energy.copy_(conflict_top10_energy)
            self.last_conflict_energy50_ratio.copy_(conflict_energy50_ratio)
            self.last_conflict_gate_suppression.fill_(conflict_gate_suppression)
            self.last_safe_suppression.fill_(total_stats["suppressed_ratio"])

            self.last_effective_conflict_ratio.fill_(effective_conflict_ratio)
            self.last_effective_conflict_strength.fill_(conflict_strength)

            self.last_private_conflict_mask_overlap.copy_(private_mask_overlap)
            self.last_private_conflict_energy_overlap.copy_(private_energy_overlap)
            self.last_private_conflict_gate_suppression.fill_(private_gate_suppression)

        branch_stats = []
        for item in branch_deltas:
            stats = self._delta_stats(item["raw_delta"], item["safe_delta"])
            branch_stats.append(self._format_delta_stats(item["name"], stats))

        logging.info(
            "Task %s layer %s LoRA branch stats: %s",
            int(task),
            int(self.layer_idx),
            " | ".join(branch_stats),
        )
        logging.info(
            "Task %s layer %s dual-mask merge: total_raw_norm=%.6f, "
            "total_safe_norm=%.6f, suppressed=%.2f%%, raw_abs_mean=%.3e, "
            "safe_abs_mean=%.3e, max_abs=%.3e, protect_density=%.4f, "
            "plastic_density=%.4f, conflict_density=%.4f, "
            "conflict_entropy=%.4f, conflict_top10_energy=%.4f, "
            "conflict_energy50_ratio=%.4f, "
            "conflict_gate_suppressed=%.2f%%, "
            "effective_conflict_ratio=%.4f, "
            "effective_conflict_strength=%.4f, "
            "private_conflict_mode=%s, "
            "private_conflict_mask_overlap=%.4f, "
            "private_conflict_energy_overlap=%.4f, "
            "private_conflict_gate_suppressed=%.2f%%, "
            "Q_safe_norm=%.6f, K_safe_norm=%.6f, V_safe_norm=%.6f",
            int(task),
            int(self.layer_idx),
            total_stats["raw_norm"],
            total_stats["safe_norm"],
            total_stats["suppressed_ratio"] * 100.0,
            total_stats["raw_abs_mean"],
            total_stats["safe_abs_mean"],
            total_stats["max_abs"],
            protect_mask.mean().item(),
            (1.0 - protect_mask).mean().item(),
            conflict_mask.float().mean().item(),
            conflict_entropy.item(),
            conflict_top10_energy.item(),
            conflict_energy50_ratio.item(),
            conflict_gate_suppression * 100.0,
            # float(conflict_ratio),
            effective_conflict_ratio,
            conflict_strength,
            self.dual_mask_private_conflict_mode,
            private_mask_overlap.item(),
            private_energy_overlap.item(),
            private_gate_suppression * 100.0,
            total_stats["q_safe_norm"],
            total_stats["k_safe_norm"],
            total_stats["v_safe_norm"],
        )

    def _contrib_from_units(self, x: torch.Tensor, t_idx: int) -> torch.Tensor:

        self._pending_safe_residual_deltas = []
        self._last_safe_residual_loss = None

        self._record_relocation_inputs(x)

        zero_output = x.new_zeros((*x.shape[:-1], self.dim * 3))

        if self.pretrained_anchor_mode:
            return zero_output

        unit_s = self.S_lora[t_idx]  # S_lora[t_idx] = 当前 task 的共享 LoRA
        unit_p = self.P_lora[t_idx]  # P_lora[t_idx] = 当前 task 的私有/隔离 LoRA

        # 当前任务已经 merge，后续只使用写入 qkv.weight 的增量。
        if unit_s is None and unit_p is None:
            return zero_output

        if not self.use_slora and not self.use_plora:
            if unit_s is None:
                return zero_output
            out = self._masked_unit_forward(x, unit_s, isolated=False)
            self._finalize_safe_residual(x)
            return out

        slora_gamma = float(self.slora_gamma)
        plora_gamma = float(self.plora_gamma)
        out = zero_output

        if unit_s is not None and (self.use_slora or t_idx == 0):
            out = out + slora_gamma * self._masked_unit_forward(
                x,
                unit_s,
                isolated=False,
                residual_scale=slora_gamma,
            )

        if t_idx > 0 and self.use_plora and unit_p is not None:
            ## P_lora 只能在 W0 非重要区域更新
            out = out + plora_gamma * self._masked_unit_forward(
                x,
                unit_p,
                isolated=True,
                residual_scale=plora_gamma,
            )

        self._finalize_safe_residual(x)
        return out

    def forward(self, x: torch.Tensor, task: int, register_hook: bool = False,
                get_feat: bool = False, get_cur_feat: bool = False):

        Bsz, N, C = x.shape
        qkv:torch.Tensor = self.qkv(x) + self._contrib_from_units(x, task) # y=W0x+ΔWx
        qkv:torch.Tensor = qkv.reshape(Bsz, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(Bsz, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def after_task(self, task: int):
        t = int(task)
        device = next(self.parameters()).device
        dtype = self.qkv.weight.dtype

        def raw_delta(name: str, unit, gamma: float, isolated: bool):
            delta = gamma * (unit.B_weight.detach() @ unit.A_weight.detach())
            return {
                "name": name,
                "isolated": isolated,
                "gamma": gamma,
                "raw_delta": delta,
            }

        def mask_delta(
                item,
                conflict_ratio: float,
                conflict_strength: float,
        ):
            safe_delta = self._compose_merge_delta(
                item["raw_delta"],
                isolated=item["isolated"],
                conflict_ratio=conflict_ratio,
                conflict_strength=conflict_strength,
                relocation_delta=self._pending_relocations.get(item["name"]),
            )
            if not torch.isfinite(safe_delta).all():
                raise RuntimeError(
                    f"Task {t} {item['name']} branch produced a non-finite masked LoRA delta"
                )
            item["safe_delta"] = safe_delta

        branch_deltas = []
        if not self.use_slora and not self.use_plora:
            branch_deltas.append(
                raw_delta("S", self.S_lora[t], 1.0, isolated=False)
            )
        else:
            if self.use_slora or t == 0:
                branch_deltas.append(
                    raw_delta(
                        "S",
                        self.S_lora[t],
                        float(self.slora_gamma),
                        isolated=False,
                    )
                )
            if t > 0 and self.use_plora and self.P_lora[t] is not None:
                branch_deltas.append(
                    raw_delta(
                        "P",
                        self.P_lora[t],
                        float(self.plora_gamma),
                        isolated=True,
                    )
                )

        if branch_deltas:
            conflict_ratio, conflict_strength = self._conflict_parameters()

            self.last_functional_merge_strength = float(conflict_strength)

            for item in branch_deltas:
                mask_delta(item, conflict_ratio, conflict_strength)

            self._save_dual_mask_snapshot(
                t,
                branch_deltas,
                conflict_ratio=conflict_ratio,
                conflict_strength=conflict_strength,
            )

            #########################
            self._log_merge_stats(
                t,
                branch_deltas,
                conflict_ratio=conflict_ratio,
                conflict_strength=conflict_strength,
            )
            with torch.no_grad():
                delta = torch.stack([item["safe_delta"] for item in branch_deltas]).sum(dim=0)
                self.qkv.weight.add_(delta.to(device, dtype))

            # safe_delta 已经永久写入 qkv.weight；旧 A/B 后续不再参与前向。
            self.S_lora[t] = None
            self.P_lora[t] = None
            self._pending_relocations = {}
            self._functional_merge_strength_override = None

        return None
