import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
from contextlib import contextmanager

from models.decomposed_lora import Attention_LoRA as BaseAttentionLoRA
from models.decomposed_lora import FrozenA_TrainableB
from models.decomposed_lora import _kaiming_A_init, _random_fixed_A_init, _zero_B_init


# 固定的跨数据集平衡策略。它复现旧版默认控制器，但不再暴露为独立超参数。
_BALANCED_COVERAGE_BASE = 0.70
_BALANCED_COVERAGE_SPAN = 0.25
_BALANCED_STRENGTH_BASE = 0.30
_BALANCED_STRENGTH_SPAN = 0.60
_BALANCED_PRIVATE_MIN_RATIO = 0.25
_BALANCED_STATIC_STRENGTH = 0.70


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


def _energy_coverage_mask(score: torch.Tensor, coverage: float) -> torch.Tensor:
    """根据传入的参数重要性分数 score[i,j], 选取数值最大的coverage权重"""
    coverage = min(max(float(coverage), 0.0), 1.0)
    flat = score.detach().float().flatten().clamp_min(0.0)
    total = flat.sum()
    if coverage <= 0.0 or total <= 0.0:
        return torch.zeros_like(score)
    if coverage >= 1.0:
        return torch.ones_like(score)

    values, indices = torch.sort(flat, descending=True)
    cumulative = torch.cumsum(values, dim=0)
    k = int(torch.searchsorted(cumulative, coverage * total).item()) + 1
    selected = torch.zeros_like(flat)
    selected[indices[:k]] = 1.0
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


class Attention_LoRA(BaseAttentionLoRA):
    """
        带W0保护与BA冲突控制的双掩码LoRA分支
        第一个掩码标记了重要的预训练权重 W₀ 方向，这些方向在训练中应受到保护。
        第二个掩码标记了重要性较低的塑性区域，在这些区域中，独立的 BA 更新可以更自由地移动。
        两者之间的交互作用通过一个轻量级的联合得分 normalize(I_W₀) × normalize(I_BA) 进行近似估计
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (self.dim * 3, self.dim)

        self.P_lora = torch.nn.ModuleList([None for _ in range(self.n_tasks)])
        """
            W0 = qkv.weight 每个位置的重要性分数 [3 * dim, dim]  | w0_importance[i, j] 越大，说明 W0 这个位置越重要，越不希望 LoRA 改它
            w0_importance        W0 每个位置的重要性分数
            general_mask         W0 重要区域，保护用
            isolated_mask        W0 非重要区域，给 P_lora 的 plastic 区域
        """
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

        self.register_buffer("pretrained_weight", torch.zeros(shape), persistent=True)
        self.register_buffer("pretrained_anchor_captured", torch.tensor(False, dtype=torch.bool), persistent=True)


        ### 先判断 W0 哪里重要，再决定 LoRA 的 BA 哪里能加、哪里不能加
        self.dual_mask_importance = "svd"
        self.dual_mask_general_ratio = 0.5  # 决定 W0 保护区多大

        self.dual_mask_svd_rank = self.rank  # 决定 SVD 重要性用多少主方向
        self.last_svd_rank = self.rank

        self.last_svd_energy_coverage = 0.0

        self.dual_mask_conflict_ratio = 0.25  # 决定 BA-W0 冲突区多大
        self.dual_mask_conflict_strength = 1.0  # 决定冲突区压制多强

        self.dual_mask_conflict_reg_enabled = True

        # 冲突区域的重叠是否自适应
        self.dual_mask_conflict_old_overlap_adaptive = False

        ## SVD 和梯度重要性混合的参数
        self.dual_mask_svd_energy_coverage = 0.0
        self.dual_mask_competence_adaptive = False

        self.dual_mask_plasticity_adaptive = False

        self.pretrained_competence = 0.0

        self.pretrained_plasticity_demand = 0.0
        self.pretrained_control_competence = 0.0

        self.pretrained_old_overlap_risk = 0.0

        self.effective_energy_coverage = 0.0
        self.effective_protect_strength = _BALANCED_STATIC_STRENGTH
        self.current_private_rank = self.rank
        self.pretrained_anchor_mode = False
        ###############################

        ################################# 掩码可视化操作
        self.dual_mask_vis = False
        self.dual_mask_vis_dir = "visualizations/dual_mask_snapshots"
        self.dual_mask_vis_layers = {0, 5, 11}
        self.dual_mask_vis_tasks = {0, 1}
        self.dual_mask_vis_save_weight = False
        self.lora_A_init = "kaiming"
        self.layer_idx = -1
        ##############################################

    def _init_params(self, args):  # 用 json/config 里的参数覆盖默认值
        super()._init_params(args)

        # slora_gamma: S_lora 分支的缩放系数  0.5
        self.slora_gamma = float(args.get("slora_gamma", 1.0))
        # plora_gamma: P_lora 分支的缩放系数  0.75
        self.plora_gamma = float(args.get("plora_gamma", 1.0))
        # LoRA_output = slora_gamma * S_lora(x) + plora_gamma * P_lora(x)
        # self.merge_gamma = float(args.get("merge_gamma", 1.0))
        if self.use_slora and self.use_plora and args.get("avg", False):
            self.slora_gamma *= 0.5
            self.plora_gamma *= 0.5
        """
            "svd"       只用 SVD 重要性
            "soft_svd"  所有奇异方向按能量连续加权
        """
        self.dual_mask_importance = str(args.get("dual_mask_importance", "svd")).lower()
        self.dual_mask_general_ratio = float(args.get("dual_mask_general_ratio", 0.5))  # 0.4

        self.dual_mask_svd_rank = int(args.get("dual_mask_svd_rank", self.rank))  # 32
        self.dual_mask_svd_energy_coverage = float(args.get("dual_mask_svd_energy_coverage", 0.0))

        # I_t = (1-α_t)·normalize(I_svd) + α_t·normalize(I_grad)

        self.dual_mask_conflict_ratio = float(args.get("dual_mask_conflict_ratio", 0.25)) # Top-k 的比例参数  0.1
        self.dual_mask_conflict_strength = float(args.get("dual_mask_conflict_strength", 1.0))  # 冲突区压制多强  也就是beta

        self.dual_mask_conflict_reg_enabled = bool(args.get("dual_mask_conflict_reg_enabled", True))
        self.dual_mask_conflict_old_overlap_adaptive = bool(args.get("dual_mask_conflict_old_overlap_adaptive", False))

        # 保护区的强度是否根据W0_competence自适应
        self.dual_mask_competence_adaptive = bool(args.get("dual_mask_competence_adaptive", False))
        self.dual_mask_plasticity_adaptive = bool(args.get("dual_mask_plasticity_adaptive", False))
        """
            static_w0 = true
                固定 Top-ratio mask，不看 competence

            static_w0 = false
                competence 决定保护覆盖率、保护强度和 private rank
        """
        ## 永久保存一份初始预训练权重
        self.capture_pretrained_anchor()
        self.set_pretrained_competence(0.0)

        ## 可视化对应的超参数
        self.dual_mask_vis = bool(args.get("dual_mask_vis", False))
        self.dual_mask_vis_dir = str(args.get("dual_mask_vis_dir", self.dual_mask_vis_dir))
        self.dual_mask_vis_layers = self._parse_vis_indices(args.get("dual_mask_vis_layers", [0, 5, 11]))
        self.dual_mask_vis_tasks = self._parse_vis_indices(args.get("dual_mask_vis_tasks", [0, 1]))
        self.dual_mask_vis_save_weight = bool(args.get("dual_mask_vis_save_weight", False))
        self.lora_A_init = str(args.get("lora_A_init", "orthogonal")).lower()
        logging.info(
            # "Dual-mask branch: importance=%s, protect_ratio=%.3f, svd_rank=%s, "
            # "grad_alpha=%.3f, conflict_ratio=%.3f, "
            # # "conflict_strength=%.3f, A_init=%s",
            # # "conflict_strength=%.3f, task_relevance=%s, "
            # "conflict_strength=%.3f, conflict_adaptive=%s, task_relevance=%s, "
            # "task_coverage=%.3f, A_init=%s",
            # self.dual_mask_importance,
            # self.dual_mask_general_ratio,
            # self.dual_mask_svd_rank,
            # self.dual_mask_grad_alpha,
            # self.dual_mask_conflict_ratio,
            # self.dual_mask_conflict_adaptive,
            # self.dual_mask_conflict_strength,
            # self.dual_mask_task_relevance_enabled,
            # self.dual_mask_task_coverage,
            # self.lora_A_init
            "Dual-mask branch: importance=%(importance)s, "
            "protect_ratio=%(protect_ratio).3f, svd_rank=%(svd_rank)s, "
            "conflict_strength=%(conflict_strength).3f, "
            "old_overlap_conflict_adaptive=%(old_overlap_conflict_adaptive)s, "
            "plasticity_adaptive=%(plasticity_adaptive)s, "
            "A_init=%(a_init)s",
            {
                "importance": self.dual_mask_importance,
                "protect_ratio": self.dual_mask_general_ratio,
                "svd_rank": self.dual_mask_svd_rank,
                "conflict_ratio": self.dual_mask_conflict_ratio,
                "conflict_strength": self.dual_mask_conflict_strength,
                "old_overlap_conflict_adaptive": self.dual_mask_conflict_old_overlap_adaptive,
                "plasticity_adaptive": self.dual_mask_plasticity_adaptive,
                "a_init": self.lora_A_init,
            },
        )

    """ 
        把当前 qkv.weight 复制一份，永久保存为 pretrained_weight，作为后续的 W_pre / W0 anchor 
            保存 W0 副本
    """
    def capture_pretrained_anchor(self, force: bool = False):
        if bool(self.pretrained_anchor_captured.item()) and not force:
            return  # 已经保存后面不再保存
        with torch.no_grad():
            self.pretrained_weight.copy_(self.qkv.weight.detach())  # W_pre = W0.clone() 永远不变的 W_pre
            # pretrained_anchor_captured = False
            ## 原始预训练权重 W_pre 是否已经保存过
            ## → 把当前 qkv.weight 复制到 pretrained_weight
            ## → pretrained_anchor_captured = True
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
            # with use_pretrained_anchor(): 内部的代码在此执行
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

    """ 
        把当前任务的 W_pre competence 映射成自适应超参数

        使用固定的平衡策略复现旧版默认行为，但不暴露 low/high/min-ratio 配置。
    """
    # def set_pretrained_competence(self, competence: float):

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
            # coverage = base + span * competence
            self.effective_energy_coverage = ( # 单调线性控制器
                _BALANCED_COVERAGE_BASE
                # + _BALANCED_COVERAGE_SPAN * competence
                + _BALANCED_COVERAGE_SPAN * control_competence
            ) # 0.7 + 0.25 * ct
            # strength = base + span * competence
            self.effective_protect_strength = (
                _BALANCED_STRENGTH_BASE
                # + _BALANCED_STRENGTH_SPAN * competence
                + _BALANCED_STRENGTH_SPAN * control_competence
            ) # # 0.3 + 0.6 * ct
            rank_ratio = 1.0 - control_competence * (
            # rank_ratio = 1.0 - competence * (
                1.0 - _BALANCED_PRIVATE_MIN_RATIO
            ) # 1 - ct*(1 - 0.25)
            self.current_private_rank = max(
                1,
                int(round(self.rank * rank_ratio)),
            )
        else:
            self.effective_energy_coverage = 0.0
            self.effective_protect_strength = _BALANCED_STATIC_STRENGTH
            self.current_private_rank = self.rank

    def set_pretrained_old_overlap_risk(self, risk: float):
        self.pretrained_old_overlap_risk = min(max(float(risk), 0.0), 1.0)

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
            # payload["qkv_weight"] = self.qkv.weight.detach().cpu().float()
            payload["pretrained_weight"] = self.pretrained_weight.detach().cpu().float()
            payload["accumulated_weight"] = self.qkv.weight.detach().cpu().float()

        torch.save(payload, save_path)
        logging.info("Saved dual-mask visualization snapshot: %s", save_path)

    ######################################

    def _init_A_weight(self, dim: int, rank: int, device, dtype) -> torch.Tensor:
        if self.lora_A_init in ("kaiming", "kaiming_uniform"):
            return _kaiming_A_init(dim, rank, device, dtype)
        if self.lora_A_init not in ("orthogonal", "qr"):
            logging.info("Unknown lora_A_init=%s; using orthogonal A init.", self.lora_A_init)
        return _random_fixed_A_init(dim, rank, device, dtype)

    """ 
        每个task开始训练前做准备
        1. 先让父类创建当前 task 的 S_lora
        2. 再给当前 task 创建 P_lora
        3. 根据当前 qkv.weight 重建 dual mask
            before_task() 是为当前 task 准备双 LoRA 分支，并根据当前主干权重重新划分“保护区域”和“可塑区域”         
    """
    def before_task(self, task: int):
        super().before_task(task)
        t = int(task)
        device = next(self.parameters()).device
        dtype = self.qkv.weight.dtype
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

    # “全谱软加权的重要性计算”，最终得到的仍然是每个权重位置的重要性分数
    ## Soft:因为它没有只保留前 k 个方向，而是使用全部768个方向
    ### self.last_svd_rank = s.numel()       # 768
    ### self.last_svd_energy_coverage = 1.0  # 100%
    ### 每个方向按照能量连续加权，而不是简单保留/丢弃
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
          
    """
        用 SVD 找出 qkv.weight 的主奇异空间，
        再根据每个行/列在主空间里的能量，估计 W0 每个位置的重要性，
        最后为后续 general_mask / isolated_mask 提供分数。
    """
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
        """
            注意：这里算出来的还不是最终 mask
            score = normalize(score)
            general_mask = top_ratio_mask(score, dual_mask_general_ratio)
            isolated_mask = 1 - general_mask
            比如 dual_mask_general_ratio = 0.5，就是把 score 最高的 50% 位置标成保护区
        """
        return score.to(device=weight.device, dtype=weight.dtype)

    """
        _combined_importance() 是在选择 W0 重要性来源：
        默认用 SVD；soft_svd 使用全部奇异方向的连续能量权重。
    """
    def _combined_importance(self) -> torch.Tensor:
        # 取当前 attention 层的 qkv.weight，detach() 表示不让这一步进入反向传播图。因为 mask 是一个启发式统计量，不需要通过它反传梯度
        # weight = self.qkv.weight.detach()   W0!
        weight = self.pretrained_weight.detach() # 永远不变的 W_pre
        # if self.dual_mask_static_w0: # 不使用 W0_competence
        #     return self._svd_importance(weight)
        # # 先默认算一份 SVD 重要性
        # svd_score = self._svd_importance(weight)

        mode = self.dual_mask_importance  # "dual_mask_importance": "svd"
        if mode == "soft_svd":
            svd_score = self._soft_svd_importance(weight) # 硬截断rank=32  覆盖率大概54%
        else:
            svd_score = self._svd_importance(weight) # 硬截断rank=32  覆盖率大概54%
        return svd_score

    """
        rebuild_dual_masks() 根据当前 qkv.weight 的重要性分数，把权重矩阵位置分成两类：
            重要位置 general_mask，用来保护 W0    保护区 mask
            非重要位置 isolated_mask，用来给 P_lora 做可塑更新  可塑区 mask
        W_pre 的重要性分数可跨任务复用；保护区域会按当前任务的
        competence coverage 重新阈值化。
    """
    def rebuild_dual_masks(self):
        with torch.no_grad():
            # SVD-only 的 W_pre 分数跨任务不变，可以复用；但每个任务仍按
            # 当前 competence 重新阈值化，使 adaptive coverage 真正生效。
            reuse_w0_score = (
                self.cur_task > 0
                # and self.dual_mask_importance == "svd"
                and self.dual_mask_importance in ("svd","soft_svd",)
                and bool(torch.count_nonzero(self.w0_importance).item())
            )
            if reuse_w0_score: # 缓存并复用 W0 的 SVD 重要性分数，避免每个任务都重复做 SVD
                score = self.w0_importance.detach().clone()
            else:
                score = _normalize_score(self._combined_importance())

            # 根据重要性分数，取 top ratio 作为保护区
            ## score 最高的 50% 位置 -> protect = 1 | 1 表示这个位置是 W0 重要位置，不希望 LoRA 改
            ## score 剩下的 50% 位置 -> protect = 0 | 0 表示这个位置可以改
            # protect = _top_ratio_mask(score, self.dual_mask_general_ratio) # mask
            use_spectral_loss = self.dual_mask_importance == "spectral_loss"
            use_adaptive_coverage = self.dual_mask_competence_adaptive
            #if use_adaptive_coverage:
            if use_spectral_loss:
                # W0 protection moves from weight coordinates to singular
                # directions. Keep every coordinate open; actual conflict
                # control remains hard or soft according to its own switch.
                mask_coverage = 0.0
                protect = torch.zeros_like(score)
            elif use_adaptive_coverage:
                # 使用能量覆盖，而不是固定 top ratio
                ## M_g = general_mask = Task 0 时 W_pre 的重要保护区
                mask_coverage = self.effective_energy_coverage
                protect = _energy_coverage_mask(score, mask_coverage)
            else:
                # 使用固定 top ratio
                mask_coverage = 0.0
                protect = _top_ratio_mask(score, self.dual_mask_general_ratio)  # mask
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

    """
        用 W0 的重要性和 LoRA 当前 BA 的改动幅度相乘，
            找出“W0 很重要且 LoRA 正在强改”的高冲突位置，
            后面用这些位置来抑制 LoRA 更新，减少对旧知识/主干权重的破坏
    """
    def _conflict_parameters(self):
        """Return the fixed conflict range and optional old-overlap strength."""
        base_ratio = min(max(self.dual_mask_conflict_ratio, 0.0), 1.0)
        base_strength = min(max(self.dual_mask_conflict_strength, 0.0), 1.0)
        if self.dual_mask_conflict_old_overlap_adaptive:
            base_strength = min(base_strength * (1.0 + self.pretrained_old_overlap_risk),1.0,)
        return base_ratio, base_strength

    def _joint_conflict(
            self,
            delta: torch.Tensor,
            conflict_ratio: Optional[float] = None,
    ):
        ## abs(delta[i, j]) 越大，说明 LoRA 越想修改这个位置
        ## ba_importance[i, j] 越大，表示 BA 在这个位置的改动越强   _normalize_score 归一化到大概 [0, 1]
        ba_importance = _normalize_score(delta.detach().abs())
        w0_importance = self.w0_importance.to(device=delta.device, dtype=delta.dtype)
        # 公式:conflict_score[i, j] = normalize(
        #     w0_importance[i, j] * ba_importance[i, j]
        # )
        # 如果 W0 在这个位置很重要，并且 LoRA 也想大幅修改这个位置，那么这个位置就是高冲突位置
        """
            e.g. W0 重要性高，BA 改动大 -> 冲突高
                W0 重要性高，BA 改动小 -> 冲突低
                W0 重要性低，BA 改动大 -> 冲突也不算高
                W0 重要性低，BA 改动小 -> 冲突低

            它不是单纯保护所有 W0 重要位置，而是进一步看:LoRA 当前到底有没有动这些重要位置
        """
        # 联合冲突
        ## 当前 BA 是否正在强改 W_pre 重要位置，训练中动态变化
        conflict_score = _normalize_score(w0_importance * ba_importance)
        # 把冲突分数最高的一部分位置标出来
        ## 比如配置 dual_mask_conflict_ratio = 0.25，就把冲突分数最高的 25% 位置标为 1，其他位置标为 0
        ## conflict_mask[i, j] = 1  表示这个位置是高冲突区域 conflict_mask[i, j] = 0  表示这个位置冲突不高
        # conflict_mask = _top_ratio_mask(conflict_score, self.dual_mask_conflict_ratio)
        # if self.dual_mask_conflict_coverage_adaptive:

        conflict_mask = _top_ratio_mask(
            conflict_score,
            self.dual_mask_conflict_ratio if conflict_ratio is None else conflict_ratio,
        )
        return conflict_score, conflict_mask

    """
        把原始 LoRA 增量 BA 变成安全增量：
            保护 W0 重要区域，抑制 BA-W0 高冲突区域；
            如果是 P_lora，还限制它只能在 plastic 区域更新

        输入原始 LoRA 增量 delta = B @ A
        输出被 dual mask 过滤后的 safe_delta

        safe_delta = delta * gate
    """
    # def _safe_delta(self, delta: torch.Tensor, isolated: bool) -> torch.Tensor:
    def _safe_delta(
            self,
            delta: torch.Tensor,
            isolated: bool,
            conflict_ratio: Optional[float] = None,
            conflict_strength: Optional[float] = None,
    ) -> torch.Tensor:
        # general_mask/protect_mask: W0 重要区域，应该保护
        protect_mask = self.general_mask.to(device=delta.device, dtype=delta.dtype)
        # isolated_mask/plastic_mask: W0 非重要区域，允许 P_lora 使用
        plastic_mask = 1.0 - protect_mask
        # conflict 高表示,W0 很重要，而且 LoRA 也想大幅修改这个位置
        _, conflict_mask = self._joint_conflict(
            delta,
            conflict_ratio=conflict_ratio,
        )

        # conflict_strength = min(max(self.dual_mask_conflict_strength, 0.0), 1.0)  # beta
        if conflict_strength is None:
            conflict_strength = self.dual_mask_conflict_strength
        conflict_strength = min(max(conflict_strength, 0.0), 1.0)

        protect_strength = min(max(self.effective_protect_strength, 0.0), 1.0)
        # 保护区统一使用 competence-adaptive 强度，复现旧版平衡控制器。
        protect_gate = 1.0 - protect_strength * protect_mask

        # 高冲突区域按 conflict_strength 压制；其余区域保持不变。
        conflict_gate = 1.0 - conflict_strength * conflict_mask.to(delta.dtype)
        # Shared LoRA 在保护区统一衰减，再叠加冲突抑制。
        # Private LoRA 额外受 plastic_mask 限制，只使用非保护区。
        gate = protect_gate * conflict_gate  # [2304,768]

        if isolated:
            gate = gate * plastic_mask
        """
            S_lora:
                safe_delta_s = delta_s * protect_gate * conflict_gate
                
                -> safe_delta_s = delta_s * protect_gate
                               * conflict_gate

            P_lora:
                safe_delta_p = delta_p * protect_gate * conflict_gate * plastic_mask
                P_lora 只能在非保护区，也就是 plastic 区域工作
                
                -> safe_delta_p = delta_p * protect_gate
                               * conflict_gate * plastic_mask
                启用 task relevance 后，P_lora 只在 task-plastic 区域工作。
        """
        return delta * gate

    """
        训练/前向传播时，LoRA 分支怎么参与 qkv 输出
        不要直接用原始 LoRA 的 BA
            先把 BA 按 mask 过滤成 safe_BA
            再让 safe_BA 参与 qkv 前向传播
            qkv = W0 x + safe_BA x
    """
    def _masked_unit_forward(self, x: torch.Tensor, unit, isolated: bool) -> torch.Tensor:
        delta = unit.B_weight @ unit.A_weight
        # safe_delta = BA * gate
        ## isolated=True: safe_delta_p = BA_p * protect_gate * conflict_gate * plastic_mask
        ## isolated=False: safe_delta_s = BA_s * protect_gate * conflict_gate
        delta = self._safe_delta(delta, isolated=isolated)
        return F.linear(x, delta)  # 输出 = x @ safe_delta.T

    """
        训练 loss 里额外加的正则项，惩罚 LoRA 在危险区域产生过大更新
            函数返回一个正则 loss
    """

    def _joint_conflict_regularization(self, unit, isolated: bool) -> torch.Tensor:
        delta = unit.B_weight @ unit.A_weight # ΔW = 0 × A = 0
        # safe_delta = BA * gate
        safe_delta = self._safe_delta(delta, isolated=isolated)
        w0_importance = self.w0_importance.to(device=delta.device, dtype=delta.dtype)
        conflict_score, _ = self._joint_conflict(delta)
        # 如果某个位置 W0 很重要，那么 safe_delta 在这个位置越大，惩罚越大
        protection = (w0_importance * safe_delta.pow(2)).mean()
        # 如果某个位置 conflict_score 高，那么 safe_delta 在这个位置越大，惩罚越大
        conflict = (conflict_score.detach() * safe_delta.pow(2)).mean()
        # return protection + conflict
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
        protect_mask = self.general_mask.detach().float()

        conflict_entropy, conflict_top10_energy = self._conflict_distribution_stats(
            conflict_score,
            conflict_mask,
        )

        conflict_energy50_ratio = self._conflict_energy50_ratio(conflict_score)
        # conflict_strength = min(max(self.dual_mask_conflict_strength, 0.0), 1.0)

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

        with torch.no_grad():
            self.last_conflict_entropy.copy_(conflict_entropy)
            self.last_conflict_top10_energy.copy_(conflict_top10_energy)
            self.last_conflict_energy50_ratio.copy_(conflict_energy50_ratio)
            self.last_conflict_gate_suppression.fill_(conflict_gate_suppression)
            self.last_safe_suppression.fill_(total_stats["suppressed_ratio"])

            self.last_effective_conflict_ratio.fill_(float(conflict_ratio))
            self.last_effective_conflict_strength.fill_(conflict_strength)

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
            float(conflict_ratio),
            conflict_strength,
            total_stats["q_safe_norm"],
            total_stats["k_safe_norm"],
            total_stats["v_safe_norm"],
        )

    """
        计算当前 task 的 LoRA 增量输出，并把 S_lora / P_lora 按规则组合起来
            把 dual-mask LoRA 真正接进 attention forward 的地方
            原始 qkv 输出
                + shared LoRA 的安全增量
                + isolated LoRA 的安全增量
        qkv(x) = W0 x + LoRA_contribution
    """

    def _contrib_from_units(self, x: torch.Tensor, t_idx: int) -> torch.Tensor:

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
            return self._masked_unit_forward(x, unit_s, isolated=False)

        slora_gamma = float(self.slora_gamma)
        plora_gamma = float(self.plora_gamma)
        out = zero_output

        # if t_idx == 0:
        #     return slora_gamma * self._masked_unit_forward(x, unit_s, isolated=False)

        if unit_s is not None and (self.use_slora or t_idx == 0):
            out = out + slora_gamma * self._masked_unit_forward(x, unit_s, isolated=False)

        if t_idx > 0 and self.use_plora and unit_p is not None:
            # safe_delta_p = BA_p * protect_gate * conflict_gate * plastic_mask
            ## P_lora 只能在 W0 非重要区域更新
            out = out + plora_gamma * self._masked_unit_forward(x, unit_p, isolated=True)

        """
            Task 0:
                LoRA(x) = gamma_s * safe_S(x)

            Task > 0:
                LoRA(x) = gamma_s * safe_S(x) + gamma_p * safe_P(x)

            safe_S = masked BA_s, 不使用 isolated_mask
            safe_P = masked BA_p, 使用 isolated_mask
        """
        return out

    """
        把当前 task 学到的 masked LoRA 增量永久合并进 qkv.weight，
            然后释放当前 LoRA，避免参数随任务数持续增长

        训练中:
            qkv(x) = W_old x + safe_delta x

        task 结束:
            W_new = W_old + safe_delta
            释放 LoRA A/B

        后续:
            qkv(x) = W_new x
    """
    def after_task(self, task: int):
        t = int(task)
        device = next(self.parameters()).device
        dtype = self.qkv.weight.dtype

        """
            1. 算当前 LoRA 分支的原始增量：
               delta = gamma * B @ A

            2. 经过 dual mask：
               safe_delta = _safe_delta(delta, isolated)
        """  ###################

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
            safe_delta = self._safe_delta(
                item["raw_delta"],
                isolated=item["isolated"],
                conflict_ratio=conflict_ratio,
                conflict_strength=conflict_strength,
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

        return None
