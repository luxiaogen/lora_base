import logging
import os
import torch
import torch.nn.functional as F

from models.decomposed_lora import Attention_LoRA as BaseAttentionLoRA
# from models.decomposed_lora import _energy_merge
from models.decomposed_lora import FrozenA_TrainableB
# from models.decomposed_lora import _random_fixed_A_init, _zero_B_init
from models.decomposed_lora import _kaiming_A_init, _random_fixed_A_init, _zero_B_init

def _normalize_score(score: torch.Tensor) -> torch.Tensor:
    score = score.float()
    score = score - score.min()
    denom = score.max().clamp_min(1e-12)
    return score / denom


def _top_ratio_mask(score: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = min(max(float(ratio), 0.0), 1.0)  # 0.5
    flat = score.flatten() # 2304*768
    if flat.max() <= flat.min():
        return torch.zeros_like(score)
    if ratio <= 0.0:
        return torch.zeros_like(score)
    if ratio >= 1.0:
        return torch.ones_like(score)
    k = max(1, int(flat.numel() * ratio)) # 2304*768*0.5
    threshold = torch.topk(flat, k, largest=True).values.min() # tensor(0.0032) | 选这50%中要保护的区域的最小值
    return (score >= threshold).to(score.dtype) # 大于这个阈值的就是要保护的区域


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
        # W0 x + S_lora(x) + P_lora(x)
        # S_lora = shared / general branch
        # P_lora = particular / isolated branch
        self.p_rank = self.rank # 每个 task 都有一个自己的 isolated LoRA 分支
        self.P_lora = torch.nn.ModuleList([None for _ in range(self.n_tasks)])
        # 都是 buffer，不是可训练参数,也就是说它们会跟着 module 一起 .to(cuda)、.eval()、.train()，但不会被 optimizer 更新
        # W0 = qkv.weight 每个位置的重要性分数 [3 * dim, dim]  | w0_importance[i, j] 越大，说明 W0 这个位置越重要，越不希望 LoRA 改它
        """
            w0_importance        W0 每个位置的重要性分数
            general_mask         W0 重要区域，保护用
            isolated_mask        W0 非重要区域，给 P_lora 的 plastic 区域
            last_conflict_score  最近一次 W0 与 BA 的冲突分数，主要 debug
            last_conflict_mask   最近一次冲突区域 mask，实际参与 gate
            grad_importance      梯度敏感度重要性，用于 grad/svd_grad 模式
        """
        self.register_buffer("w0_importance", torch.zeros(shape), persistent=False)
        # general_mask[i, j] = 1  表示这个 W0 位置重要，要保护 | general_mask[i, j] = 0  表示这个位置相对不重要，可以改
        self.register_buffer("general_mask", torch.ones(shape), persistent=False)
        # isolated_mask[i, j] = 1  表示这个位置不太重要，可以给 isolated branch 改 | isolated_mask[i, j] = 0  表示这个位置重要，不给 P_lora 改
        ## P_lora 的增量: BA_p * isolated_mask
        self.register_buffer("isolated_mask", torch.ones(shape), persistent=False)
        # 记录最近一次算出来的 conflict 分数 conflict_score = normalize(w0_importance * abs(BA))
        ## W0 这个位置本来就重要，同时 LoRA 又想大幅修改它，那么冲突高
        self.register_buffer("last_conflict_score", torch.zeros(shape), persistent=False)
        # "dual_mask_conflict_ratio": 0.25
        ## 那就是把 conflict 分数最高的 25% 位置标为 1
        self.register_buffer("last_conflict_mask", torch.zeros(shape), persistent=False)
        # 保存基于梯度的 W0 重要性
        ## "dual_mask_importance": "grad" | "svd_grad"
        ## 代码会先跑几个 batch，计算 sensitivity = abs(grad(qkv.weight) * qkv.weight)
        ## 表示:某个 W0 位置对当前 task loss 越敏感，越重要。
        self.register_buffer("grad_importance", torch.zeros(shape), persistent=False)
        ### 先判断 W0 哪里重要，再决定 LoRA 的 BA 哪里能加、哪里不能加
        self.dual_mask_importance = "svd"
        self.dual_mask_general_ratio = 0.5 # 决定 W0 保护区多大
        self.dual_mask_svd_rank = self.rank # 决定 SVD 重要性用多少主方向
        self.dual_mask_grad_alpha = 0.5 # 决定 SVD 和 gradient 怎么混合
        self.dual_mask_conflict_ratio = 0.25 # 决定 BA-W0 冲突区多大
        self.dual_mask_protect_strength = 1.0 # 决定保护区压制多强
        self.dual_mask_conflict_strength = 1.0 # 决定冲突区压制多强

        ##############################################
        self.dual_mask_vis = False
        self.dual_mask_vis_dir = "visualizations/dual_mask_snapshots"
        self.dual_mask_vis_layers = {0, 5, 11}
        self.dual_mask_vis_tasks = {0, 1}
        self.dual_mask_vis_save_weight = False
        self.lora_A_init = "kaiming"
        self.layer_idx = -1
        ##############################################
        self.n_grad_importance = 0 # 记录梯度重要性累积了几个 batch

    def _init_params(self, args): # 用 json/config 里的参数覆盖默认值
        super()._init_params(args)

        self.lora_eps = float(args.get("lora_eps", 1e-5))
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
            "grad"      只用梯度重要性
            "svd_grad"  SVD + 梯度混合
        """
        self.dual_mask_importance = str(args.get("dual_mask_importance", "svd")).lower() # svd
        self.dual_mask_general_ratio = float(args.get("dual_mask_general_ratio", 0.5)) # 0.5
        self.dual_mask_svd_rank = int(args.get("dual_mask_svd_rank", self.rank)) # 32
        self.dual_mask_grad_alpha = float(args.get("dual_mask_grad_alpha", 0.5))# 梯度 目前没有用到
        self.dual_mask_conflict_ratio = float(args.get("dual_mask_conflict_ratio", 0.25))
        self.dual_mask_protect_strength = float(args.get("dual_mask_protect_strength", 1.0)) # 0.7
        self.dual_mask_conflict_strength = float(args.get("dual_mask_conflict_strength", 1.0)) # 0.5
        

        #####################
        self.dual_mask_vis = bool(args.get("dual_mask_vis", False))
        self.dual_mask_vis_dir = str(args.get("dual_mask_vis_dir", self.dual_mask_vis_dir))
        self.dual_mask_vis_layers = self._parse_vis_indices(
            args.get("dual_mask_vis_layers", [0, 5, 11])
        )
        self.dual_mask_vis_tasks = self._parse_vis_indices(
            args.get("dual_mask_vis_tasks", [0, 1])
        )
        self.dual_mask_vis_save_weight = bool(args.get("dual_mask_vis_save_weight", False))
        self.lora_A_init = str(args.get("lora_A_init", "orthogonal")).lower()
        logging.info(
            "Dual-mask branch: importance=%s, protect_ratio=%.3f, svd_rank=%s, "
            "grad_alpha=%.3f, conflict_ratio=%.3f, protect_strength=%.3f, "
            # "conflict_strength=%.3f",
            "conflict_strength=%.3f, A_init=%s",
            self.dual_mask_importance,
            self.dual_mask_general_ratio,
            self.dual_mask_svd_rank,
            self.dual_mask_grad_alpha,
            self.dual_mask_conflict_ratio,
            self.dual_mask_protect_strength,
            self.dual_mask_conflict_strength,
            self.lora_A_init
        )
    
    ###############################
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

    def _save_dual_mask_snapshot(self, task: int, branch_deltas):
        if not self._should_save_dual_mask_snapshot(task):
            return

        raw_delta = torch.stack([item["raw_delta"] for item in branch_deltas]).sum(dim=0)
        safe_delta = torch.stack([item["safe_delta"] for item in branch_deltas]).sum(dim=0)
        conflict_score, conflict_mask = self._joint_conflict(raw_delta)

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
            "conflict_ratio": float(self.dual_mask_conflict_ratio),
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
            payload["qkv_weight"] = self.qkv.weight.detach().cpu().float()

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
        # a_rand = _random_fixed_A_init(self.dim, self.p_rank, device, dtype) # A随机初始化为正交阵
        if self.lora_A_init in ("kaiming", "kaiming_uniform"):
            with torch.no_grad(): # A 初始化正态分布
                self.S_lora[t].A.weight.copy_(
                    self._init_A_weight(self.dim, self.rank, device, dtype).to(
                        self.S_lora[t].A.weight.device,
                        dtype=self.S_lora[t].A.weight.dtype,
                    )
                )
                self.S_lora[t].B.weight.zero_() # B初始化为0

        a_rand = self._init_A_weight(self.dim, self.p_rank, device, dtype)  # A随机初始化为正交阵或 Kaiming
        b_zero = _zero_B_init(self.dim * 3, self.p_rank, device, dtype) # B初始化0
        # 正式创建当前 task 的 isolated branch
        self.P_lora[t] = FrozenA_TrainableB(
            self.dim,
            self.dim * 3,
            self.p_rank,
            a_rand,
            b_zero,
            device=device,
            dtype=dtype,
        )
        self.rebuild_dual_masks() # Dual masks rebuilt: W0 protect density 0.5000, plastic density 0.5000

    def set_task_and_stage(self, task: int, layer_idx: int, stage: int = 0):
        task = int(task)
        ##################
        self.layer_idx = int(layer_idx)

        self.cur_task = task
        for p in self.qkv.parameters():
            p.requires_grad_(False)
        for p in self.proj.parameters():
            p.requires_grad_(False)

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

    def clear_gradient_importance(self):
        self.grad_importance.zero_()
        self.n_grad_importance = 0

    def accumulate_qkv_gradient(self):
        grad = self.qkv.weight.grad
        if grad is None:
            return
        with torch.no_grad():
            sensitivity = (grad.detach() * self.qkv.weight.detach()).abs()
            self.grad_importance.add_(
                sensitivity.to(
                    device=self.grad_importance.device,
                    dtype=self.grad_importance.dtype,
                )
            )
            self.n_grad_importance += 1
    """
        用 SVD 找出 qkv.weight 的主奇异空间，
        再根据每个行/列在主空间里的能量，估计 W0 每个位置的重要性，
        最后为后续 general_mask / isolated_mask 提供分数。
    """
    def _svd_importance(self, weight: torch.Tensor) -> torch.Tensor:
        weight_f = weight.detach().float()
        # s 里的值越大，说明对应方向越重要
        u, s, vh = torch.linalg.svd(weight_f, full_matrices=False)
        # 取前 k 个奇异值
        k = max(1, min(int(self.dual_mask_svd_rank), s.numel()))
        s_top = s[:k].clamp_min(0.0)
        # 算行重要性: row_score[i] = sum_l U[i, l]^2 * s[l]
        ## qkv.weight 的第 i 行，在 top-k 主奇异方向里贡献多大   关心的是它参与这个主方向的强度，不是方向正负
        ## u_l[i] 大:表示第 i 行强烈参与第 l 个左奇异方向 | v_l[j] 大:表示第 j 列强烈参与第 l 个右奇异方向
        ## u_l[i]^2:可以理解成第 l 个左奇异方向的能量有多少比例落在第 i 行上
        ## v_l[j]^2: l 个右奇异方向的能量有多少比例落在第 j 列上
        row_score = (u[:, :k].pow(2) * s_top.unsqueeze(0)).sum(dim=1)
        # 算列重要性: col_score[j] = sum_l V[j, l]^2 * s[l]
        ## qkv.weight 的第 j 列，在 top-k 主奇异方向里贡献多大
        col_score = (vh[:k, :].t().pow(2) * s_top.unsqueeze(0)).sum(dim=1)

        # 把行重要性和列重要性做外积，得到每个位置的综合重要性分数
        ## score[i, j] = row_score[i] * col_score[j]
        ## 如果第 i 行重要，而且第 j 列也重要，那么 W[i, j] 这个位置就被认为重要
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
            默认用 SVD；如果需要 task-aware，可以用梯度；也可以两者加权融合。
    """
    def _combined_importance(self) -> torch.Tensor:
        # 取当前 attention 层的 qkv.weight，detach() 表示不让这一步进入反向传播图。因为 mask 是一个启发式统计量，不需要通过它反传梯度
        weight = self.qkv.weight.detach()
        # 先默认算一份 SVD 重要性
        svd_score = self._svd_importance(weight)
        mode = self.dual_mask_importance # "dual_mask_importance": "svd"

        # grad_score = 多个 batch 的平均梯度敏感度
        grad_score = None
        if self.n_grad_importance > 0: # 使用看遮挡哪个参数后会导致loss升高生成相应的mask
            grad_score = self.grad_importance / float(self.n_grad_importance)
            grad_score = grad_score.to(device=weight.device, dtype=weight.dtype)

        if mode == "svd":
            return svd_score
        if mode in ("grad", "gradient"):
            if grad_score is None:
                logging.info("Gradient sensitivity is empty; falling back to SVD masks.")
                return svd_score
            return grad_score
        # 如果模式是混合模式，就把 SVD 和 grad 融合
        if mode in ("svd_grad", "grad_svd", "hybrid"):
            if grad_score is None:
                logging.info("Gradient sensitivity is empty; using SVD-only masks.")
                return svd_score
            alpha = min(max(self.dual_mask_grad_alpha, 0.0), 1.0)
            return (1.0 - alpha) * _normalize_score(svd_score) + alpha * _normalize_score(grad_score)

        logging.info("Unknown dual_mask_importance=%s; falling back to SVD masks.", mode)
        return svd_score

    """
        rebuild_dual_masks() 根据当前 qkv.weight 的重要性分数，把权重矩阵位置分成两类：
            重要位置 general_mask，用来保护 W0    保护区 mask
            非重要位置 isolated_mask，用来给 P_lora 做可塑更新  可塑区 mask
    """
    def rebuild_dual_masks(self):
        with torch.no_grad():
            score = _normalize_score(self._combined_importance()) # 最终的重要性分数 svd[2304,768]
            # 根据重要性分数，取 top ratio 作为保护区
            ## score 最高的 50% 位置 -> protect = 1 | 1 表示这个位置是 W0 重要位置，不希望 LoRA 改
            ## score 剩下的 50% 位置 -> protect = 0 | 0 表示这个位置可以改
            protect = _top_ratio_mask(score, self.dual_mask_general_ratio) # mask
            # plastic[i, j] = 1 表示这个位置可以给 P_lora 使用
            # plastic[i, j] = 0 表示这个位置是保护区
            plastic = 1.0 - protect # 可塑性区域
            self.w0_importance.copy_(
                score.to(device=self.w0_importance.device, dtype=self.w0_importance.dtype)
            )
            self.general_mask.copy_(
                protect.to(device=self.general_mask.device, dtype=self.general_mask.dtype)
            ) # 50% 位置被保护
            self.isolated_mask.copy_(
                plastic.to(device=self.isolated_mask.device, dtype=self.isolated_mask.dtype)
            ) # 50% 位置可塑


            logging.info(
                "Dual masks rebuilt: W0 protect density %.4f, plastic density %.4f",
                self.general_mask.float().mean().item(),
                self.isolated_mask.float().mean().item(),
            )

    """
        用 W0 的重要性和 LoRA 当前 BA 的改动幅度相乘，
            找出“W0 很重要且 LoRA 正在强改”的高冲突位置，
            后面用这些位置来抑制 LoRA 更新，减少对旧知识/主干权重的破坏
    """
    ## 算:当前 LoRA 增量 BA 和 W0 重要区域之间的冲突程度
    def _joint_conflict(self, delta: torch.Tensor):
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
        conflict_score = _normalize_score(w0_importance * ba_importance)
        # 把冲突分数最高的一部分位置标出来
        ## 比如配置 dual_mask_conflict_ratio = 0.25，就把冲突分数最高的 25% 位置标为 1，其他位置标为 0
        ## conflict_mask[i, j] = 1  表示这个位置是高冲突区域 conflict_mask[i, j] = 0  表示这个位置冲突不高
        conflict_mask = _top_ratio_mask(conflict_score, self.dual_mask_conflict_ratio)
        return conflict_score, conflict_mask

    """
        把原始 LoRA 增量 BA 变成安全增量：
            保护 W0 重要区域，抑制 BA-W0 高冲突区域；
            如果是 P_lora，还限制它只能在 plastic 区域更新
            
        输入原始 LoRA 增量 delta = B @ A
        输出被 dual mask 过滤后的 safe_delta
        
        safe_delta = delta * gate
    """
    def _safe_delta(self, delta: torch.Tensor, isolated: bool) -> torch.Tensor:
        # general_mask/protect_mask: W0 重要区域，应该保护
        protect_mask = self.general_mask.to(device=delta.device, dtype=delta.dtype)
        # isolated_mask/plastic_mask: W0 非重要区域，允许 P_lora 使用
        plastic_mask = self.isolated_mask.to(device=delta.device, dtype=delta.dtype)
        # conflict 高表示,W0 很重要，而且 LoRA 也想大幅修改这个位置
        conflict_score, conflict_mask = self._joint_conflict(delta)
        # 把最近一次的冲突分数和冲突 mask 存下来,这主要是为了 debug / 可视化 / 后面分析，不是训练必须项
        with torch.no_grad():
            self.last_conflict_score.copy_(
                conflict_score.to(
                    device=self.last_conflict_score.device,
                    dtype=self.last_conflict_score.dtype,
                )
            )
            self.last_conflict_mask.copy_(
                conflict_mask.to(
                    device=self.last_conflict_mask.device,
                    dtype=self.last_conflict_mask.dtype,
                )
            )

        protect_strength = min(max(self.dual_mask_protect_strength, 0.0), 1.0) # alpha
        conflict_strength = min(max(self.dual_mask_conflict_strength, 0.0), 1.0) # beta
        # 保护区 gate
        # protect_mask = 1 -> protect_gate = 0
        # protect_mask = 0 -> protect_gate = 1
        # W0 重要保护区：LoRA 增量压成 0
        # 非保护区：LoRA 增量保留
        protect_gate = 1.0 - protect_strength * protect_mask
        # 高冲突区域：LoRA 增量压成 0
        # 低冲突区域：LoRA 增量保留
        conflict_gate = 1.0 - conflict_strength * conflict_mask.to(delta.dtype)
        # 就是一个位置最终能否被 LoRA 修改，要同时满足
        ## 不是 W0 保护区 && 不是高冲突
        gate = protect_gate * conflict_gate  # [2304,768]
        if isolated:
            gate = gate * plastic_mask
        """
            S_lora:
                safe_delta_s = delta_s * protect_gate * conflict_gate
            
            P_lora:
                safe_delta_p = delta_p * protect_gate * conflict_gate * plastic_mask
                P_lora 只能在非保护区，也就是 plastic 区域工作
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
        return F.linear(x, delta) # 输出 = x @ safe_delta.T

    """
        训练 loss 里额外加的正则项，惩罚 LoRA 在危险区域产生过大更新
            函数返回一个正则 loss
    """
    def _joint_conflict_regularization(self, unit, isolated: bool) -> torch.Tensor:
        delta = unit.B_weight @ unit.A_weight
        # safe_delta = BA * gate
        safe_delta = self._safe_delta(delta, isolated=isolated)
        w0_importance = self.w0_importance.to(device=delta.device, dtype=delta.dtype)
        conflict_score, _ = self._joint_conflict(delta)
        # 如果某个位置 W0 很重要，那么 safe_delta 在这个位置越大，惩罚越大
        protection = (w0_importance * safe_delta.pow(2)).mean()
        # 如果某个位置 conflict_score 高，那么 safe_delta 在这个位置越大，惩罚越大
        conflict = (conflict_score.detach() * safe_delta.pow(2)).mean()
        return protection + conflict

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

    def _log_merge_stats(self, task: int, branch_deltas):
        raw_total = torch.stack([item["raw_delta"] for item in branch_deltas]).sum(dim=0)
        safe_total = torch.stack([item["safe_delta"] for item in branch_deltas]).sum(dim=0)
        total_stats = self._delta_stats(raw_total, safe_total)
        _, conflict_mask = self._joint_conflict(raw_total)

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
            "Q_safe_norm=%.6f, K_safe_norm=%.6f, V_safe_norm=%.6f",
            int(task),
            int(self.layer_idx),
            total_stats["raw_norm"],
            total_stats["safe_norm"],
            total_stats["suppressed_ratio"] * 100.0,
            total_stats["raw_abs_mean"],
            total_stats["safe_abs_mean"],
            total_stats["max_abs"],
            self.general_mask.float().mean().item(),
            self.isolated_mask.float().mean().item(),
            conflict_mask.float().mean().item(),
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
        unit_s = self.S_lora[t_idx] # S_lora[t_idx] = 当前 task 的共享 LoRA
        # unit_p = self.P_lora[t_idx]
        if not self.use_slora and not self.use_plora:
            return self._masked_unit_forward(x, unit_s, isolated=False)

        slora_gamma = float(self.slora_gamma)
        plora_gamma = float(self.plora_gamma)
        out = 0.0

        # if t_idx == 0:
        #     return slora_gamma * self._masked_unit_forward(x, unit_s, isolated=False)

        if self.use_slora or t_idx == 0:
            out = out + slora_gamma * self._masked_unit_forward(x, unit_s, isolated=False)

        unit_p = self.P_lora[t_idx] # P_lora[t_idx] = 当前 task 的私有/隔离 LoRA
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
            然后清零并冻结当前 LoRA，避免下一轮重复加或继续被训练
        
        训练中:
            qkv(x) = W_old x + safe_delta x
        
        task 结束:
            W_new = W_old + safe_delta
            清零 LoRA B
        
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
        """ ###################
        def masked_delta(name: str, unit, gamma: float, isolated: bool):
            delta = gamma * (unit.B_weight.detach() @ unit.A_weight.detach())
            safe_delta = self._safe_delta(delta, isolated=isolated)
            return {
                "name": name,
                "isolated": isolated,
                "gamma": gamma,
                "raw_delta": delta,
                "safe_delta": safe_delta,
            }
        branch_deltas = []
        if not self.use_slora and not self.use_plora:
            branch_deltas.append(masked_delta("S", self.S_lora[t], 1.0, isolated=False))
        else:
            if self.use_slora or t == 0:
                branch_deltas.append(
                    masked_delta("S", self.S_lora[t], float(self.slora_gamma), isolated=False)
                )
            if t > 0 and self.use_plora and self.P_lora[t] is not None:
                branch_deltas.append(
                    masked_delta("P", self.P_lora[t], float(self.plora_gamma), isolated=True)
                )

        if branch_deltas:
            self._save_dual_mask_snapshot(t, branch_deltas)

            #########################
            self._log_merge_stats(t, branch_deltas)
            with torch.no_grad():
                delta = torch.stack([item["safe_delta"] for item in branch_deltas]).sum(dim=0)
                self.qkv.weight.add_(delta.to(device, dtype))
                self.S_lora[t].B.weight.zero_()
                if self.P_lora[t] is not None:
                    self.P_lora[t].B.weight.zero_()
            self.S_lora[t].A.weight.requires_grad_(False)
            self.S_lora[t].B.weight.requires_grad_(False)
            if self.P_lora[t] is not None:
                self.P_lora[t].A.weight.requires_grad_(False)
                self.P_lora[t].B.weight.requires_grad_(False)

        return None
        # deltas = [] # 收集要合并进 qkv.weight 的所有 LoRA 增量
        # if not self.use_slora and not self.use_plora:
        #     deltas.append(masked_delta(self.S_lora[t], 1.0, isolated=False))
        # else:
        #     if self.use_slora or t == 0:
        #         deltas.append(masked_delta(self.S_lora[t], float(self.slora_gamma), isolated=False))
        #     if t > 0 and self.use_plora and self.P_lora[t] is not None:
        #         deltas.append(masked_delta(self.P_lora[t], float(self.plora_gamma), isolated=True))
        # # 打印 deltas 的摘要信息
        # logging.info("Task %d: merging %s LoRA deltas into qkv.weight", t,
        #              [f"shape={d.shape}, mean={d.mean().item():.6f}" for d in deltas])
        # # logging.info("Task %d: merging %s LoRA deltas into qkv.weight, deltas=%s", t, deltas, deltas)
        # if deltas:
        #     with torch.no_grad():
        #         delta = torch.stack(deltas).sum(dim=0)
        #         q, k, v = delta.chunk(3, dim=0)
        #         logging.info("Task %d: Delta total shape=%s, Q norm=%.6f, K norm=%.6f, V norm=%.6f",
        #                      t, delta.shape, q.norm().item(), k.norm().item(), v.norm().item())
        #         self.qkv.weight.add_(delta.to(device, dtype))
        #         self.S_lora[t].B.weight.zero_()
        #         if self.P_lora[t] is not None:
        #             self.P_lora[t].B.weight.zero_()
        #     self.S_lora[t].A.weight.requires_grad_(False) # 当前 task 结束后，冻结 S_lora 的 A 和 B，避免下一轮训练继续修改它们
        #     self.S_lora[t].B.weight.requires_grad_(False)
        #     if self.P_lora[t] is not None:
        #         self.P_lora[t].A.weight.requires_grad_(False)
        #         self.P_lora[t].B.weight.requires_grad_(False)
        # # W0_new = W0_old + safe_delta_s + safe_delta_p
        # return None
