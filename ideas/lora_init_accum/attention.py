import logging

import torch

from models.decomposed_lora import Attention_LoRA as BaseAttentionLoRA
from models.decomposed_lora import FrozenA_TrainableB


class Attention_LoRA(BaseAttentionLoRA): # 继承d_lora的属性和方法
    """
        FRLoRA风格的主空间初始化与残差累积
        有效注意力权重保持为 W_hat + BA 的形式。在首次使用时，权重 W 会被分解一次，得到由前 top 个奇异分量构成的 B₀A₀，
        同时将 qkv.weight 调整为 W_hat = W - B₀A₀。每个任务都从相同的 B₀、A₀ 开始，随后训练得到的残差 BA - B₀A₀
        会被累积到 W_hat 中
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("frlora_A0", torch.zeros(self.rank, self.dim), persistent=False) # [32,768]
        self.register_buffer("frlora_B0", torch.zeros(self.dim * 3, self.rank), persistent=False) # [768*3=2304,32]
        self.register_buffer("frlora_base_delta", torch.zeros(self.dim * 3, self.dim), persistent=False) # [768*3=2304,768]
        self.register_buffer("cumulative_lora_delta", torch.zeros(self.dim * 3, self.dim), persistent=False) # [768*3=2304,768]
        self.register_buffer("last_residual_delta", torch.zeros(self.dim * 3, self.dim), persistent=False) # [768*3=2304,768]
        self.frlora_initialized = False
        self.track_accumulated_delta = True
        self.residual_scale = 1.0


    def _init_params(self, args):
        super()._init_params(args)
        self.use_slora = False
        # self.use_plora = False
        self.track_accumulated_delta = bool(args.get("track_accumulated_delta", True))
        self.residual_scale = float(args.get("residual_scale", 1.0))
        logging.info(
            "Idea1 FRLoRA init/accum enabled: track_accumulated_delta=%s, residual_scale=%.4f",
            self.track_accumulated_delta,
            self.residual_scale,
        )
    # 从W0中初始化得到B0和A0，作为LoRA的初始权重
    def _principal_lora_from_weight(self):
        weight = self.qkv.weight.detach().float() # [2304,768]
        u, s, vh = torch.linalg.svd(weight, full_matrices=False) # [2304,768] [768] [768,768]
        rank = min(int(self.rank), s.numel()) # --- lora的rank
        sqrt_s = s[:rank].clamp_min(0.0).sqrt() #
        b0 = u[:, :rank] * sqrt_s.unsqueeze(0)   # B0: U [: r]S[: r],
        a0 = sqrt_s.unsqueeze(1) * vh[:rank, :] # A0:  S[: r]V [: r]
        return a0, b0

    def _ensure_frlora_base(self):
        if self.frlora_initialized:
            return

        device = self.qkv.weight.device
        dtype = self.qkv.weight.dtype
        a0, b0 = self._principal_lora_from_weight()
        a0 = a0.to(device=device, dtype=dtype) # [32,768]
        b0 = b0.to(device=device, dtype=dtype) # [2304,32]
        base_delta = b0 @ a0 # [2304,768]

        with torch.no_grad():
            self.frlora_A0.copy_(a0.to(self.frlora_A0.device, dtype=self.frlora_A0.dtype))
            self.frlora_B0.copy_(b0.to(self.frlora_B0.device, dtype=self.frlora_B0.dtype))
            self.frlora_base_delta.copy_(
                base_delta.to(self.frlora_base_delta.device, dtype=self.frlora_base_delta.dtype)
            )   # B0 @ A0 ≈ W0 的 top-rank 主成分
            self.qkv.weight.sub_(base_delta) # W' = W0 - B0A0

        self.frlora_initialized = True
        logging.info(
            "Idea1 FRLoRA base initialized: rank=%s, base_delta_norm=%.6f, W_hat_norm=%.6f",
            self.rank,
            base_delta.norm().item(),
            self.qkv.weight.detach().norm().item(),
        )

    def before_task(self, task: int):
        self._ensure_frlora_base()

        t = int(task)
        self.cur_task = t
        device = next(self.parameters()).device
        dtype = self.qkv.weight.dtype
        a0 = self.frlora_A0.to(device=device, dtype=dtype)
        b0 = self.frlora_B0.to(device=device, dtype=dtype)
        self.S_lora[t] = FrozenA_TrainableB(
            self.dim,
            self.dim * 3,
            self.rank,
            a0,
            b0,
            device=device,
            dtype=dtype,
        )

        for p in self.qkv.parameters():
            p.requires_grad_(False)
        for p in self.proj.parameters():
            p.requires_grad_(False)

    def _init_lora_weight(self, task, layer_idx: int = 0):
        self._ensure_frlora_base()
        unit = self.S_lora[int(task)] # 本任务的LoRA单元
        device = unit.A.weight.device
        dtype = unit.A.weight.dtype
        with torch.no_grad():
            unit.A.weight.copy_(self.frlora_A0.to(device=device, dtype=dtype))
            unit.B.weight.copy_(self.frlora_B0.to(device=unit.B.weight.device, dtype=unit.B.weight.dtype))

        logging.info("Idea1 layer %s initialized from principal singular space.", layer_idx)

    def set_task_and_stage(self, task: int, layer_idx: int, stage: int = 0):
        self.cur_task = int(task)
        for p in self.qkv.parameters():
            p.requires_grad_(False)
        for p in self.proj.parameters():
            p.requires_grad_(False)

        for idx, unit in enumerate(self.S_lora):
            if unit is None:
                continue
            trainable = idx == int(task)
            unit.A.weight.requires_grad_(trainable) # 当前 task 的 A/B 训练，旧 task 的 A/B 冻结
            unit.B.weight.requires_grad_(trainable)

    def _contrib_from_units(self, x: torch.Tensor, t_idx: int) -> torch.Tensor:
        # return self.S_lora[t_idx](x)
        unit_out = self.S_lora[t_idx](x)
        if self.residual_scale == 1.0:
            return unit_out

        a0 = self.frlora_A0.to(device=x.device, dtype=x.dtype)
        b0 = self.frlora_B0.to(device=x.device, dtype=x.dtype)
        base_out = torch.nn.functional.linear(torch.nn.functional.linear(x, a0), b0)
        return base_out + self.residual_scale * (unit_out - base_out)

    def after_task(self, task: int):
        t = int(task)
        unit = self.S_lora[t]
        device = self.qkv.weight.device
        dtype = self.qkv.weight.dtype

        with torch.no_grad():
            learned_delta = unit.B_weight.detach() @ unit.A_weight.detach() # B_t A_t
            base_delta = self.frlora_base_delta.to(device=learned_delta.device, dtype=learned_delta.dtype) # B0 A0
            residual_delta = learned_delta - base_delta # Delta W^t = B_t A_t - B0 A0 更新的部分
            # self.qkv.weight.add_(residual_delta.to(device=device, dtype=dtype)) # W_hat^t = W_hat^{t-1} + Delta W^t
            scaled_residual_delta = self.residual_scale * residual_delta
            self.qkv.weight.add_(scaled_residual_delta.to(device=device, dtype=dtype))

            # 记录用的 buffer，不参与 forward
            self.last_residual_delta.copy_(
                residual_delta.to(
                    device=self.last_residual_delta.device,
                    dtype=self.last_residual_delta.dtype,
                )
            )
            if self.track_accumulated_delta:
                self.cumulative_lora_delta.add_(
                    scaled_residual_delta.to(
                        device=self.cumulative_lora_delta.device,
                        dtype=self.cumulative_lora_delta.dtype,
                    )
                )

            unit.A.weight.copy_(self.frlora_A0.to(device=unit.A.weight.device, dtype=unit.A.weight.dtype))
            unit.B.weight.copy_(self.frlora_B0.to(device=unit.B.weight.device, dtype=unit.B.weight.dtype))
            unit.A.weight.requires_grad_(False)
            unit.B.weight.requires_grad_(False)

        logging.info(
            "Idea1 task %s residual norm %.6f, scaled residual norm %.6f, cumulative residual norm %.6f",
            task,
            residual_delta.norm().item(),
            scaled_residual_delta.norm().item(),
            self.cumulative_lora_delta.norm().item(),
        )
        return None

    def _process_feature_mat(self):
        return None
