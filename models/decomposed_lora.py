import torch
import torch.nn as nn
from typing import Tuple
import math
from copy import deepcopy
import logging

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



def _split_qkv_weight(qkv: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    W = qkv.weight
    C = W.shape[1]
    return W[:C, :], W[C:2*C, :], W[2*C:, :]


def _merge_qkv_weight(qkv: nn.Linear, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    C = qkv.weight.shape[1]
    with torch.no_grad():
        qkv.weight[:C, :].copy_(W_q)
        qkv.weight[C:2*C, :].copy_(W_k)
        qkv.weight[2*C:, :].copy_(W_v)


def _random_fixed_A_init(dim: int, r: int, device, dtype) -> torch.Tensor:
    M = torch.randn(dim, r, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q.T.contiguous()  # (r, dim)
"""
    S_lora.A：
    先由父类进行 QR/正交初始化
    → 随后被 DualMask 的 Kaiming 权重完全覆盖
    → 最终使用 Kaiming
    
    P_lora.A：
    直接通过 _init_A_weight() 进行 Kaiming 初始化
    → 最终使用 Kaiming
"""
def _kaiming_A_init(dim: int, r: int, device, dtype) -> torch.Tensor:
    A = torch.empty(r, dim, device=device, dtype=dtype)
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    return A

def _zero_B_init(dim: int, r: int, device, dtype) -> torch.Tensor:
    return torch.zeros(dim, r, device=device, dtype=dtype)


def _energy_merge(W0:torch.Tensor,
                  B_new: torch.Tensor,
                  A_fixed: torch.Tensor,
                  prev_matrix: torch.Tensor,
                  cur_matrix: torch.Tensor,
                  gamma:float,
                  eps: float, out_type='merge') -> torch.Tensor:

    device = B_new.device
    dtype = B_new.dtype
    r = A_fixed.shape[0]

    P_prev = prev_matrix.to(device=device, dtype=dtype)
    P_cur = cur_matrix.to(device=device, dtype=dtype)

    out_cols = []
    eta_list = []
    for i in range(r):
        if out_type == 'merge':
            a_i = A_fixed[i].unsqueeze(0)
            g_prev = (a_i @ P_prev @ a_i.T).clamp_min(0.0).squeeze() ## 1
            g_cur = (a_i @ P_cur @ a_i.T).clamp_min(0.0).squeeze()

            eta = g_prev / (g_prev + float(gamma) * g_cur + eps)
            col = (1.0 - eta) * B_new[:, i]
            eta_list.append(eta)

        elif out_type == 'add':  #'add'
            col = B_new[:, i]
        out_cols.append(col) # 求得G的缩放矩阵

    if out_type == 'merge':
        eta_list = torch.stack(eta_list)
        msg=('Merging weights: mean:{:.2f} // max:{:.2f} // min:{:.2f}'.format(
            eta_list.mean().item(), eta_list.max().item(), eta_list.min().item()))
        logging.info(msg)

    out_cols = torch.stack(out_cols, dim=1).contiguous()
    return out_cols @ A_fixed  # (dim_out, dim_in)


# def drift_regularization(A_fixed: torch.Tensor, B_new: torch.Tensor, ## A:r, D
#                          prev_matrix: torch.Tensor, cur_matrix: torch.Tensor, eps: float) -> torch.Tensor:
#
#     device = B_new.device
#     dtype = B_new.dtype
#     r = A_fixed.shape[0]
#
#     P_prev = prev_matrix.to(device=device, dtype=dtype)
#     P_cur = cur_matrix.to(device=device, dtype=dtype)
#
#     G_prev = A_fixed @ P_prev @ A_fixed.T
#     G_cur = A_fixed @ P_cur @ A_fixed.T ## r, r
#
#     with torch.cuda.amp.autocast(enabled=False):
#         mat = (G_prev.float() + G_cur.float())
#         I = torch.eye(mat.size(-1), device=mat.device, dtype=mat.dtype)
#         mat = mat + (eps + 1e-5) * I
#         inv_mat = torch.linalg.pinv(mat)
#         loss = torch.trace(B_new @ (inv_mat @ G_prev.float()) @ B_new.T)
#
#         return loss.to(G_prev.dtype)


class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, r=64, n_tasks=10, eps=1e-12):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.qkv_bias = qkv_bias
        self.rank = r
        # self.p_rank = r
        self.n_tasks = n_tasks # 20 任务数
        self.eps = float(eps)

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.S_lora = nn.ModuleList([None for _ in range(n_tasks)])
        # self.P_lora = nn.ModuleList([None for _ in range(n_tasks)])

        self.sigmoid = torch.nn.Sigmoid()
        
        # self.n_cur_matrix = 0
        # self.cur_matrix = torch.zeros(dim, dim)
        # self.n_init_matrix = 0
        # self.init_matrix = torch.zeros(dim, dim)
        # self.cur_feats = []

        # self.cur_output = torch.zeros(dim, dim)

        # self.prev_matrix = torch.zeros(dim, dim)
        # self.prev_output = torch.zeros(dim, dim)
        self.cur_task = 0

        # self.merge_gamma = 5.0
        # self.lora_eps = 1e-5

        self.feature_list = []

    def _init_params(self, args):
        self.args = args
        self.use_slora: bool = args["use_slora"]
        self.use_plora: bool = args["use_plora"]
        msg = f'Use slora:{self.use_slora} and Use plora:{self.use_plora}'
        print(msg)
        logging.info(msg)

    # def _init_params(self, args):
    #
    #     self.args = args
    #     self.lora_eps = args["lora_eps"] # 0.1  没有用到
    #     self.slora_gamma = args["slora_gamma"] # 0.5
    #     self.plora_gamma = args["plora_gamma"] # 1.0
    #     self.merge_gamma = args["merge_gamma"] # 3.0
    #     self.use_slora:bool = args["use_slora"]
    #     self.use_plora:bool = args["use_plora"]
    #
    #     # if self.use_slora and self.use_plora and args["avg"]:
    #     #     self.slora_gamma = self.slora_gamma * 0.5; self.plora_gamma = self.plora_gamma * 0.5
    #
    #     msg = f'Use slora:{self.use_slora} and Use plora:{self.use_plora};' +\
    #           'LoRA eps:{:.3f}; SLoRA weight | PLoRA weight:{:.3f} | {:.3f}; Merge gamma: {:.3f}'.format(
    #           self.lora_eps, self.slora_gamma, self.plora_gamma, self.merge_gamma)
    #     print(msg)
    #     logging.info(msg)
    #
    #     return

    # def _acc_cov(self, x: torch.Tensor, which: str):
    #     xt = x.detach()
    #     B, N, C = xt.shape ## B, 197, 768
    #     # self.feature_list.append(xt.cpu())
    #     X = xt.reshape(B * N, C)
    #     # X = xt[:, 0, :]  # Extract only the CLS token [B, C]
    #     cov = X.T @ X
    #     if which == "clean":
    #         total = self.n_init_matrix * self.init_matrix + cov.cpu()
    #         self.n_init_matrix += B * N
    #         # self.n_init_matrix += B
    #         self.init_matrix = total / max(self.n_init_matrix, 1)
    #     elif which == 'cur':
    #         total = self.n_cur_matrix * self.cur_matrix + cov.cpu()
    #         self.n_cur_matrix += B * N
    #         # self.n_cur_matrix += B
    #         self.cur_matrix = total / max(self.n_cur_matrix, 1)
    #     return
    
    def _class_wise_cov(self, targets: torch.Tensor):

        # feature_mat = torch.cat(self.feature_list, dim=0)
        # targets = torch.cat(targets, dim=0).cpu() ## Nt, 197, 768

        # cov_list = []
        # for label in targets.unique():
        #     cls_features = feature_mat[targets == label] ## Nc, 197, 768
        #     Bc, N, C = cls_features.shape
        #     cls_features = cls_features.reshape(Bc * N, C)
        #     cls_cov = (cls_features.T @ cls_features) / Bc
        #     cov_list.append(cls_cov)

        # self.cur_matrix = torch.stack(cov_list).mean(dim=0)
        self.feature_list = []
        return


    def before_task(self, task: int):

        t = int(task)
        self.cur_task = t
        device = next(self.parameters()).device
        dtype = self.qkv.weight.dtype
        rs = self.rank # 64
        # rp = self.p_rank # 64

        # init P_q / P_v
        A_rand_pq = _random_fixed_A_init(self.dim, rs, device, dtype)  # 随机QR分解正交初始化 A--Q  [rs,768]
        B_zero_pq = _zero_B_init(self.dim*3, rs, device, dtype) # 初始化 B 为全 0  # [768*3=2304,rs]
        self.S_lora[t] = FrozenA_TrainableB(self.dim, self.dim*3, rs, A_rand_pq, B_zero_pq, device=device, dtype=dtype)

        # 动态挂载当前 Task 的隔离分支（P-LoRA / Particular-LoRA）
        # A_rand_pq = _random_fixed_A_init(self.dim, rp, device, dtype)
        # B_zero_pq = _zero_B_init(self.dim*3, rp, device, dtype)
        # self.P_lora[t] = FrozenA_TrainableB(self.dim, self.dim*3, rp, A_rand_pq, B_zero_pq, device=device, dtype=dtype)

        # freeze backbone  双重保险
        for p in self.qkv.parameters(): p.requires_grad_(False)
        for p in self.proj.parameters(): p.requires_grad_(False)

    def set_task_and_stage(self, task: int, layer_idx:int, stage:int = 0):

        self.cur_task = int(task)
        for p in self.qkv.parameters(): p.requires_grad_(False)
        for p in self.proj.parameters(): p.requires_grad_(False)

        ## 冻结之前所有 task 的 LoRA
        for t in range(task):
            self.S_lora[t].A.weight.requires_grad_(False)
            self.S_lora[t].B.weight.requires_grad_(False)

        ## sequential tuning (vanilla baseline)
        if not self.use_slora:
            self.S_lora[task].A.weight.requires_grad_(True)
            self.S_lora[task].B.weight.requires_grad_(True)

        ## detach previous loras
        # for t in range(task):
        #     self.S_lora[t].A.weight.requires_grad_(False)
        #     self.S_lora[t].B.weight.requires_grad_(False)
        #     self.P_lora[t].A.weight.requires_grad_(False)
        #     self.P_lora[t].B.weight.requires_grad_(False)
        #
        # ## train A and B for the first task
        # if task == 0:
        #     self.S_lora[task].A.weight.requires_grad_(True)
        #     self.S_lora[task].B.weight.requires_grad_(True)
        #     self.P_lora[task].A.weight.requires_grad_(False)
        #     self.P_lora[task].B.weight.requires_grad_(False)
        # else:
        #     self.S_lora[task].A.weight.requires_grad_(False)
        #     self.S_lora[task].B.weight.requires_grad_(False)
        #     self.P_lora[task].A.weight.requires_grad_(False)
        #     self.P_lora[task].B.weight.requires_grad_(False)

            ## sequential tuning
            # if not self.use_slora and not self.use_plora:
            #     self.S_lora[task].A.weight.requires_grad_(True)
            #     self.S_lora[task].B.weight.requires_grad_(True)
            # ## ours
            # else:
            #     if self.use_slora:
            #         self.S_lora[task].B.weight.requires_grad_(True)
            #     if self.use_plora:
            #         self.P_lora[task].B.weight.requires_grad_(True)

        return


    def after_task(self, task: int):

        t = int(task)
        device = next(self.parameters()).device
        dtype = self.qkv.weight.dtype

        # def merge(W0, A, B, out_type:str='merge'):
        #     W_merged = _energy_merge(W0, B, A, #
        #                              self.prev_matrix,
        #                              self.cur_matrix,
        #                              self.merge_gamma,
        #                              self.eps, out_type=out_type)
        #     return (W0 + W_merged.clone()).contiguous()
        
        # if self.use_slora or self.use_plora:
        #     slora_gamma = float(self.slora_gamma); plora_gamma = float(self.plora_gamma)
        #
        #     if task == 0:
        #         W = merge(self.qkv.weight,
        #                   self.S_lora[t].A_weight.detach(),
        #                   slora_gamma*self.S_lora[t].B_weight.detach(), out_type='add')
        #     elif task >= 1:
        #         W = merge(self.qkv.weight,  #  共享分支 S_lora[t]  → 使用 out_type='merge' 由于前面已经有了历史旧知识，共享方向上的改动容易干扰旧任务。因此必须使用能量比值
        #                   self.S_lora[t].A_weight.detach(),
        #                   slora_gamma*self.S_lora[t].B_weight.detach(), out_type='merge')
        #         W = merge(W,      # add:（直接全量相加）：不进行能量比值衰减，将 LoRA 增量 100% 完整无损地加进主干权重（即缩放系数为 1.0
        #                   self.P_lora[t].A_weight.detach(),  # 这个方向上旧任务的干扰能量几乎为 0（天然零干扰）
        #                   plora_gamma*self.P_lora[t].B_weight.detach(), out_type='add')
        #
        #     self.qkv.weight.data.copy_(W.to(device, dtype))
        #     self.S_lora[t].B.weight.zero_()
        #     self.S_lora[t].B.weight.requires_grad_(False)
        #     self.P_lora[t].B.weight.zero_()
        #     self.P_lora[t].B.weight.requires_grad_(False)

        # Vanilla baseline: 直接把 S_lora 的 BA 加进 W0
        delta_W = (self.S_lora[t].B_weight @ self.S_lora[t].A_weight).detach()
        self.qkv.weight.data.add_(delta_W.to(device, dtype))
        self.S_lora[t].B.weight.zero_()
        self.S_lora[t].B.weight.requires_grad_(False)

        return None

    def _contrib_from_units(self, x: torch.Tensor, t_idx: int) -> torch.Tensor:
        unit_S = self.S_lora[t_idx]
        # unit_P = self.P_lora[t_idx]
        # if not self.use_slora and not self.use_plora:
        x = unit_S(x)
        # else:
        #     slora_gamma = float(self.slora_gamma); plora_gamma = float(self.plora_gamma)
        #
        #     if t_idx == 0:
        #         x = slora_gamma*unit_S(x) # 0.25*共享lora
        #     else:
        #         x = slora_gamma*unit_S(x) + plora_gamma*unit_P(x) # 0.25*共享lora + 0.25*任务特定lora
        return x
    
    # def _get_reg_loss(self, task):
    #
    #     t = int(task)
    #
    #     loss =\
    #     drift_regularization(self.S_lora[t].A_weight.detach(),
    #                          self.S_lora[t].B_weight.detach(),
    #                          self.prev_matrix,
    #                          self.cur_matrix,
    #                          self.eps)
    #
    #     return loss


    def forward(self, x: torch.Tensor, task: int, register_hook: bool = False,
                get_feat: bool = False, get_cur_feat: bool = False):

        # if get_cur_feat:
        #     self._acc_cov(x, which="cur")

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

    def _init_lora_weight(self, task, layer_idx:int=0):

        # lora_init_scale = 3.0
        # M_cur, M_prev = self.cur_matrix, self.prev_matrix # 协方差矩阵
        # total_matrix = M_cur + M_prev # 累计协方差矩阵
        # U, S, V = torch.linalg.svd(total_matrix)
        # 不使用原文的lora--直接B初始化为0,A均值分布
        if not self.use_plora and not self.use_slora: ## sequential tuning
            # if task == 0:
            # 所有 task 都用随机初始化
            nn.init.kaiming_uniform_(self.S_lora[task].A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.S_lora[task].B.weight)
            # if task >= 1:
            #     self.S_lora[task].A.weight.data.copy_(self.S_lora[task-1].A.weight.data)
            #     self.S_lora[task].B.weight.data.copy_(self.S_lora[task-1].B.weight.data)
