import torch

from methods.dlora import Learner as DLoraLearner

from .attention import Attention_LoRA
from .network import MANet


class Learner(DLoraLearner):
    network_cls = MANet
    attention_cls = Attention_LoRA

    def __init__(self, args):
        removed_options = (
            "dual_mask_alpha_calibration",
            "dual_mask_competence_margin_scale",
            "dual_mask_competence_mix_lambda",
            "dual_mask_old_overlap_enabled",
            "dual_mask_conflict_adaptive",
            "dual_mask_conflict_coverage_adaptive",
            "dual_mask_conflict_task_adaptive",
            "dual_mask_conflict_energy50",
            "dual_mask_grad_alpha",
            "dual_mask_grad_batches",
            "dual_mask_task_relevance_enabled",
            "dual_mask_task_relevance_batches",
            "dual_mask_task_coverage",
            "dual_mask_spectral_conflict_adaptive",
            "dual_mask_static_w0",
            "dual_mask_plasticity_rank_only",
            "dual_mask_plasticity_diagnostics",
        )
        stale_options = [name for name in removed_options if name in args]
        if stale_options:
            raise ValueError(
                "Removed DualMask options: {}. Delete them from the config or "
                "--set overrides.".format(", ".join(stale_options))
            )
        super().__init__(args)
        for layer_idx, module in enumerate(self._iter_lora_modules()):
            module.layer_idx = layer_idx

    # 微调的是当前 task 正在训练的 LoRA 参数 | 惩罚 safe_delta 在 W0 重要区域太大 | safe_delta 在动态 conflict 区域太大
    def _extra_training_loss(self):
        reg_weight = float(self.args.get("dual_mask_reg_weight", 0.0)) # 0.01
        self._last_training_loss_metrics = {}
        if reg_weight <= 0.0:
            return None

        losses = []
        for module in self._iter_lora_modules():
            task = self._cur_task
            if task < 0 or module.S_lora[task] is None:
                continue
            if self.args.get("use_slora", True):
                losses.append(module._joint_conflict_regularization(module.S_lora[task], isolated=False))
            # if task > 0 and self.args.get("use_plora", True):
            if (
                    task > 0
                    and self.args.get("use_plora", True)
                    and hasattr(module, "P_lora")
                    and module.P_lora[task] is not None
            ):
                losses.append(module._joint_conflict_regularization(module.P_lora[task], isolated=True))

        if not losses:
            return None
        regularization = torch.stack(losses).mean()
        return reg_weight * regularization
