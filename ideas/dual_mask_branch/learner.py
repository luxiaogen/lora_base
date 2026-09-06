import torch
import torch.nn.functional as F
from methods.dlora import Learner as DLoraLearner

from .attention import Attention_LoRA
from .network import MANet


class Learner(DLoraLearner):
    network_cls = MANet
    attention_cls = Attention_LoRA

    @staticmethod
    def _selective_functional_anchor_loss(
            current_features,
            w0_features,
            targets,
            prototypes,
            min_margin,
            tolerance,
    ):
        current_features = F.normalize(current_features, dim=1)
        w0_features = F.normalize(w0_features.detach(), dim=1)
        prototypes = F.normalize(prototypes.detach(), dim=1)

        current_scores = current_features @ prototypes.t()
        w0_scores = w0_features @ prototypes.t()
        class_mask = F.one_hot(
            targets,
            num_classes=prototypes.shape[0],
        ).bool()

        current_positive = current_scores.gather(1, targets[:, None]).squeeze(1)
        current_negative = current_scores.masked_fill(
            class_mask,
            float("-inf"),
        ).max(dim=1).values
        w0_positive = w0_scores.gather(1, targets[:, None]).squeeze(1)
        w0_negative = w0_scores.masked_fill(
            class_mask,
            float("-inf"),
        ).max(dim=1).values

        current_margin = current_positive - current_negative
        w0_margin = w0_positive - w0_negative
        selected = (w0_scores.argmax(dim=1) == targets) & (
            w0_margin >= float(min_margin)
        )
        violation = F.relu(
            w0_margin.detach() - float(tolerance) - current_margin
        )

        selected_ratio = selected.float().mean()
        if selected.any():
            selected_violation = violation[selected]
            weights = w0_margin[selected].detach().clamp_min(0.0)
            loss = (
                weights * selected_violation.square()
            ).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
            violation_ratio = (selected_violation > 0).float().mean()
        else:
            loss = current_features.sum() * 0.0
            violation_ratio = selected_ratio

        return loss, {
            "selected_ratio": selected_ratio.detach(),
            "violation_ratio": violation_ratio.detach(),
        }

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

        if (
            bool(args.get("dual_mask_functional_merge_calibration", False))
            and str(args.get("dual_mask_conflict_merge_mode", "suppress")).lower()
            != "suppress"
        ):
            raise ValueError(
                "dual_mask_functional_merge_calibration requires "
                "dual_mask_conflict_merge_mode='suppress'."
            )

        super().__init__(args)
        for layer_idx, module in enumerate(self._iter_lora_modules()):
            module.layer_idx = layer_idx

    def _extra_training_context(self, inputs, targets, epoch):
        enabled = bool(
            self.args.get("dual_mask_selective_anchor_enabled", False)
        )
        weight = float(
            self.args.get("dual_mask_selective_anchor_weight", 0.0)
        )
        start_epoch = int(
            self.args.get("dual_mask_selective_anchor_start_epoch", 0)
        )
        if (
                not enabled
                or weight <= 0.0
                or self._cur_task != 0
                or int(epoch) < start_epoch
        ):
            return {}

        was_training = self._network.training
        self._network.eval()
        try:
            with self._pretrained_anchor_context(), torch.no_grad():
                w0_features = self._network(inputs)["features"].detach()
        finally:
            self._network.train(was_training)
        return {"selective_anchor_w0_features": w0_features}

    # 微调的是当前 task 正在训练的 LoRA 参数 | 惩罚 safe_delta 在 W0 重要区域太大 | safe_delta 在动态 conflict 区域太大
    # def _extra_training_loss(self):
    def _extra_training_loss(
            self,
            output=None,
            inputs=None,
            targets=None,
            epoch=None,
            batch_context=None,
    ):
        reg_weight = float(self.args.get("dual_mask_reg_weight", 0.0)) # 0.01

        anchor_enabled = bool(
            self.args.get("dual_mask_anchor_reg_enabled", False)
        )
        anchor_weight = float(
            self.args.get("dual_mask_anchor_reg_weight", 0.0)
        )

        anchor_task0_only = bool(
            self.args.get("dual_mask_anchor_reg_task0_only", False)
        )
        anchor_applies = (
            anchor_enabled
            and anchor_weight > 0.0
            and (not anchor_task0_only or self._cur_task == 0)
        )

        safe_residual_enabled = bool(
            self.args.get("dual_mask_safe_residual_enabled", False)
        )
        safe_residual_weight = float(
            self.args.get("dual_mask_safe_residual_weight", 0.0)
        )
        safe_residual_applies = (
            safe_residual_enabled and safe_residual_weight > 0.0
        )

        selective_anchor_enabled = bool(
            self.args.get("dual_mask_selective_anchor_enabled", False)
        )
        selective_anchor_weight = float(
            self.args.get("dual_mask_selective_anchor_weight", 0.0)
        )
        selective_anchor_start_epoch = int(
            self.args.get("dual_mask_selective_anchor_start_epoch", 0)
        )
        selective_anchor_ramp_epochs = max(
            1,
            int(self.args.get("dual_mask_selective_anchor_ramp_epochs", 1)),
        )
        current_epoch = 0 if epoch is None else int(epoch)
        if current_epoch < selective_anchor_start_epoch:
            selective_anchor_ramp = 0.0
        else:
            selective_anchor_ramp = min(
                1.0,
                (current_epoch - selective_anchor_start_epoch + 1)
                / selective_anchor_ramp_epochs,
            )
        selective_anchor_applies = (
            selective_anchor_enabled
            and selective_anchor_weight > 0.0
            and self._cur_task == 0
            and selective_anchor_ramp > 0.0
        )

        if safe_residual_applies and isinstance(
                self._network,
                torch.nn.DataParallel,
        ):
            raise RuntimeError(
                "dual_mask_safe_residual is currently supported only for "
                "single-GPU training; nn.DataParallel would drop its "
                "forward-side auxiliary loss."
            )

        self._last_training_loss_metrics = {}
        # if reg_weight <= 0.0:
        # if reg_weight <= 0.0 and (not anchor_enabled or anchor_weight <= 0.0):
        # if reg_weight <= 0.0 and not anchor_applies:
        if (
                reg_weight <= 0.0
                and not anchor_applies
                and not safe_residual_applies
                and not selective_anchor_applies
        ):
            return None

        # losses = []
        # for module in self._iter_lora_modules():
        #     task = self._cur_task
        #     if task < 0 or module.S_lora[task] is None:
        #         continue
        #     if self.args.get("use_slora", True):
        #         losses.append(module._joint_conflict_regularization(module.S_lora[task], isolated=False))
        #     # if task > 0 and self.args.get("use_plora", True):
        #     if (
        #             task > 0
        #             and self.args.get("use_plora", True)
        #             and hasattr(module, "P_lora")
        #             and module.P_lora[task] is not None
        #     ):
        #         losses.append(module._joint_conflict_regularization(module.P_lora[task], isolated=True))

        modules = [
            module
            for module in self._iter_lora_modules()
            if self._cur_task >= 0 and module.S_lora[self._cur_task] is not None
        ]
        weighted_losses = []

        # if not losses:
        if reg_weight > 0.0:
            conflict_losses = []
            for module in modules:
                task = self._cur_task
                if self.args.get("use_slora", True):
                    conflict_losses.append(
                        module._joint_conflict_regularization(
                            module.S_lora[task],
                            isolated=False,
                        )
                    )
                if (
                        task > 0
                        and self.args.get("use_plora", True)
                        and hasattr(module, "P_lora")
                        and module.P_lora[task] is not None
                ):
                    conflict_losses.append(
                        module._joint_conflict_regularization(
                            module.P_lora[task],
                            isolated=True,
                        )
                    )
            if conflict_losses:
                weighted_losses.append(
                    reg_weight * torch.stack(conflict_losses).mean()
                )

        # if anchor_enabled and anchor_weight > 0.0 and modules:
        if anchor_applies and modules:
            anchor_regularization = torch.stack(
                [module.anchor_regularization() for module in modules]
            ).mean()
            weighted_anchor = anchor_weight * anchor_regularization
            weighted_losses.append(weighted_anchor)
            # self._last_training_loss_metrics = {
            self._last_training_loss_metrics.update({
                "anchor_reg": anchor_regularization.detach(),
                "anchor_reg_weighted": weighted_anchor.detach(),
            # }
            })
        if safe_residual_applies and modules:
          safe_residual_losses = [
              module.safe_residual_regularization()
              for module in modules
          ]
          safe_residual_losses = [
              loss for loss in safe_residual_losses if loss is not None
          ]
          if safe_residual_losses:
              safe_residual = torch.stack(safe_residual_losses).mean()
              weighted_safe_residual = (
                  safe_residual_weight * safe_residual
              )
              weighted_losses.append(weighted_safe_residual)
              self._last_training_loss_metrics.update({
                  "safe_residual": safe_residual.detach(),
                  "safe_residual_weighted": weighted_safe_residual.detach(),
              })
        if selective_anchor_applies:
            if output is None or targets is None or batch_context is None:
                raise RuntimeError(
                    "selective functional anchor requires training output, "
                    "targets, and the W0 batch context"
                )
            w0_features = batch_context.get("selective_anchor_w0_features")
            if w0_features is None:
                raise RuntimeError(
                    "selective functional anchor W0 features were not prepared"
                )
            prototype_ids = sorted(self._w0_class_means)
            if prototype_ids != list(range(len(prototype_ids))):
                raise RuntimeError(
                    "Task-0 W0 prototypes must use contiguous local class ids"
                )
            current_features = output["features"]
            prototypes = torch.stack([
                self._w0_class_means[class_id]
                for class_id in prototype_ids
            ]).to(
                device=current_features.device,
                dtype=current_features.dtype,
            )
            selective_anchor, selective_metrics = (
                self._selective_functional_anchor_loss(
                    current_features,
                    w0_features,
                    targets,
                    prototypes,
                    min_margin=float(self.args.get(
                        "dual_mask_selective_anchor_min_margin",
                        0.05,
                    )),
                    tolerance=float(self.args.get(
                        "dual_mask_selective_anchor_tolerance",
                        0.05,
                    )),
                )
            )
            weighted_selective_anchor = (
                selective_anchor_weight
                * selective_anchor_ramp
                * selective_anchor
            )
            weighted_losses.append(weighted_selective_anchor)
            self._last_training_loss_metrics.update({
                "selective_anchor": selective_anchor.detach(),
                "selective_anchor_weighted": (
                    weighted_selective_anchor.detach()
                ),
                "selective_anchor_selected_ratio": (
                    selective_metrics["selected_ratio"]
                ),
                "selective_anchor_violation_ratio": (
                    selective_metrics["violation_ratio"]
                ),
                "selective_anchor_ramp": current_features.new_tensor(
                    selective_anchor_ramp
                ),
            })



        if not weighted_losses:
            return None
        # regularization = torch.stack(losses).mean()
        # return reg_weight * regularization
        return torch.stack(weighted_losses).sum()
