import logging

import torch

from methods.dlora import Learner as DLoraLearner
from models.losses import AngularPenaltySMLoss

from .attention import Attention_LoRA
from .network import MANet

class Learner(DLoraLearner):
    network_cls = MANet
    attention_cls = Attention_LoRA

    # 微调的是当前 task 正在训练的 LoRA 参数 | 惩罚 safe_delta 在 W0 重要区域太大 | safe_delta 在动态 conflict 区域太大
    def _extra_training_loss(self):
        reg_weight = float(self.args.get("dual_mask_reg_weight", 0.0)) # 0.01
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
        return reg_weight * torch.stack(losses).mean()

    def _before_lora_weight_init(self, train_loader):
        mode = str(self.args.get("dual_mask_importance", "svd")).lower()
        if "grad" not in mode:
            return

        grad_batches = int(self.args.get("dual_mask_grad_batches", 1))
        if grad_batches <= 0:
            return

        modules = list(self._iter_lora_modules())
        for module in modules:
            module.clear_gradient_importance()

        previous_requires_grad = []
        for module in modules:
            previous_requires_grad.append(module.qkv.weight.requires_grad)
            module.qkv.weight.requires_grad_(True)

        was_training = self._network.training
        self._network.train()
        loss_cos = AngularPenaltySMLoss(
            loss_type="cosface",
            s=self.scale,
            m=self.margin,
        ).to(self._device)

        used_batches = 0
        for _, inputs, targets in self._iter_new_class_batches(train_loader):
            if used_batches >= grad_batches:
                break

            outputs = self._network(inputs)
            loss = loss_cos(outputs["logits"], targets)

            self._network.zero_grad(set_to_none=True)
            loss.backward()

            for module in modules:
                module.accumulate_qkv_gradient()

            used_batches += 1

        self._network.zero_grad(set_to_none=True)
        for module, requires_grad in zip(modules, previous_requires_grad):
            module.qkv.weight.requires_grad_(requires_grad)
            module.rebuild_dual_masks()

        if not was_training:
            self._network.eval()

        logging.info("Collected dual-mask gradient sensitivity from %s batches.", used_batches)

    def _iter_new_class_batches(self, train_loader):
        for _, inputs, targets in train_loader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            if mask.numel() == 0:
                continue
            inputs = torch.index_select(inputs, 0, mask)
            targets = torch.index_select(targets, 0, mask) - self._known_classes
            yield _, inputs, targets
