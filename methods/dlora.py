import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import copy
import logging
import numpy as np
from tqdm import tqdm

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from models.network import MANet
from models.attention import Attention_LoRA

from utils.schedulers import CosineSchedule
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.toolkit import count_parameters
from models.losses import AngularPenaltySMLoss
from contextlib import ExitStack


class Learner(BaseLearner):
    @staticmethod
    def _selective_functional_anchor_loss(current_features,w0_features,targets,prototypes,min_margin,tolerance,):
        current_features = F.normalize(current_features, dim=1)
        w0_features = F.normalize(w0_features.detach(), dim=1)
        prototypes = F.normalize(prototypes.detach(), dim=1)

        current_scores = current_features @ prototypes.t()
        w0_scores = w0_features @ prototypes.t()
        class_mask = F.one_hot(targets,num_classes=prototypes.shape[0],).bool()

        current_positive = current_scores.gather(1, targets[:, None]).squeeze(1)
        current_negative = current_scores.masked_fill(class_mask,float("-inf"),).max(dim=1).values

        w0_positive = w0_scores.gather(1, targets[:, None]).squeeze(1)
        w0_negative = w0_scores.masked_fill(class_mask,float("-inf"),).max(dim=1).values

        current_margin = current_positive - current_negative
        w0_margin = w0_positive - w0_negative
        selected = (w0_scores.argmax(dim=1) == targets) & (w0_margin >= float(min_margin))

        violation = F.relu(w0_margin.detach() - float(tolerance) - current_margin)

        selected_ratio = selected.float().mean()
        if selected.any():
            selected_violation = violation[selected]
            weights = w0_margin[selected].detach().clamp_min(0.0)
            loss = (weights * selected_violation.square()).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
            violation_ratio = (selected_violation > 0).float().mean()
        else:
            loss = current_features.sum() * 0.0
            violation_ratio = selected_ratio

        return loss, {
            "selected_ratio": selected_ratio.detach(),
            "violation_ratio": violation_ratio.detach(),
        }

    def __init__(self, args):
        super().__init__(args)

        self._network = MANet(args)
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module._init_params(args)


        self.args = args
        self.optim = args["optim"]  # sgd
        self.init_epoch = args["init_epoch"]  # 20
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]  # 20
        self.lrate = args["lrate"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.scale = args["scale"]
        self.margin = args["margin"]  # 分类损失函数 CosFace（Large Margin Cosine Loss）中的“角度边距 / 余弦裕度”（Cosine Margin）超参数

        self.total_sessions = args["total_sessions"]  # 任务数
        self.total_classnum = self.args["init_cls"] + self.args["increment"] * (self.total_sessions - 1)
        self.dataset = args["dataset"]
        self.logit_norm = args["logit_norm"]  # 用于CA
        if self.logit_norm == "none":
            self.logit_norm = None
        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.task_sizes = []

        # class prototypes
        self._class_means = None
        self._class_covs = None

        self.acc_matrix = np.zeros((self.total_sessions, self.total_sessions))

        self._w0_class_means = {}
        self._w0_competence = 0.0

        self._w0_competence_new = None
        self._w0_competence_all_seen = None
        self._w0_old_overlap_risk = None

        self._w0_ncm_loss_new = None
        self._w0_plasticity_demand = None

        self._w0_control_competence = None

        self._w0_accuracy_curve = []
        self._feature_drift_curve = []
        self._weight_drift_curve = []

        self._functional_merge_calibration = None

        for layer_idx, module in enumerate(self._iter_lora_modules()):
            module.layer_idx = layer_idx

    def _iter_lora_modules(self):
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                yield module

    def _extra_training_context(self, inputs, targets, epoch):
        enabled = bool(self.args.get("dual_mask_selective_anchor_enabled", False))
        weight = float(self.args.get("dual_mask_selective_anchor_weight", 0.0))
        start_epoch = int(self.args.get("dual_mask_selective_anchor_start_epoch", 0))
        if (not enabled or weight <= 0.0 or self._cur_task != 0 or int(epoch) < start_epoch):
            return {}

        was_training = self._network.training
        self._network.eval()
        try:
            with self._pretrained_anchor_context(), torch.no_grad():
                w0_features = self._network(inputs)["features"].detach()
        finally:
            self._network.train(was_training)
        return {"selective_anchor_w0_features": w0_features}

    def _extra_training_loss(self,output=None,inputs=None,targets=None,epoch=None,batch_context=None,):
        reg_weight = float(self.args.get("dual_mask_reg_weight", 0.0)) # 0.01

        anchor_enabled = bool(self.args.get("dual_mask_anchor_reg_enabled", False))
        anchor_weight = float(self.args.get("dual_mask_anchor_reg_weight", 0.0))

        anchor_task0_only = bool(self.args.get("dual_mask_anchor_reg_task0_only", False))
        anchor_applies = (anchor_enabled and anchor_weight > 0.0 and (not anchor_task0_only or self._cur_task == 0))

        safe_residual_enabled = bool(self.args.get("dual_mask_safe_residual_enabled", False))
        safe_residual_weight = float(self.args.get("dual_mask_safe_residual_weight", 0.0))
        safe_residual_applies = (safe_residual_enabled and safe_residual_weight > 0.0)

        selective_anchor_enabled = bool(self.args.get("dual_mask_selective_anchor_enabled", False))
        selective_anchor_weight = float(self.args.get("dual_mask_selective_anchor_weight", 0.0))
        selective_anchor_start_epoch = int(self.args.get("dual_mask_selective_anchor_start_epoch", 0))
        selective_anchor_ramp_epochs = max(1,int(self.args.get("dual_mask_selective_anchor_ramp_epochs", 1)),)

        current_epoch = 0 if epoch is None else int(epoch)
        if current_epoch < selective_anchor_start_epoch:
            selective_anchor_ramp = 0.0
        else:
            selective_anchor_ramp = min(1.0,(current_epoch - selective_anchor_start_epoch + 1) / selective_anchor_ramp_epochs,)
        selective_anchor_applies = (
            selective_anchor_enabled
            and selective_anchor_weight > 0.0
            and self._cur_task == 0
            and selective_anchor_ramp > 0.0
        )

        self._last_training_loss_metrics = {}
        if (reg_weight <= 0.0
                and not anchor_applies and not safe_residual_applies and not selective_anchor_applies):
            return None

        modules = [
            module
            for module in self._iter_lora_modules()
            if self._cur_task >= 0 and module.S_lora[self._cur_task] is not None
        ]
        weighted_losses = []

        if reg_weight > 0.0:
            conflict_losses = []
            for module in modules:
                task = self._cur_task
                if self.args.get("use_slora", True):
                    conflict_losses.append(module._joint_conflict_regularization(module.S_lora[task],isolated=False,))
                if (task > 0 and self.args.get("use_plora", True) and hasattr(module, "P_lora") and module.P_lora[task] is not None):
                    conflict_losses.append(module._joint_conflict_regularization(module.P_lora[task],isolated=True,))
            if conflict_losses:
                weighted_losses.append(reg_weight * torch.stack(conflict_losses).mean())

        if anchor_applies and modules:
            anchor_regularization = torch.stack([module.anchor_regularization() for module in modules]).mean()
            weighted_anchor = anchor_weight * anchor_regularization
            weighted_losses.append(weighted_anchor)
            self._last_training_loss_metrics.update({"anchor_reg": anchor_regularization.detach(),"anchor_reg_weighted": weighted_anchor.detach(),})
        if safe_residual_applies and modules:
          safe_residual_losses = [
              module.safe_residual_regularization()
              for module in modules
          ]
          safe_residual_losses = [loss for loss in safe_residual_losses if loss is not None]
          if safe_residual_losses:
              safe_residual = torch.stack(safe_residual_losses).mean()
              weighted_safe_residual = (safe_residual_weight * safe_residual)
              weighted_losses.append(weighted_safe_residual)
              self._last_training_loss_metrics.update({
                  "safe_residual": safe_residual.detach(),
                  "safe_residual_weighted": weighted_safe_residual.detach(),
              })
        if selective_anchor_applies:
            w0_features = batch_context.get("selective_anchor_w0_features")
            prototype_ids = sorted(self._w0_class_means)
            current_features = output["features"]
            prototypes = torch.stack([self._w0_class_means[class_id] for class_id in prototype_ids]).to(device=current_features.device, dtype=current_features.dtype)
            selective_anchor, selective_metrics = (
                self._selective_functional_anchor_loss(current_features,w0_features,targets,prototypes,
                    min_margin=float(self.args.get("dual_mask_selective_anchor_min_margin",0.05,)),
                    tolerance=float(self.args.get("dual_mask_selective_anchor_tolerance",0.05,)),
                )
            )
            weighted_selective_anchor = (selective_anchor_weight * selective_anchor_ramp * selective_anchor)
            weighted_losses.append(weighted_selective_anchor)
            self._last_training_loss_metrics.update({
                "selective_anchor": selective_anchor.detach(),
                "selective_anchor_weighted": (weighted_selective_anchor.detach()),
                "selective_anchor_selected_ratio": (selective_metrics["selected_ratio"]),
                "selective_anchor_violation_ratio": (selective_metrics["violation_ratio"]),
                "selective_anchor_ramp": current_features.new_tensor(selective_anchor_ramp),
            })



        if not weighted_losses:
            return None
        return torch.stack(weighted_losses).sum()

    def _backward_and_step(self, task_loss, extra_loss, optimizer, output, targets):
        """Optimization extension point used by experimental learners."""
        loss = task_loss if extra_loss is None else task_loss + extra_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss

    def _pretrained_anchor_context(self):
        """Temporarily switch every LoRA attention layer to immutable W_pre."""
        stack = ExitStack()
        for module in self._iter_lora_modules():
            stack.enter_context(module.use_pretrained_anchor())
        return stack

    def _collect_features(self, loader, use_pretrained_anchor=False):
        """Collect deterministic feature, index, and label tensors."""
        was_training = self._network.training
        self._network.eval()
        indices, features, targets = [], [], []
        context = (self._pretrained_anchor_context() if use_pretrained_anchor else ExitStack())
        with context, torch.no_grad():
            for batch_indices, inputs, batch_targets in loader:
                vectors = self._network.extract_vector(inputs.to(self._device))  # 图片 → ViT / W_pre → 768 维 feature
                indices.append(batch_indices.detach().cpu())  # 样本编号
                features.append(vectors.detach().cpu())  # W_pre 特征
                targets.append(batch_targets.detach().cpu())  # 真实标签
        if was_training:
            self._network.train()
        return torch.cat(indices), torch.cat(features), torch.cat(targets)

    def _collect_anchor_features(self, loader):
        return self._collect_features(loader, use_pretrained_anchor=True)

    def _calibrate_functional_merge(self, loader):
        from utils.dual_mask_metrics import functional_merge_diagnostics, select_functional_merge_candidate

        modules = list(self._iter_lora_modules())
        if not modules:
            return

        tolerance = max(0.0,float(self.args.get("dual_mask_functional_merge_tolerance", 0.05)),)


        base_beta = min(max(float(self.args.get("dual_mask_conflict_strength", 0.5)), 0.0),1.0,)
        candidate_betas = sorted({0.0, base_beta})
        _, anchor_features, _ = self._collect_anchor_features(loader)
        old_prototypes = None
        old_class_ids = None
        if self._known_classes > 0:
            old_class_ids = torch.arange(self._known_classes, dtype=torch.long)
            old_prototypes = torch.stack([self._w0_class_means[int(class_id)] for class_id in old_class_ids])

        candidates = []
        for beta in candidate_betas:
            for module in modules:
                module.set_functional_merge_strength(beta)
            indices, features, targets = self._collect_features(loader)
            metrics = functional_merge_diagnostics(
                anchor_features,
                features,
                targets,
                indices,
                holdout_mod=int(self.args.get("dual_mask_competence_holdout_mod", 5)),
                scale=self.scale,
                old_prototypes=old_prototypes,
                old_class_ids=old_class_ids,
            )
            metrics["beta"] = beta
            candidates.append(metrics)
            logging.info(
                "Task %s functional merge candidate: beta=%.3f, "
                "current_acc=%.2f%%, current_loss=%.6f, "
                "anchor_reference_loss=%.6f, anchor_candidate_loss=%.6f, "
                "anchor_damage=%.6f, eligible=%s (tolerance=%.6f)",
                self._cur_task,
                beta,
                metrics["current_accuracy"] * 100.0,
                metrics["current_loss"],
                metrics["anchor_reference_loss"],
                metrics["anchor_candidate_loss"],
                metrics["anchor_damage"],
                metrics["anchor_damage"] <= tolerance,
                tolerance,
            )

        selected = select_functional_merge_candidate(candidates, tolerance)
        for module in modules:
            module.set_functional_merge_strength(selected["beta"])
        self._functional_merge_calibration = {
            "selected_beta": selected["beta"],
            "selected_current_accuracy": selected["current_accuracy"],
            "selected_current_loss": selected["current_loss"],
            "selected_anchor_damage": selected["anchor_damage"],
        }
        for candidate in candidates:
            candidate_name = "beta_{:03d}".format(int(round(candidate["beta"] * 100.0)))
            self._functional_merge_calibration.update({
                f"{candidate_name}_current_accuracy": candidate["current_accuracy"],
                f"{candidate_name}_current_loss": candidate["current_loss"],
                f"{candidate_name}_anchor_damage": candidate["anchor_damage"],
            })
        logging.info(
            "Task %s functional merge selected beta=%.3f: "
            "current_acc=%.2f%%, current_loss=%.6f, anchor_damage=%.6f",
            self._cur_task,
            selected["beta"],
            selected["current_accuracy"] * 100.0,
            selected["current_loss"],
            selected["anchor_damage"],
        )

    # 测量 W_pre competence
    def _prepare_w0_prototypes(self, loader):
        # from utils.dual_mask_metrics import split_prototype_competence
        from utils.dual_mask_metrics import (split_prototype_competence,split_prototype_ncm_diagnostics,)

        # 训练集中的特征
        indices, features, targets = self._collect_anchor_features(loader)

        ## Ct 是否使用的是所有已见类的原型，还是仅使用当前任务的原型
        use_all_seen_prototypes = bool(self.args.get("dual_mask_competence_all_seen", False))

        competence_metric = str(self.args.get("dual_mask_competence_metric", "accuracy")).lower()

        use_old_overlap_conflict = bool(self.args.get("dual_mask_conflict_old_overlap_adaptive", False))

        

        need_all_seen_competence = (use_all_seen_prototypes or use_old_overlap_conflict)

        old_prototypes = None
        old_class_ids = None
        if need_all_seen_competence and self._known_classes > 0:
            old_class_ids = torch.arange(self._known_classes,dtype=torch.long,)
            old_prototypes = torch.stack([self._w0_class_means[int(class_id)] for class_id in old_class_ids])



        # 使用当前任务训练样本构建类别原型，并做确定性 holdout
        w0_competence_new, prototypes, class_ids = split_prototype_competence(
            features, # 80% → 建立类别原型
            targets, # 20% → 测试 W0 NCM 准确率
            indices,
            holdout_mod=int(self.args.get("dual_mask_competence_holdout_mod", 5)),
            metric=competence_metric,
        )


        plasticity_adaptive = bool(self.args.get("dual_mask_plasticity_adaptive", False))
        self._w0_ncm_loss_new = None
        self._w0_plasticity_demand = None
        if plasticity_adaptive:
            (self._w0_ncm_loss_new,self._w0_plasticity_demand,) = split_prototype_ncm_diagnostics(
                features,
                targets,
                indices,
                holdout_mod=int(self.args.get("dual_mask_competence_holdout_mod", 5)),
                scale=self.scale,
            )

        w0_competence_all_seen = None
        old_overlap_risk = None
        if need_all_seen_competence:
            w0_competence_all_seen, _, _ = split_prototype_competence(
                features,
                targets,
                indices,
                holdout_mod=int(self.args.get("dual_mask_competence_holdout_mod", 5)),
                old_prototypes=old_prototypes,
                old_class_ids=old_class_ids,
                metric=competence_metric,
            )
            old_overlap_risk = max(0.0, w0_competence_new - w0_competence_all_seen,)
        w0_competence = (
            w0_competence_new
            if use_old_overlap_conflict
            else (w0_competence_all_seen if use_all_seen_prototypes else w0_competence_new))


        self._w0_competence = w0_competence

        self._w0_competence_new = w0_competence_new
        self._w0_competence_all_seen = w0_competence_all_seen
        self._w0_old_overlap_risk = old_overlap_risk

        for prototype, class_id in zip(prototypes, class_ids):
            self._w0_class_means[int(class_id.item())] = prototype.cpu()
        for module in self._iter_lora_modules():
            # 同一个任务中，12 层 Attention 使用的是完全相同的 task-level competence
            module.set_pretrained_competence(w0_competence,self._w0_plasticity_demand or 0.0,)
            module.set_pretrained_old_overlap_risk(old_overlap_risk or 0.0)

        first_module = next(self._iter_lora_modules())
        self._w0_control_competence = (first_module.pretrained_control_competence)

        logging.info(
            # "Task %s W_pre train-only competence: %.2f%%, "
            # "Task %s W_pre train-only competence: %.2f%%, candidate_scope=%s, "
            "Task %s W_pre train-only competence: %.2f%%, metric=%s, candidate_scope=%s, "
            "C_new=%.2f%%, C_all=%s, R_old=%s, "
            "C_control=%.2f%%, D_t=%s, "
            "importance_coverage=%.3f, "
            "protect_strength=%.3f, private_rank=%s",
            self._cur_task,
            w0_competence * 100.0,
            competence_metric,
            # "all_seen" if use_all_seen_prototypes else "new_only",
            "all_seen" if use_all_seen_prototypes and not use_old_overlap_conflict else "new_only",
            w0_competence_new * 100.0,
            "n/a" if w0_competence_all_seen is None else "{:.2f}%".format(w0_competence_all_seen * 100.0),
            "n/a" if old_overlap_risk is None else "{:.2f}%".format(old_overlap_risk * 100.0),
            self._w0_control_competence * 100.0,
            "n/a" if self._w0_plasticity_demand is None else "{:.4f}".format(self._w0_plasticity_demand),
            first_module.effective_energy_coverage,
            first_module.effective_protect_strength,
            first_module.current_private_rank,
        )

        if plasticity_adaptive:
            logging.info(
                "Task %s W_pre new-only NCM diagnostics: "
                # "L_NCM=%.6f, D_t=%.4f, scale=%.3f (logging only).",
                "L_NCM=%.6f, D_t=%.4f, scale=%.3f (controls competence).",
                self._cur_task,
                self._w0_ncm_loss_new,
                self._w0_plasticity_demand,
                self.scale
            )



    def eval_w0_task(self):
        if not self._w0_class_means:
            return None

        class_ids = torch.tensor(
            sorted(self._w0_class_means),
            device=self._device,
            dtype=torch.long,
        )
        prototypes = torch.stack([self._w0_class_means[int(class_id)] for class_id in class_ids.cpu()]).to(self._device)  # [C*t,768]
        correct, total = 0, 0
        was_training = self._network.training
        self._network.eval()
        with self._pretrained_anchor_context(), torch.no_grad():
            for _, inputs, targets in self.test_loader:
                features = self._network.extract_vector(inputs.to(self._device))
                logits = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).T
                predictions = class_ids[logits.argmax(dim=1)]
                targets = targets.to(self._device)
                correct += int((predictions == targets).sum().item())
                total += targets.numel()
        if was_training:
            self._network.train()
        accuracy = 100.0 * correct / max(total, 1)
        self._w0_accuracy_curve.append(accuracy)
        return accuracy
    def _measure_pretrained_drift(self, loader):
        """Log feature cosine drift and relative QKV weight drift from W_pre."""
        max_batches = max(1, int(self.args.get("dual_mask_metric_batches", 4)))
        batches = []
        for batch_id, (_, inputs, _) in enumerate(loader):
            if batch_id >= max_batches:
                break
            batches.append(inputs)

        was_training = self._network.training
        self._network.eval()
        current_features = []
        with torch.no_grad():
            for inputs in batches:
                current_features.append(self._network.extract_vector(inputs.to(self._device)).detach())

        anchor_features = []
        with self._pretrained_anchor_context(), torch.no_grad():
            for inputs in batches:
                anchor_features.append(self._network.extract_vector(inputs.to(self._device)).detach())
        if was_training:
            self._network.train()

        current_features = torch.cat(current_features)
        anchor_features = torch.cat(anchor_features)
        feature_drift = float((1.0 - F.cosine_similarity(current_features, anchor_features, dim=1)).mean().item())
        weight_drifts = [module.relative_weight_drift() for module in self._iter_lora_modules()]
        mean_weight_drift = float(np.mean(weight_drifts))
        max_weight_drift = float(np.max(weight_drifts))
        self._feature_drift_curve.append(feature_drift)
        self._weight_drift_curve.append(mean_weight_drift)
        logging.info(
            "Task %s W_pre drift: feature_cosine=%.6f, weight_relative_mean=%.6f, "
            "weight_relative_max=%.6f",
            self._cur_task,
            feature_drift,
            mean_weight_drift,
            max_weight_drift,
        )

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):

        self._cur_task += 1


        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(data_manager.get_task_size(self._cur_task))  # 当前这个 Task 新增的类别数量
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=True)  # 随机增强视图：用于优化 LoRA
        # 拿到所有已见类的 test set
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers, pin_memory=True)

        track_w0 = bool(self.args.get("dual_mask_track_w0_metrics", False))
        # 开启参数自适应 --- 也就是使用训练集测试W0原型的能力
        competence_adaptive = bool(self.args.get("dual_mask_competence_adaptive", False))

        plasticity_adaptive = bool(self.args.get("dual_mask_plasticity_adaptive", False))

        all_seen_competence = bool(self.args.get("dual_mask_competence_all_seen", False))
        old_overlap_conflict = bool(self.args.get("dual_mask_conflict_old_overlap_adaptive", False))

        functional_merge_calibration = bool(self.args.get("dual_mask_functional_merge_calibration", False))

        selective_anchor_enabled = bool(self.args.get("dual_mask_selective_anchor_enabled", False))

        if (track_w0 or competence_adaptive or plasticity_adaptive or all_seen_competence
                or old_overlap_conflict or functional_merge_calibration or selective_anchor_enabled
        ):
            w0_dataset = data_manager.get_dataset(  # 所有训练样本，顺序固定  | 确定性测试视图：用于判断冻结 W0 的原始能力
                np.arange(self._known_classes, self._total_classes),
                source='train',  # 用训练集样本，但模拟最终测试时的输入方式，评估 W0 的原始能力
                mode='test',  # 同一批训练图，但用测试预处理 | e.g. 固定 resize / center crop
            )  # 946
            self.w0_loader = DataLoader(
                w0_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            self._network.to(self._device)
            self._prepare_w0_prototypes(self.w0_loader)

        self._train(self.train_loader, self.test_loader)

        if track_w0:
            self._measure_pretrained_drift(self.w0_loader)

        # update mean and cov and classifier alignment
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task > 0 and self.args['ca'] is True:
            self._stage2_compact_classifier( # CA 分类器对齐
                self.task_sizes[-1],ca_epochs=int(self.args.get("ca_epochs", 5)),)

    def _train(self, train_loader, test_loader):
        try:
            current_task = self._network.module.numtask - 1  # 多卡
        except AttributeError:
            current_task = self._network.numtask - 1
        current_classifier = "classifier_pool" + "." + str(current_task) + "."

        self._network.to(self._device)
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)  # 1. 先把主干 (ViT Backbone) 所有参数全部冻结
            if name.startswith(current_classifier):
                param.requires_grad_(True)  # 将 分类头打开可训

        for module in self._iter_lora_modules():
            module.before_task(task=self._cur_task)

        if len(self._multiple_gpus) > 1:
            self._network = torch.nn.DataParallel(self._network, self._multiple_gpus)

        kk = 0  # Transformer 层号计数器（0 到 11 层）
        for module in self._iter_lora_modules():
            print(f'********** LoRA weights initialization for layer {kk} **********')
            module._init_lora_weight(task=self._cur_task, layer_idx=kk)  # 初始化 LoRA 的 A B 矩阵权重
            module.set_task_and_stage(task=self._cur_task, layer_idx=kk)  # 设置lora可不可训练
            kk += 1


        ############################## set learning rates ##################################
        flora_params, other_params = [], []  # flora_params:收集的是名称带 lora 的参数（即各个 Transformer 层中 LoRA 的 B 矩阵）
        for name, p in self._network.named_parameters():
            if p.requires_grad:
                if 'lora' in name.lower():
                    flora_params.append(p)
                else:
                    other_params.append(p) # 分类头
        print(f"[Param Group] LoRA params: {len(flora_params)}, Other params: {len(other_params)}")

        enabled = {name for name, p in self._network.named_parameters() if p.requires_grad}
        print(f"[LoRA-Stage] Parameters to be updated: {enabled}")

        lr = self.init_lr if self._cur_task == 0 else self.lrate
        weight_decay = self.init_weight_decay if self._cur_task == 0 else self.weight_decay
        param_groups = [
            {'params': flora_params, 'lr': lr, 'momentum': 0.9, 'weight_decay': weight_decay},
            {'params': other_params, 'lr': lr, 'momentum': 0.9, 'weight_decay': weight_decay}
        ]
        ############################## set learning rates ##################################

        if self._cur_task == 0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(params=param_groups)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(params=param_groups, weight_decay=self.init_weight_decay, betas=(0.9, 0.999))
                scheduler = CosineSchedule(optimizer=optimizer, K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(params=param_groups)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(params=param_groups, weight_decay=self.weight_decay, betas=(0.9, 0.999))
                scheduler = CosineSchedule(optimizer=optimizer, K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        lora_modules = list(self._iter_lora_modules())
        if bool(self.args.get("dual_mask_functional_merge_calibration", False)):
            calibration_loader = getattr(self, "w0_loader", train_loader)
            self._calibrate_functional_merge(calibration_loader)
            
        with torch.no_grad():
            # Task t 的 LoRA刚训练完，但增量还没有融合进主干网络 W0
            print('*' * 10 + 'Extrace features for merging shared component!' + '*' * 10)
            for module in lora_modules:
                module.after_task(task=self._cur_task)

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info("Parameters to be updated (%d):\n  %s", len(enabled), "\n  ".join(sorted(enabled)), )
        prog_bar = tqdm(range(self.run_epoch))
        # 角度惩罚损失
        label_smoothing = float(self.args.get('label_smoothing', 0.0))
        if (bool(self.args.get('label_smoothing_task0_only', False)) and self._cur_task != 0):
            label_smoothing = 0.0
        loss_cos:AngularPenaltySMLoss = AngularPenaltySMLoss(
            loss_type='cosface',
            s=self.scale,
            m=self.margin,
            label_smoothing=label_smoothing,
        )

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.
            correct, total = 0, 0
            training_metric_totals = {}
            training_metric_batches = 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                batch_context = self._extra_training_context(
                    inputs,
                    targets,
                    epoch,
                )

                output = self._network(inputs)
                logits = output['logits']
                task_loss = loss_cos(logits, targets)

                extra_loss = self._extra_training_loss(
                    output=output,
                    inputs=inputs,
                    targets=targets,
                    epoch=epoch,
                    batch_context=batch_context,
                )

                batch_training_metrics = getattr(self,"_last_training_loss_metrics",{},)
                if batch_training_metrics:
                    for name, value in batch_training_metrics.items():
                        value = value.detach()
                        training_metric_totals[name] = (training_metric_totals.get(name, 0.0) + value)
                    training_metric_batches += 1

                loss = self._backward_and_step(
                    task_loss,
                    extra_loss,
                    optimizer,
                    output,
                    targets,
                )

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            self._last_epoch_training_loss_metrics = {}
            if training_metric_batches > 0:
                self._last_epoch_training_loss_metrics = {
                    name: float((value / training_metric_batches).item())
                    for name, value in training_metric_totals.items()
                }
            metric_info = "".join(
                ", {} {:.3e}".format(name, value)
                for name, value in self._last_epoch_training_loss_metrics.items()
            )

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task,
                epoch + 1,
                self.run_epoch,
                losses / len(train_loader),
                train_acc
            ) + metric_info
            prog_bar.set_description(info)

        # test train finished  当前任务 LoRA 训练完成→ LoRA 尚未 merge→ CA 分类器校准尚未执行
        test_acc = self._compute_accuracy(self._network, test_loader)
        # pre-merge / pre-CA Test_accy
        final_info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
            self._cur_task,
            epoch + 1,
            self.run_epoch,
            losses / len(train_loader),
            train_acc,
            test_acc,
        ) + metric_info
        logging.info(final_info)

    def accuracy(self, y_pred, y_true, accuracy_matrix=False):
        assert len(y_pred) == len(y_true), 'Data length error.'

        all_acc = {}
        all_acc['total'] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

        i = 0
        # Grouped accuracy
        for class_id in range(0, np.max(y_true), self.class_num):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + self.class_num))[0]
            label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id + self.class_num - 1).rjust(2, '0'))
            all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
            if accuracy_matrix:
                self.acc_matrix[i, self._cur_task] = all_acc[label]
            i += 1

        # Old accuracy
        idxes = np.where(y_true < self._known_classes)[0]
        all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes),
                                                             decimals=2)

        # New accuracy
        idxes = np.where(y_true >= self._known_classes)[0]
        all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

        return all_acc

    def _evaluate(self, y_pred, y_true, accuracy_matrix=False):
        ret = {}
        # print(len(y_pred), len(y_true))
        # {'00-19': 93.93, '20-39': 97.1, '40-59': 95.41, 'new': 95.41, 'old': 95.37, 'total': 95.39}
        grouped = self.accuracy(y_pred, y_true, accuracy_matrix=accuracy_matrix)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                task_id = (targets // self.class_num).cpu()
                y_true_task.append(task_id)
                # 前向推理，不给真实 task id  | 全局 logits
                outputs = self._network.interface(inputs)  # [bs,C*num_task]
            # topk1 [bs]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts // self.class_num).cpu())
            # CNN top1 with task
            outputs_with_task = torch.zeros_like(outputs)[:, :self.class_num]  # 创建一个只装 20 类 logits 的矩阵
            for idx, i in enumerate(targets // self.class_num):  # 用真实标签算真实 task id
                en, be = self.class_num * i, self.class_num * (i + 1)  # task1: en=20, be=40
                outputs_with_task[idx] = outputs[idx, en:be]  # idx -- 真实task标签 [bs,C]
            predicts_with_task = outputs_with_task.argmax(dim=1)  # [bs]
            predicts_with_task = predicts_with_task + (
                        targets // self.class_num) * self.class_num  # 再加回 task 偏移量，变回全局类别编号

            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        # 转成 []
        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(
            y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.interface(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        """Align classifier heads using Gaussian pseudo-features."""
        ca_epochs = int(self.args.get("ca_epochs", 5))
        for p in self._network.classifier_pool[:self._cur_task + 1].parameters():
            p.requires_grad = True

        run_epochs = ca_epochs
        crct_num = self._total_classes
        param_list = [p for p in self._network.classifier_pool.parameters() if p.requires_grad]
        classifier_lr = self.args["ca_lrate"]
        network_params = [{'params': param_list, 'lr': classifier_lr,
                           'weight_decay': 0.0005}]
        optimizer = optim.SGD(network_params, lr=classifier_lr, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        # loss_cos:AngularPenaltySMLoss = AngularPenaltySMLoss(loss_type='cosface',s=1.0, m=self.margin)

        self._network.to(self._device)

        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256

            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                cls_mean = self._class_means[c_id].to(self._device) * (0.9 + decay)
                cls_cov = self._class_covs[c_id].to(self._device)

                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id] * num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                # -stage two only use classifiers
                outputs = self._network(inp, fc_only=True)
                logits = outputs

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task + 1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)

                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits[:, :crct_num] * self.args["scale"], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            info = (
                'CA Task {} => Loss {:.3f} '
                '(classifier alignment; final accuracy is logged after CA)'
            ).format(self._cur_task, losses / self._total_classes)
            logging.info(info)

    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self,'_class_means') and self._class_means is not None and not check_diff:  # 已经完成过 Task 0，模型中已经存在之前算好的 _class_means（旧类别的均值矩阵）
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = torch.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:  # 首次创建分支 —— 适用于 Task 0
            self._class_means = torch.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            class_mean = torch.mean(torch.tensor(vectors), dim=0)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean.detach()
            self._class_covs[class_idx, ...] = class_cov.detach()

    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = Y2 - Y1
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
        W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement

    def extract_features(self, trainloader, model, task_id=None):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data, task_id)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    ###################################################################################################
    def setup_RP(self):
        self.initiated_G = False
        self._network.use_RP = True
        if self.args['M'] > 0:
            # RP with M > 0
            M = self.args['M']
            self._network.weight = torch.nn.Parameter(
                torch.Tensor(self._total_classes, M).to(device=self._device))  # num classes in task x M
            self._network.W_rand = torch.randn(self._network.dim, M).to(device=self._device)
            self.W_rand = copy.deepcopy(
                self._network.W_rand)  # make a copy that gets passed each time the head is replaced
        else:
            # no RP, only decorrelation
            M = self._network.dim  # this M is L in the paper
        self.Q = torch.zeros(M, self.total_classnum)
        self.G = torch.zeros(M, M)

    def replace_fc(self, trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            # these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.use_RP = True
            if self.args['M'] > 0:
                self._network.W_rand = self.W_rand
            else:
                self._network.W_rand = None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = self._network(data)["features"]
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        def target2onehot(targets, n_classes):
            onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
            onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
            return onehot

        Y = target2onehot(label_list, self.total_classnum)
        if self.args['M'] > 0:
            Features_h = torch.nn.functional.relu(Features_f @ self._network.W_rand.cpu())
        else:
            Features_h = Features_f
        self.Q = self.Q + Features_h.T @ Y
        self.G = self.G + Features_h.T @ Features_h
        ridge = self.optimise_ridge_parameter(Features_h, Y)
        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)),
                                self.Q).T  # better nmerical stability than .inv
        self._network.weight.data = Wo[0:self._total_classes, :].to(device=self._device)

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(3, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val).T  # better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: " + str(ridge))
        return ridge
    ###################################################################################################
