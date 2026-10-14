"""固定分层B训练：Task0 anchor；增量权限/幅度门；原CA。"""
from contextlib import ExitStack
import json
import logging
import time
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

from methods.base import BaseLearner
from models.network import MANet
from models.attention import Attention_LoRA
from models.losses import AngularPenaltySMLoss
from utils.toolkit import tensor2numpy
from utils.dual_mask_metrics import split_prototype_competence
from utils.depth_permissions import install_depth_permissions
from utils.ca_drift_augmentation import collect_current_features, augment_old_covariances


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = MANet(args)
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module._init_params(args)
        self.init_epoch, self.epochs = args['init_epoch'], args['epochs']
        self.init_lr, self.lrate = args['init_lr'], args['lrate']
        self.init_weight_decay, self.weight_decay = args['init_weight_decay'], args['weight_decay']
        self.batch_size, self.num_workers = args['batch_size'], args['num_workers']
        self.scale, self.margin = args['scale'], args['margin']
        self.total_sessions, self.class_num = args['total_sessions'], self._network.class_num
        self.logit_norm = args['logit_norm']
        self.topk = 1
        self.task_sizes = []
        self._class_means = self._class_covs = None
        self.acc_matrix = np.zeros((self.total_sessions, self.total_sessions))
        self._w0_class_means = {}
        self._w0_accuracy_curve, self._feature_drift_curve, self._weight_drift_curve = [], [], []
        self._depth_coverage_center = None
        for idx, module in enumerate(self._iter_lora_modules()):
            module.layer_idx = idx

    def _iter_lora_modules(self):
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                yield module

    def _pretrained_anchor_context(self):
        stack = ExitStack()
        for module in self._iter_lora_modules():
            stack.enter_context(module.use_pretrained_anchor())
        return stack
    # 用固定的预训练模型 W_pre，提取 loader 中每张图片的特征，并返回样本编号、特征和标签
    def _collect_anchor_features(self, loader):
        was_training = self._network.training
        self._network.eval() # 关闭 Dropout 等训练期行为，让特征提取更稳定
        indices, features, targets = [], [], []
        with self._pretrained_anchor_context(), torch.no_grad(): # 结合前面的实现，临时使用原始 W_pre，并跳过 LoRA 增量；退出后恢复当前模型权重 && 只做特征提取，不计算梯度、不更新参数
            for ids, inputs, labels in loader: # 样本编号,图片,类别标签
                vectors = self._network.extract_vector(inputs.to(self._device))
                indices.append(ids.detach().cpu()) # 用于数据的划分
                features.append(vectors.detach().cpu())
                targets.append(labels.detach().cpu())
        if was_training:
            self._network.train()
        return torch.cat(indices), torch.cat(features), torch.cat(targets)

    def _prepare_w0_prototypes(self, loader):
        ids, features, targets = self._collect_anchor_features(loader)
        old_ids = old_prototypes = None
        if self._known_classes > 0:
            old_ids = torch.arange(self._known_classes, dtype=torch.long)
            old_prototypes = torch.stack([self._w0_class_means[int(c)] for c in old_ids])
        c_new, prototypes, class_ids = split_prototype_competence(features, targets, ids, holdout_mod=5)

        c_all, _, _ = split_prototype_competence(features, targets, ids, holdout_mod=5,
                                               old_prototypes=old_prototypes, old_class_ids=old_ids)
        risk = max(0., c_new - c_all) # R_old
        self._w0_competence_new, self._w0_competence_all_seen, self._w0_old_overlap_risk = c_new, c_all, risk

        for prototype, c in zip(prototypes, class_ids):
            self._w0_class_means[int(c.item())] = prototype.cpu()

        for module in self._iter_lora_modules(): # 为每一层设置R_old
            module.set_pretrained_old_overlap_risk(risk) # R_old

    def eval_w0_task(self):
        class_ids = torch.tensor(sorted(self._w0_class_means), device=self._device, dtype=torch.long)
        prototypes = torch.stack([self._w0_class_means[int(c)] for c in class_ids.cpu()]).to(self._device)
        correct, total = 0, 0
        was_training = self._network.training
        self._network.eval()
        with self._pretrained_anchor_context(), torch.no_grad():
            for _, inputs, targets in self.test_loader:
                features = self._network.extract_vector(inputs.to(self._device))
                logits = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).T
                pred = class_ids[logits.argmax(dim=1)]
                targets = targets.to(self._device)
                correct += int((pred == targets).sum().item())
                total += targets.numel()
        if was_training:
            self._network.train()
        accuracy = 100. * correct / max(total, 1)
        self._w0_accuracy_curve.append(accuracy)
        return accuracy

    def _measure_pretrained_drift(self, loader):
        # 保留旧配方的数据迭代：删除这一只读遍历会改变下一任务的RNG。
        batches = []
        for batch_id, (_, inputs, _) in enumerate(loader):
            if batch_id >= 4:
                break
            batches.append(inputs)
        was_training = self._network.training
        self._network.eval()
        current, anchor = [], []
        with torch.no_grad():
            for inputs in batches:
                current.append(self._network.extract_vector(inputs.to(self._device)).detach())
        with self._pretrained_anchor_context(), torch.no_grad():
            for inputs in batches:
                anchor.append(self._network.extract_vector(inputs.to(self._device)).detach())
        if was_training:
            self._network.train()
        drift = float((1. - F.cosine_similarity(torch.cat(current), torch.cat(anchor), dim=1)).mean().item())
        self._feature_drift_curve.append(drift)
        self._weight_drift_curve.append(float(np.mean([m.relative_weight_drift() for m in self._iter_lora_modules()])))

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(data_manager.get_task_size(self._cur_task))
        self._network.advance_task()
        classes = np.arange(self._known_classes, self._total_classes)
        train_data = data_manager.get_dataset(classes, source='train', mode='train')
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers, pin_memory=True)
        test_data = data_manager.get_dataset(np.arange(self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers, pin_memory=True)
        # anchor数据集仅用于评估预训练特征漂移和计算原始类均值 不参与训练
        anchor_data = data_manager.get_dataset(classes, source='train', mode='test')
        self.w0_loader = DataLoader(anchor_data, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers, pin_memory=True)
        self._network.to(self._device)

        self._prepare_w0_prototypes(self.w0_loader)

        self._ca_drift_audit = {}
        drift_mode = self.args.get('ca_drift_augmentation', 'off')
        drift_before = None
        probe_seconds = 0.
        if self._cur_task > 0 and drift_mode != 'off':
            started = time.perf_counter()
            drift_before = collect_current_features(self._network, self.w0_loader, self._device)
            probe_seconds = time.perf_counter() - started
        self._train(self.train_loader, self.test_loader)

        if drift_before is not None:
            started = time.perf_counter()
            drift_after = collect_current_features(self._network, self.w0_loader, self._device)
            self._class_covs, self._ca_drift_audit = augment_old_covariances(
                self._class_covs, drift_before, drift_after, drift_mode, self.args['seed'])
            self._ca_drift_audit['probe_seconds'] = probe_seconds + time.perf_counter() - started
            logging.info('CADriftAudit %s', json.dumps(self._ca_drift_audit, sort_keys=True))
            del drift_before, drift_after
        self._measure_pretrained_drift(self.w0_loader)
        self._compute_class_mean(data_manager)
        if self._cur_task > 0:
            self._stage2_compact_classifier(self.task_sizes[-1])

    def _train(self, train_loader, test_loader):
        current_classifier = 'classifier_pool.' + str(self._network.numtask - 1) + '.'
        self._network.to(self._device)
        for name, p in self._network.named_parameters():
            p.requires_grad_(name.startswith(current_classifier))
        for module in self._iter_lora_modules():
            module.before_task(self._cur_task)

        self._depth_coverage_center = install_depth_permissions(self._iter_lora_modules(), self._cur_task,
                                                               self.args, self._depth_coverage_center)
        for idx, module in enumerate(self._iter_lora_modules()):
            module.set_task_and_stage(self._cur_task, idx)
        factors, classifier = [], []
        for name, p in self._network.named_parameters():
            if p.requires_grad:
                (factors if 'lora' in name.lower() else classifier).append(p)
        lr = self.init_lr if self._cur_task == 0 else self.lrate
        decay = self.init_weight_decay if self._cur_task == 0 else self.weight_decay
        groups = [{'params': factors, 'lr': lr, 'momentum': .9, 'weight_decay': decay},
                  {'params': classifier, 'lr': lr, 'momentum': .9, 'weight_decay': decay}]
        optimizer = optim.SGD(params=groups)
        self.run_epoch = self.init_epoch if self._cur_task == 0 else self.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.run_epoch)
        self.train_function(train_loader, test_loader, optimizer, scheduler)
        with torch.no_grad():
            for module in self._iter_lora_modules():
                module.after_task(self._cur_task)

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        loss_cos = AngularPenaltySMLoss(s=self.scale, m=self.margin)
        for epoch in range(self.run_epoch):
            self._network.train()
            losses, correct, total = 0., 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                targets = targets - self._known_classes
                output = self._network(inputs)
                logits = output['logits']
                loss = loss_cos(logits, targets)
                if self._cur_task == 0:
                    anchor = torch.stack([m.anchor_regularization() for m in self._iter_lora_modules()]).mean()
                    # 保持旧extra_loss的stack/sum运算结构。
                    loss = loss + torch.stack([2.5 * anchor]).sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            logging.info('Task %s, Epoch %s/%s => Loss %.3f, Train_accy %.2f',
                         self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
        # 不输出额外一行，但保留原训练末期评估遍历，避免改变CA/RNG。
        self._compute_accuracy(self._network, test_loader)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            with torch.no_grad():
                outputs = model.interface(inputs.to(self._device))
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def accuracy(self, y_pred, y_true, accuracy_matrix=False):
        all_acc = {'total': np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)}
        for i, class_id in enumerate(range(0, np.max(y_true), self.class_num)):
            indices = np.where(np.logical_and(y_true >= class_id, y_true < class_id + self.class_num))[0]
            label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id + self.class_num - 1).rjust(2, '0'))
            all_acc[label] = np.around((y_pred[indices] == y_true[indices]).sum() * 100 / len(indices), decimals=2)
            if accuracy_matrix:
                self.acc_matrix[i, self._cur_task] = all_acc[label]
        indices = np.where(y_true < self._known_classes)[0]
        all_acc['old'] = 0 if len(indices) == 0 else np.around((y_pred[indices] == y_true[indices]).sum() * 100 / len(indices), decimals=2)
        indices = np.where(y_true >= self._known_classes)[0]
        all_acc['new'] = np.around((y_pred[indices] == y_true[indices]).sum() * 100 / len(indices), decimals=2)
        return all_acc

    def _evaluate(self, y_pred, y_true, accuracy_matrix=False):
        grouped = self.accuracy(y_pred, y_true, accuracy_matrix)
        return {'grouped': grouped, 'top1': grouped['total']}

    def _eval_cnn(self, loader):
        self._network.eval()
        predictions, predictions_with_task, labels, task_pred, task_true = [], [], [], [], []
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                task_true.append((targets // self.class_num).cpu())
                outputs = self._network.interface(inputs)
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)
            task_pred.append((predicts // self.class_num).cpu())
            # 仅报告用Task-ID诊断；正式预测始终使用上面的全已见类outputs。
            local = torch.zeros_like(outputs)[:, :self.class_num]
            for idx, task in enumerate(targets // self.class_num):
                local[idx] = outputs[idx, self.class_num * task:self.class_num * (task + 1)]
            local_pred = local.argmax(dim=1) + (targets // self.class_num) * self.class_num
            predictions.append(predicts.cpu().numpy())
            predictions_with_task.append(local_pred.cpu().numpy())
            labels.append(targets.cpu().numpy())
        return (np.concatenate(predictions), np.concatenate(predictions_with_task), np.concatenate(labels),
                torch.cat(task_pred), torch.cat(task_true))

    def _stage2_compact_classifier(self, task_size):
        epochs = self.args['ca_epochs']
        for p in self._network.classifier_pool[:self._cur_task + 1].parameters():
            p.requires_grad = True
        params = [p for p in self._network.classifier_pool.parameters() if p.requires_grad]
        lr = self.args['ca_lrate']
        optimizer = optim.SGD([{'params': params, 'lr': lr, 'weight_decay': .0005}],
                              lr=lr, momentum=.9, weight_decay=.0005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self._network.to(self._device)
        self._network.eval()
        for epoch in range(epochs):
            losses, sampled_data, sampled_label = 0., [], []
            for c in range(self._total_classes):
                task = c // task_size
                decay = (task + 1) / (self._cur_task + 1) * .1
                mean = self._class_means[c].to(self._device) * (.9 + decay)
                cov = self._class_covs[c].to(self._device)
                sampled_data.append(MultivariateNormal(mean.float(), cov.float()).sample(sample_shape=(256,)))
                sampled_label.extend([c] * 256)
            inputs = torch.cat(sampled_data, dim=0).float().to(self._device)
            targets = torch.tensor(sampled_label).long().to(self._device)
            order = torch.randperm(inputs.size(0))
            inputs, targets = inputs[order], targets[order]
            for step in range(self._total_classes):
                logits = self._network(inputs[step * 256:(step + 1) * 256], fc_only=True)
                tgt = targets[step * 256:(step + 1) * 256]
                norms, previous, current = [], 0, 0
                for task in range(self._cur_task + 1):
                    current += self.task_sizes[task]
                    norms.append(torch.norm(logits[:, previous:current], p=2, dim=-1, keepdim=True) + 1e-7)
                    previous = current
                norms = torch.cat(norms, dim=-1).mean(dim=-1, keepdim=True)
                loss = F.cross_entropy(torch.div(logits[:, :self._total_classes], norms) / self.logit_norm, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            scheduler.step()
            logging.info('CA Task %s, Epoch %s/%s => Loss %.3f', self._cur_task, epoch + 1, epochs,
                         losses / self._total_classes)

    def _compute_class_mean(self, data_manager):
        means = torch.zeros((self._total_classes, self.feature_dim))
        covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
        if self._class_means is not None:
            means[:self._known_classes] = self._class_means
            covs[:self._known_classes] = self._class_covs
        self._class_means, self._class_covs = means, covs
        for c in range(self._known_classes, self._total_classes):
            _, _, dataset = data_manager.get_dataset(np.arange(c, c + 1), source='train', mode='test', ret_data=True)
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(loader)
            mean = torch.mean(torch.tensor(vectors), dim=0)
            cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(mean.shape[-1]) * 1e-3
            self._class_means[c, :] = mean.detach()
            self._class_covs[c, ...] = cov.detach()
