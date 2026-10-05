"""DualMask learner: current-task training, gated merge, Gaussian classifier alignment."""
import logging
from contextlib import ExitStack

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

from methods.base import BaseLearner
from models.network import MANet
from models.attention import Attention_LoRA
from models.losses import AngularPenaltySMLoss
from utils.toolkit import tensor2numpy
from utils.dual_mask_metrics import split_prototype_competence, split_prototype_ncm_diagnostics
from utils.reproducibility import advance_loader_rng


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = MANet(args)
        for layer_idx, module in enumerate(self._iter_lora_modules()):
            module._init_params(args)
            module.layer_idx = layer_idx
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.scale = args["scale"]
        self.margin = args["margin"]
        self.total_sessions = args["total_sessions"]
        self.logit_norm = args["logit_norm"]
        self.class_num = self._network.class_num
        self.task_sizes = []
        self._class_means = None
        self._class_covs = None
        self._w0_class_means = {}
        self.acc_matrix = np.zeros((self.total_sessions, self.total_sessions))

    def _iter_lora_modules(self):
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                yield module

    def _extra_training_loss(self):
        modules = list(self._iter_lora_modules())
        losses = []
        for module in modules:
            losses.append(module._joint_conflict_regularization(
                module.S_lora[self._cur_task], isolated=False))
            if self._cur_task > 0:
                losses.append(module._joint_conflict_regularization(
                    module.P_lora[self._cur_task], isolated=True))
        weighted = [float(self.args["dual_mask_reg_weight"]) * torch.stack(losses).mean()]
        if self._cur_task == 0:
            anchor = torch.stack([module.anchor_regularization() for module in modules]).mean()
            weighted.append(float(self.args["dual_mask_anchor_reg_weight"]) * anchor)
        return torch.stack(weighted).sum()

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
                vectors = self._network.extract_vector(inputs.to(self._device))
                indices.append(batch_indices.detach().cpu())
                features.append(vectors.detach().cpu())
                targets.append(batch_targets.detach().cpu())
        if was_training:
            self._network.train()
        return torch.cat(indices), torch.cat(features), torch.cat(targets)

    def _collect_anchor_features(self, loader):
        return self._collect_features(loader, use_pretrained_anchor=True)

    def _prepare_w0_prototypes(self, loader):
        indices, features, targets = self._collect_anchor_features(loader)
        holdout_mod = int(self.args["dual_mask_competence_holdout_mod"])
        competence, prototypes, class_ids = split_prototype_competence(
            features, targets, indices, holdout_mod=holdout_mod)
        ncm_loss, demand = split_prototype_ncm_diagnostics(
            features, targets, indices, holdout_mod=holdout_mod, scale=self.scale)
        old_ids, old_prototypes = None, None
        if self._known_classes > 0:
            old_ids = torch.arange(self._known_classes, dtype=torch.long)
            old_prototypes = torch.stack([self._w0_class_means[int(c)] for c in old_ids])
        all_seen, _, _ = split_prototype_competence(
            features, targets, indices, holdout_mod=holdout_mod,
            old_prototypes=old_prototypes, old_class_ids=old_ids)
        overlap = max(0.0, competence - all_seen)
        for prototype, class_id in zip(prototypes, class_ids):
            self._w0_class_means[int(class_id.item())] = prototype.cpu()
        for module in self._iter_lora_modules():
            module.set_pretrained_competence(competence, demand)
            module.set_pretrained_old_overlap_risk(overlap)
        first = next(self._iter_lora_modules())
        logging.info(
            "Task %s W_pre control: C_new=%.4f, C_all=%.4f, D=%.4f, R_old=%.4f, "
            "coverage=%.4f, alpha=%.4f, P_rank=%s",
            self._cur_task, competence, all_seen, demand, overlap,
            first.effective_energy_coverage, first.effective_protect_strength,
            first.current_private_rank)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(data_manager.get_task_size(self._cur_task))
        self._network.update_fc(self._total_classes)
        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)
        classes = np.arange(self._known_classes, self._total_classes)
        train = data_manager.get_dataset(classes, source="train", mode="train")
        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=True)
        test = data_manager.get_dataset(np.arange(self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers, pin_memory=True)
        anchor = data_manager.get_dataset(classes, source="train", mode="test")
        self.w0_loader = DataLoader(anchor, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers, pin_memory=True)
        self._network.to(self._device)
        self._prepare_w0_prototypes(self.w0_loader)
        self._train(self.train_loader, self.test_loader)
        # The removed feature-drift pass created one sequential loader iterator.
        advance_loader_rng()
        self._compute_class_mean(data_manager)
        if self._cur_task > 0:
            self._stage2_compact_classifier(self.task_sizes[-1])

    def _train(self, train_loader, test_loader):
        current_classifier = "classifier_pool." + str(self._cur_task) + "."
        self._network.to(self._device)
        for name, param in self._network.named_parameters():
            param.requires_grad_(name.startswith(current_classifier))
        for module in self._iter_lora_modules():
            module.before_task(self._cur_task)
        if len(self._multiple_gpus) > 1:
            self._network = torch.nn.DataParallel(self._network, self._multiple_gpus)
        for layer_idx, module in enumerate(self._iter_lora_modules()):
            module.set_task_and_stage(self._cur_task, layer_idx)
        lora_params, classifier_params = [], []
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                (lora_params if "lora" in name.lower() else classifier_params).append(param)
        lr = self.init_lr if self._cur_task == 0 else self.lrate
        decay = self.init_weight_decay if self._cur_task == 0 else self.weight_decay
        groups = [
            {"params": lora_params, "lr": lr, "momentum": 0.9, "weight_decay": decay},
            {"params": classifier_params, "lr": lr, "momentum": 0.9, "weight_decay": decay},
        ]
        optimizer = optim.SGD(params=groups)
        self.run_epoch = self.init_epoch if self._cur_task == 0 else self.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.run_epoch)
        self.train_function(train_loader, test_loader, optimizer, scheduler)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        with torch.no_grad():
            for module in self._iter_lora_modules():
                module.after_task(self._cur_task)

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        loss_cos = AngularPenaltySMLoss(s=self.scale, m=self.margin)
        prog_bar = tqdm(range(self.run_epoch))
        for epoch in prog_bar:
            self._network.train()
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                selected = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, selected)
                targets = torch.index_select(targets, 0, selected) - self._known_classes
                logits = self._network(inputs)["logits"]
                task_loss = loss_cos(logits, targets)
                loss = task_loss + self._extra_training_loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, predictions = torch.max(logits, dim=1)
                correct += predictions.eq(targets.expand_as(predictions)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
        test_acc = self._compute_accuracy(self._network, test_loader)
        logging.info("%s, Test_accy %.2f", info, test_acc)

    def accuracy(self, y_pred, y_true, accuracy_matrix=False):

        all_acc = {}
        all_acc['total'] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

        i = 0
        for class_id in range(0, np.max(y_true), self.class_num):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + self.class_num))[0]
            label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id + self.class_num - 1).rjust(2, '0'))
            all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
            if accuracy_matrix:
                self.acc_matrix[i, self._cur_task] = all_acc[label]
            i += 1

        idxes = np.where(y_true < self._known_classes)[0]
        all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes),
                                                             decimals=2)

        idxes = np.where(y_true >= self._known_classes)[0]
        all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

        return all_acc

    def _evaluate(self, y_pred, y_true, accuracy_matrix=False):
        ret = {}
        grouped = self.accuracy(y_pred, y_true, accuracy_matrix=accuracy_matrix)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def eval_task(self):
        prediction, targets = self._eval_cnn(self.test_loader)
        result = self._evaluate(prediction, targets, accuracy_matrix=True)
        # Removed NME and W_pre test reports each created a sequential iterator.
        advance_loader_rng()
        advance_loader_rng()
        return result

    def _eval_cnn(self, loader):
        self._network.eval()
        prediction, targets = [], []
        with torch.no_grad():
            for _, inputs, labels in loader:
                outputs = self._network.interface(inputs.to(self._device))
                top1 = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1].view(-1)
                prediction.append(top1.cpu().numpy())
                targets.append(labels.numpy())
        return np.concatenate(prediction), np.concatenate(targets)

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

    def _stage2_compact_classifier(self, task_size):
        for param in self._network.classifier_pool[:self._cur_task + 1].parameters():
            param.requires_grad = True
        params = [p for p in self._network.classifier_pool.parameters() if p.requires_grad]
        lr = self.args["ca_lrate"]
        optimizer = optim.SGD([{"params": params, "lr": lr, "weight_decay": 0.0005}],
                              lr=lr, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args["ca_epochs"])
        self._network.to(self._device)
        self._network.eval()
        per_class = 256
        for epoch in range(self.args["ca_epochs"]):
            data, labels = [], []
            for class_id in range(self._total_classes):
                task_id = class_id // task_size
                decay = (task_id + 1) / (self._cur_task + 1) * 0.1
                mean = self._class_means[class_id].to(self._device) * (0.9 + decay)
                cov = self._class_covs[class_id].to(self._device)
                distribution = MultivariateNormal(mean.float(), cov.float())
                data.append(distribution.sample(sample_shape=(per_class,)))
                labels.extend([class_id] * per_class)
            inputs = torch.cat(data, dim=0).float().to(self._device)
            targets = torch.tensor(labels).long().to(self._device)
            order = torch.randperm(inputs.size(0))
            inputs, targets = inputs[order], targets[order]
            losses = 0.0
            for batch in range(self._total_classes):
                inp = inputs[batch * per_class:(batch + 1) * per_class]
                tgt = targets[batch * per_class:(batch + 1) * per_class]
                logits = self._network(inp, fc_only=True)
                if self.logit_norm is not None:
                    norms, start, end = [], 0, 0
                    for task in range(self._cur_task + 1):
                        end += self.task_sizes[task]
                        norms.append(torch.norm(logits[:, start:end], p=2, dim=-1, keepdim=True) + 1e-7)
                        start += self.task_sizes[task]
                    norm = torch.cat(norms, dim=-1).mean(dim=-1, keepdim=True)
                    loss_logits = torch.div(logits[:, :self._total_classes], norm) / self.logit_norm
                else:
                    loss_logits = logits[:, :self._total_classes] * self.scale
                loss = F.cross_entropy(loss_logits, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            scheduler.step()
            logging.info("CA Task %s Epoch %s => Loss %.3f", self._cur_task, epoch + 1,
                         losses / self._total_classes)

    def _compute_class_mean(self, data_manager):
        means = torch.zeros((self._total_classes, self.feature_dim))
        covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
        if self._class_means is not None:
            means[:self._known_classes] = self._class_means
            covs[:self._known_classes] = self._class_covs
        self._class_means, self._class_covs = means, covs
        for class_id in range(self._known_classes, self._total_classes):
            _, _, dataset = data_manager.get_dataset(
                np.arange(class_id, class_id + 1), source="train", mode="test", ret_data=True)
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(loader)
            mean = torch.mean(torch.tensor(vectors), dim=0)
            cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)
            cov = cov + torch.eye(mean.shape[-1]) * 1e-3
            self._class_means[class_id] = mean.detach()
            self._class_covs[class_id] = cov.detach()

    def after_task(self):
        self._known_classes = self._total_classes
