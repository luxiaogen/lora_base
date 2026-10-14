"""单编码器的特征提取与评估；不包含回放或其它方法入口。"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.distance import cdist
from utils.toolkit import tensor2numpy


class BaseLearner:
    def __init__(self, args):
        self._cur_task, self._known_classes, self._total_classes = -1, 0, 0
        self._device = args['device'][0]
        self._multiple_gpus = args['device']

    @property
    def feature_dim(self):
        return self._network.feature_dim

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for _, inputs, labels in loader:
                vectors.append(tensor2numpy(self._network.extract_vector(inputs.to(self._device))))
                targets.append(labels.numpy())
        return np.concatenate(vectors), np.concatenate(targets)

    def _eval_nme(self, loader, class_means):
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-8)).T
        means = [mean.cpu().numpy() for mean in F.normalize(class_means, p=2, dim=-1)]
        scores = cdist(means, vectors, 'sqeuclidean').T
        return np.argsort(scores, axis=1)[:, :1], y_true

    def eval_task(self):
        pred, pred_task, targets, task_pred, task_true = self._eval_cnn(self.test_loader)
        cnn = self._evaluate(pred, targets, accuracy_matrix=True)
        with_task = self._evaluate(pred_task, targets)
        task_accuracy = (task_pred == task_true).sum().item() / len(task_pred)
        # 该特征遍历属于旧配方的评估序列，保留以对齐下一任务RNG。
        nme_pred, nme_true = self._eval_nme(self.test_loader, self._class_means)
        # 修正旧诊断的[N,1]×[N]广播；不改变遍历、RNG或正式CNN结果。
        return cnn, with_task, self._evaluate(nme_pred.reshape(-1), nme_true), task_accuracy
