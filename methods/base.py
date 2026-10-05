import numpy as np
import torch
from torch import nn
from utils.toolkit import tensor2numpy


class BaseLearner:
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self._device)))
                else:
                    _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
