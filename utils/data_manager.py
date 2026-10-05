import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR100, iIMAGENET_R, iIMAGENET_A, iCUB, iCIFAR10


class DataManager:
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        remaining = len(self._class_order) - sum(self._increments)
        if remaining > 0:
            self._increments.append(remaining)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode, ret_data=False):
        x, y = (self._train_data, self._train_targets) if source == "train" else (
            self._test_data, self._test_targets)
        trsf = self._train_trsf if mode == "train" else self._test_trsf
        trsf = transforms.Compose([*trsf, *self._common_trsf])
        data, targets = [], []
        for index in indices:
            class_data, class_targets = self._select(x, y, index, index + 1)
            data.append(class_data)
            targets.append(class_targets)
        data, targets = np.concatenate(data), np.concatenate(targets)
        dataset = DummyDataset(data, targets, trsf, self.use_path)
        return (data, targets, dataset) if ret_data else dataset

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == 'cifar100':
        return iCIFAR100(args)
    elif name == 'cifar10':
        return iCIFAR10(args)
    elif name == 'imagenet_r':
        return iIMAGENET_R(args)
    elif name == 'imagenet_a':
        return iIMAGENET_A(args)
    elif name in ['cars', 'domainnet']:
        return iIMAGENET_A(args)
    elif name == 'cub':
        return iCUB(args)
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
