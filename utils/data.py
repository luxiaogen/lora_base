"""只读取已划分的ImageNet-R；不下载、移动或重切数据。"""
import os
import numpy as np
from torchvision import datasets, transforms


def build_transform(is_train):
    if is_train:
        return [transforms.RandomResizedCrop(224, scale=(.05, 1.), ratio=(3. / 4., 4. / 3.)),
                transforms.RandomHorizontalFlip(p=.5), transforms.ToTensor()]
    return [transforms.Resize(256, interpolation=3), transforms.CenterCrop(224), transforms.ToTensor()]


class iIMAGENET_R:
    use_path = True
    train_trsf = build_transform(True)
    test_trsf = build_transform(False)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # 原接口名称保留，但这里只读现有目录；缺失切分由ImageFolder报错。
        train = datasets.ImageFolder(os.path.join(self.args['data_path'], 'train')).samples
        test = datasets.ImageFolder(os.path.join(self.args['data_path'], 'test')).samples
        self.train_data = np.array([row[0] for row in train])
        self.train_targets = np.array([row[1] for row in train])
        self.test_data = np.array([row[0] for row in test])
        self.test_targets = np.array([row[1] for row in test])
