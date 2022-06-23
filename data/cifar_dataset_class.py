from torchvision import transforms
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader


class CIFARDataset():

    def __init__(self):
        test_transform = transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]),
                                 np.array([0.24703225141799082, 0.24348516474564, 0.26158783926049628]))
        ))

        train_transform = transforms.Compose((
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(
                (32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]),
                                 np.array([0.24703225141799082, 0.24348516474564, 0.26158783926049628]))
        ))

        trainval_set = CIFAR10(root="datasets/CIFAR10", train=True, transform=train_transform,
                               download=True)

        self.test_set = CIFAR10(root="datasets/CIFAR10", train=False, transform=test_transform,
                                download=True)

        self.train_set, self.val_set = random_split(
            trainval_set, [45000, 5000])

    def get_train(self):
        return DataLoader(self.train_set, batch_size=128, shuffle=True, num_workers=8)

    def get_val(self):
        return DataLoader(self.val_set, batch_size=128, shuffle=True, num_workers=8)

    def get_test(self):
        return DataLoader(self.test_set, batch_size=128, shuffle=True, num_workers=8)


