from unicodedata import normalize
from torchvision import transforms
import numpy as np
from torchvision.datasets import CIFAR100


class CIFARDataset():

    def __init__(self):
        self.data_transform = transforms.Compose((
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                 np.array([0.229, 0.224, 0.225]))
        ))

        self.train_set = CIFAR100(root="datasets", train=True, transform=self.data_transform,
                                  download=True)

        self.test_set = CIFAR100(root="datasets", train=False, transform=self.data_transform,
                                 download=True)
                                 
        

    def get_train(self):
        return self.train_set

    def get_test(self):
        return self.test_set
