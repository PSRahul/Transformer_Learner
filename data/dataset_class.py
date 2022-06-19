from unicodedata import normalize
from torchvision import transforms
import numpy as np
from torchvision.datasets import CIFAR100

data_transform = transforms.Compose((
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                         np.array([0.229, 0.224, 0.225]))
))

train_set = CIFAR100(root="datasets", train=True, transform=data_transform,
                     download=True)


test_set = CIFAR100(root="datasets", train=False, transform=data_transform,
                    download=True)
