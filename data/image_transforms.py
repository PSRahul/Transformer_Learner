import os
import sys
sys.path.append("/home/psrahul/MasterThesis/repo/ViT_Learner/")
from data.cifar_dataset_class import CIFARDataset


def image_to_patches(image, patch_size, flatten_channels=True):

    B, C, H, W = image.shape
    image = image.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    # B, C, NP, PS, NP, PS -> B, NP, NP, C, PS, PS
    image = image.permute(0, 2, 4, 1, 3, 5)
    # B, NP, NP, C, PS, PS -> B, NP * NP, C, PS, PS
    image = image.flatten(1, 2)
    if flatten_channels:
        # B, NP * NP, C, PS, PS -> B, NP * NP, C * PS * PS
        image = image.flatten(2, 4)
    return image


cifar_dataset = CIFARDataset()
train_dataset = cifar_dataset.get_train()
for idx in train_dataset:
    print(idx[0].shape)
    image = idx[0]
    break

image_patch=image_to_patches(image,patch_size=2)