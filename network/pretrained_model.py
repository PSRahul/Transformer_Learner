
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision.models import resnet34
import sys
import os
sys.path.append("/home/psrahul/MasterThesis/repo/ViT_Learner/")
from data.cifar_dataset_class import CIFARDataset


class FeatureExtractor():

    def __init__(self):

        pretrained_model = resnet34(pretrained=True)
        pretrained_model.fc = nn.Sequential()
        pretrained_model = pretrained_model.to("cuda")
        pretrained_model.eval()
        for param in pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model = pretrained_model

    @torch.no_grad()
    def extract_features(self,dataset, save_file_name):

        if not os.path.isfile(save_file_name):
            cifar_loader = DataLoader(
                dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=8)

            feature_list = []
            for img, _ in tqdm(cifar_loader):
                img = img.to("cuda")
                feature = self.pretrained_model(img)
                feature_list.append(feature)
                img = img.to("cpu")
            feature_list=torch.cat(feature_list,dim=0)
            torch.save(feature_list,save_file_name)


        else:
            feature_list=torch.load(save_file_name)
        return feature_list

def main():
    cifar_dataset=CIFARDataset()
    feature_extractor=FeatureExtractor()
    train_features=feature_extractor.extract_features(cifar_dataset.get_train(),
    "/home/psrahul/MasterThesis/repo/ViT_Learner/data/train_set_features.tar")

    test_features=feature_extractor.extract_features(
        cifar_dataset.get_test(), "/home/psrahul/MasterThesis/repo/ViT_Learner/data/test_set_features.tar")

if __name__ == "__main__":
    main()
 
