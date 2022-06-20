
from turtle import shape
from wsgiref import validate
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

def get_train_val_set(train=True):
    cifar_dataset=CIFARDataset()
    feature_extractor=FeatureExtractor()
    train_val_features=feature_extractor.extract_features(cifar_dataset.get_train(),
    "/home/psrahul/MasterThesis/repo/ViT_Learner/data/train_set_features.tar")

    test_features=feature_extractor.extract_features(
        cifar_dataset.get_test(), "/home/psrahul/MasterThesis/repo/ViT_Learner/data/test_set_features.tar")

    #print("Train Features",train_features.shape)
    #print("Test Features",test_features.shape)

    labels=cifar_dataset.get_train().targets
    labels = torch.LongTensor(labels)
    num_labels=labels.max()+1

    # Reshpae into [Num of Classes, Image per classes]
    sorted_label_indices=torch.argsort(labels).reshape(num_labels,-1)
    
    num_val_examples=sorted_label_indices.shape[1]//10

    val_indices=sorted_label_indices[:,:num_val_examples].reshape(-1)
    train_indices=sorted_label_indices[:,num_val_examples:].reshape(-1)
    
    train_features,train_labels=train_val_features[train_indices],labels[train_indices]
    val_features,val_labels=train_val_features[val_indices],labels[val_indices]
    
    if(train):
        return train_features, train_labels
    else:
        return val_features,val_labels

    
def main():
    get_train_val_set()

if __name__ == "__main__":
    main()
 
