from torch.utils.data import Dataset
import numpy as np
import torch
import sys;sys.path.append("/home/psrahul/MasterThesis/repo/ViT_Learner/")
from network.pretrained_model import get_train_val_set

class AnomalyDataset(Dataset):

    def __init__(self,image_features,labels,set_size=10,train=True) :
        super().__init__()
        self.image_features=image_features
        self.labels=labels
        self.set_size=set_size
        self.train=train
        self.num_class=labels.max()+1
        # Reshape into [Num of Classes, Image per classes]
        self.image_index_with_labels=torch.argsort(self.labels).reshape(self.num_class,-1)
        print("Image Index with Labels",self.image_index_with_labels.shape)
        if not train:
            self.test_data=self.create_test_set()
    

    def create_test_set(self):
        test_data=[]
        num_images= self.image_features.shape[0]
        test_data=[self.sample_image_index(self.labels[idx]) for idx in range(num_images)]
        test_data=torch.stack(test_data,dim=0)
        return test_data

    def sample_image_index(self,anomaly_class):
        correct_class=np.random.randint(self.num_class-1)
        if correct_class>=anomaly_class:
            correct_class+=1

        correct_image_index=np.random.choice(self.image_index_with_labels[1],size=self.set_size-1,replace=False)
        correct_image_index=self.image_index_with_labels[correct_class,correct_image_index]
        return correct_image_index

    def __len__(self):
        return self.image_features.shape[0]

    def __getitem__(self, index):
        anomaly_class=self.image_features[index]

        if self.train:
            image_index=self.sample_image_index(self.labels[index])
        else:
            image_index=self.test_sets[index]

#train_features, train_labels = get_train_val_set (train=True)
test_features, test_labels = get_train_val_set (train=False)

anomaly_dataset=AnomalyDataset(test_features, test_labels ,train=False)