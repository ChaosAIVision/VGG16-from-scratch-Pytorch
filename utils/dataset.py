import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.general import ManagerDataYaml

class CustomDataset(Dataset):
    def __init__(self, is_train, data_yaml, transform=None):
       
        data_yaml_manage = ManagerDataYaml(data_yaml)
        data_yaml_manage.load_yaml()
        self.categories = data_yaml_manage.get_properties(key='categories')
        if is_train == 'train':
            data_path = data_yaml_manage.get_properties(key='train')
        elif is_train == 'valid':
            data_path = data_yaml_manage.get_properties(key='valid')
        else:
            data_path = data_yaml_manage.get_properties(key='test')
        self.image_paths = []
        self.labels = []
        for index, category in enumerate(self.categories):
            subdir_path = os.path.join(data_path, category)
            for file_name in os.listdir(subdir_path):
                self.image_paths.append(os.path.join(subdir_path, file_name))
                self.labels.append(index)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label
if __name__ =="__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)


