import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2


class CustomDataset(Dataset):
    def __init__(self, root, is_train, transform=None):
        if is_train == 'train':
            data_path = os.path.join(root, "train")
        elif is_train == 'valid':
            data_path = os.path.join(root, "valid")
        else:
            data_path = os.path.join(root, 'test')
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                           "squirrel"]
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

