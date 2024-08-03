import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from general import ManagerDataYaml
from train import data_yaml
from utils.augmentations import transform_input

class CustomDataLoader():
    def __init__(self, mode,  batch_size, num_workers):

        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
     
     
    def create_dataloader(self):

        if self.mode == 'train':
            data = CustomDataset('train', data_yaml, transform_input(224,is_train= True))
        elif self.mode == 'valid':
            data = CustomDataset('valid', data_yaml, transform_input(224,is_train= False))
        else:
            data = CustomDataset('test', data_yaml, transform_input(224,is_train= False))
        dataloader = DataLoader(
            dataset= data,
            batch_size= self.batch_size,
            num_workers= self.num_workers,
            shuffle= True if self.mode == 'train' else False,
            drop_last= False)
        
        return dataloader
     