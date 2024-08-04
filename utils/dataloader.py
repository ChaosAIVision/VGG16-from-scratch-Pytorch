import torch
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.general import ManagerDataYaml
from utils.augmentations import transform_input

class CustomDataLoader():
    def __init__(self,data_yaml, mode :str,  batch_size:int, num_workers:int):

        self.mode = mode
        self.data_yaml = data_yaml
        self.batch_size = batch_size
        self.num_workers = num_workers
     
     
    def create_dataloader(self):

        if self.mode == 'train':
            data = CustomDataset('train', self.data_yaml, transform_input(224,is_train= True))
        elif self.mode == 'valid':
            data = CustomDataset('valid', self.data_yaml, transform_input(224,is_train= False))
        else:
            data = CustomDataset('test', self.data_yaml, transform_input(224,is_train= False))
        dataloader = DataLoader(
            dataset= data,
            batch_size= self.batch_size,
            num_workers= self.num_workers,
            shuffle= True if self.mode == 'train' else False,
            drop_last= False)
        
        return dataloader
     