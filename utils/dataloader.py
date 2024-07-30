import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset

class CustomDataLoader():
    def __init__(self, mode,dataset_path,  batch_size, num_workers):

        self.mode = mode
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def create_dataloader(self):


        if self.mode == 'train':
            self.dataset_path = 'custom'
        elif self.mode == 'valid':
            self.datset_path = 'custom'
        else:
            self.dataset_path = 'custom'
        dataloader = DataLoader(
            dataset= self.datset_path,
            batch_size= self.batch_size,
            num_workers= self.num_workers,
            shuffle= True if self.mode == 'train' else False,
            drop_last= False)
        
        return dataloader
     