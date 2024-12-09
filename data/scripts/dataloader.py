from torch.utils.data import DataLoader, random_split, Subset
from data.scripts.polyp_dataset import PolypDataset
from hyperparameters import Hyperparameters
from scripts.seed import worker_init_fn, set_seed
import torch
import numpy as np


class DataLoading:
    def __init__(self, include_data="both", shuffle=True, num_workers=8, pin_memory=False):
        self.include_data = include_data
        self.train_ratio = Hyperparameters.TRAIN_RATIO
        self.val_ratio = Hyperparameters.VAL_RATIO
        self.batch_size = Hyperparameters.BATCH_SIZE
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.worker_init_fn = worker_init_fn
        self.train_dataset = PolypDataset(mode="train", include_data=self.include_data)
        self.val_test_dataset = PolypDataset(mode="val_test", include_data=self.include_data)


    def get_loader(self):
        set_seed()
        #   split data
        indices = torch.randperm(len(self.train_dataset))
        train_size = int(np.floor(self.train_ratio * len(self.train_dataset)))
        val_size   = int(np.floor(self.val_ratio * len(self.val_test_dataset)))
        self.train_dataset = Subset(self.train_dataset, indices[:train_size])
        self.val_dataset   = Subset(self.val_test_dataset, indices[train_size:train_size + val_size])
        self.test_dataset  = Subset(self.val_test_dataset, indices[train_size + val_size:])
        #   create separate dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)
        val_loader   = DataLoader(self.val_dataset,   batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)
        test_loader  = DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)
        return train_loader, val_loader, test_loader

