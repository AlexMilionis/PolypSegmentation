import torch
from torch.utils.data import DataLoader, Subset
from src.data.polyp_dataset import PolypDataset
from src.config.hyperparameters import Hyperparameters
from src.config.seed import set_generator
import numpy as np


class DataLoading:
    def __init__(self, config, shuffle_train_data=True, pin_memory=True, persistent_workers=True):
        # set_seed()
        self.include_data = config['dataset']['include_data']
        self.train_ratio = config['dataset']['train_ratio']
        self.val_ratio = config['dataset']['val_ratio']
        self.batch_size = config['training']['batch_size']
        self.shuffle_train_data = shuffle_train_data  # after we iterate over all batches the data is shuffled
        self.num_workers = config['training']['batch_size']
        # self.worker_init_fn = worker_init_fn
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        #   Load datasets
        self.dataset_full_with_train_transformations = PolypDataset(mode="train", include_data=self.include_data)
        self.dataset_full_with_val_test_transformations = PolypDataset(mode="val_test", include_data=self.include_data)


    def split_datasets(self):
        # Shuffle indices
        train_indices = torch.randperm(len(self.dataset_full_with_train_transformations), generator=set_generator())
        # Calculate sizes
        train_size = int(np.floor(self.train_ratio * len(self.dataset_full_with_train_transformations)))
        val_size = int(np.floor(self.val_ratio * len(self.dataset_full_with_val_test_transformations)))

        # Split datasets
        train_dataset = Subset(self.dataset_full_with_train_transformations, train_indices[:train_size])
        val_dataset = Subset(self.dataset_full_with_val_test_transformations, train_indices[train_size:train_size + val_size])
        test_dataset = Subset(self.dataset_full_with_val_test_transformations, train_indices[train_size + val_size:])
        return train_dataset, val_dataset, test_dataset


    def get_loaders(self):
        train_dataset, val_dataset, test_dataset = self.split_datasets()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train_data, num_workers=self.num_workers, pin_memory=self.pin_memory)
        val_loader   = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        test_loader  = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        return train_loader, val_loader, test_loader

