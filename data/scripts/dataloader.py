from torch.utils.data import DataLoader, Subset
from data.scripts.polyp_dataset import PolypDataset
from hyperparameters import Hyperparameters
from scripts.seed import set_generator, worker_init_fn
import torch
import numpy as np


def samples_per_class(dataset):
    positives, negatives = 0, 0
    for _, mask, _ in dataset:
        if torch.sum(mask == 1).item():
            positives += 1
        else:
            negatives += 1
    print(f"positives: {positives}, negatives: {negatives}")


class DataLoading:
    def __init__(self, include_data="both", shuffle=True, pin_memory=True, persistent_workers=True):
        # set_seed()
        self.include_data = include_data
        self.train_ratio = Hyperparameters.TRAIN_RATIO
        self.val_ratio = Hyperparameters.VAL_RATIO
        self.batch_size = Hyperparameters.BATCH_SIZE
        self.shuffle = shuffle  # after we iterate over all batches the data is shuffled
        self.num_workers = Hyperparameters.NUM_WORKERS
        self.worker_init_fn = worker_init_fn
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        #   Load datasets
        self.dataset_full_with_train_transformations = PolypDataset(mode="train", include_data=self.include_data)
        self.dataset_full_with_val_test_transformations = PolypDataset(mode="val_test", include_data=self.include_data)


    def split_datasets(self):
        # set_seed()
        # Shuffle indices
        train_indices = torch.randperm(len(self.dataset_full_with_train_transformations), generator=set_generator())
        # Calculate sizes
        train_size = int(np.floor(self.train_ratio * len(self.dataset_full_with_train_transformations)))
        val_size = int(np.floor(self.val_ratio * len(self.dataset_full_with_val_test_transformations)))
        test_size = len(self.dataset_full_with_val_test_transformations) - val_size
        # Split datasets
        train_dataset = Subset(self.dataset_full_with_train_transformations, train_indices[:train_size])
        val_dataset = Subset(self.dataset_full_with_val_test_transformations, train_indices[train_size:train_size + val_size])
        test_dataset = Subset(self.dataset_full_with_val_test_transformations, train_indices[train_size + val_size:])
        return train_dataset, val_dataset, test_dataset


    def create_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


    def get_loaders(self):
        train_dataset, val_dataset, test_dataset = self.split_datasets()
        # self.samples_per_class(train_dataset)
        # self.samples_per_class(val_dataset)
        # self.samples_per_class(test_dataset)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)
        val_loader   = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        test_loader  = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        return train_loader, val_loader, test_loader

