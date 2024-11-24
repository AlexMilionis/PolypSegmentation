import os
import torch
from torch.utils.data import DataLoader
from data.scripts.polyp_dataset import PolypDataset
# from data.scripts.split_data import train_test
from data.constants import Constants
from data.scripts.visualization import *
from data.scripts.dataloader import DataLoading

if __name__ == '__main__':

    train_loader = DataLoading(mode="train", shuffle=False).get_loader()
    test_loader = DataLoading(mode="test", shuffle=False).get_loader()

    # Visualize a single batch by index
    visualize_samples_from_random_batch(train_loader, num_samples = 5)
