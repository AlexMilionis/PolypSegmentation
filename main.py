import os
import torch
from torch.utils.data import DataLoader
from data.scripts.polyp_dataset import PolypDataset
# from data.scripts.split_data import train_test
from data.constants import Constants
from data.scripts.visualization import *
from data.scripts.dataloader import DataLoading
from scripts.train import train_model
from torch.utils.data import Subset
from scripts.seed import set_seed

if __name__ == '__main__':
    set_seed()

    train_loader = DataLoading(mode="train", include_data = 'both', shuffle=False).get_loader()
    test_loader = DataLoading(mode="test", include_data = 'both',shuffle=False).get_loader()

    #
    # first_batch = next(iter(train_loader))
    # images, masks, paths = first_batch
    # print(images.shape, masks.shape)

    train_model(train_loader)

    # # Visualize a single batch by index
    # visualize_samples_from_random_batch(train_loader, num_samples = 3)

