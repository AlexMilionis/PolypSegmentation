import os
import torch
from torch.utils.data import DataLoader
from data.scripts.polyp_dataset import PolypDataset
from data.constants import Constants
from data.scripts.visualization import *


# TODO: Fix augmentation errors, create train/test split, train/test dataloaders


if __name__ == '__main__':
    # Create the dataset
    dataset = PolypDataset(images_dir = Constants.IMAGE_DIR, masks_dir  = Constants.MASK_DIR)

    # Create the DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Visualize a single batch by index
    visualize_samples_from_random_batch(train_loader, num_samples = 5)
