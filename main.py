import os
import torch
from torch.utils.data import DataLoader
from data.scripts.polyp_dataset import PolypDataset
from data.constants import Constants
from data.scripts.visualization import *


# TODO: 1.create train/test split,
#       2. better documentation on all scripts
#       3. write down all processes before continuing further,


if __name__ == '__main__':
    # Create the dataset
    dataset = PolypDataset(images_dir = Constants.IMAGE_DIR, masks_dir  = Constants.MASK_DIR)

    # Create the DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Visualize a single batch by index
    visualize_samples_from_random_batch(train_loader, num_samples = 5)
