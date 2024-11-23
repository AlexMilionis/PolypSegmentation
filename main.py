import os
import torch
from torch.utils.data import DataLoader
from data.scripts.polyp_dataset import PolypDataset
from data.constants import Constants
from data.scripts.visualization import visualize_batch_from_loader


# TODO: Check that all images and masks are padded with zeros on all 4 dimensions, to get in dataloader batches


if __name__ == '__main__':
    # Create the dataset
    dataset = PolypDataset(images_dir = Constants.IMAGE_DIR, masks_dir  = Constants.MASK_DIR)

    # Create the DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Visualize a single batch by index
    visualize_batch_from_loader(train_loader)
