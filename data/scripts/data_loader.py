import torch
from torch.utils.data import DataLoader
from data.scripts.polyp_dataset import PolypDataset
from data.constants import Constants

def get_data_loader(images_dir, masks_dir, batch_size, shuffle=True, num_workers=4):
    """
    Create and return a DataLoader for the PolypDataset.

    Args:
        images_dir (str): Path to the directory containing images.
        masks_dir (str): Path to the directory containing masks.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset. Defaults to True.
        num_workers (int): Number of worker threads for data loading. Defaults to 4.

    Returns:
        DataLoader: PyTorch DataLoader for the PolypDataset.
    """
    dataset = PolypDataset(images_dir, masks_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return data_loader
