import torch
import sys
from scripts.data.dataloader import DataLoading
from scripts.experiments.experiment_utils import ExperimentLogger

def get_train_mean_std(train_loader):
# Initialize accumulators and pixel count
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_pixels = 0
    # if len(sys.argv) > 1: config_name = sys.argv[1]
    
    # config = ExperimentLogger.load_config(config_name)
    
    # Get training data loader
    # train_loader, _, _ = DataLoading(config).get_loaders(viz=False)
    
    # Initialize accumulators and pixel count
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_pixels = 0
    
    # Iterate over the loader
    for batch in train_loader:
        
        images = batch[0]  # shape: [B, C, H, W]
        # images = images.float() / 255.0
        batch_size, channels, height, width = images.shape
        total_pixels += batch_size * height * width
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])
    
    # Compute per-channel mean and std
    mean = channel_sum / total_pixels
    std = torch.sqrt(channel_sum_sq / total_pixels - mean ** 2)
    return mean, std

