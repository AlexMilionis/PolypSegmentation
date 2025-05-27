from torch.utils.data import Subset
from scripts.data.dataloader import DataLoading
import os
import torch
import sys
from scripts.experiments.experiment_utils import ExperimentLogger


# def test_for_overlaps_in_datasets():
#     # Helper function to get dataset indices
#     def extract_dataset_indices(dataset):
#         if isinstance(dataset, Subset):
#             return set(dataset.indices)
#         raise ValueError("Dataset must be a Subset instance.")
#     # Initialize DataLoaders
#     train_loader = DataLoading(mode="train", shuffle=False).get_loaders()
#     test_loader = DataLoading(mode="test", shuffle=False).get_loaders()
#     # Extract train and test datasets from DataLoading
#     train_dataset, test_dataset = DataLoading.split_data(DataLoading(mode="train"))
#     # Validate indices
#     train_indices = extract_dataset_indices(train_dataset)
#     test_indices = extract_dataset_indices(test_dataset)
#     # Check for overlap
#     overlap = train_indices & test_indices
#     if len(overlap) == 0:
#         print("No overlap found")
#     else:
#         print(f"Overlap between train and test sets: {overlap}")
#
#
# def count_images(config):
#     singleframe_images_count, seqframe_images_count = 0,0
#     singleframe_masks_count, seqframe_masks_count   = 0,0
#     for img in os.listdir(config['paths']['images_dir']):
#         if "seq_" in img:
#             seqframe_images_count += 1
#         else:
#             singleframe_images_count += 1
#     print(f"Singleframe images count: {singleframe_images_count}, Sequence images count: {seqframe_images_count}")
#     for mask in os.listdir(config['paths']['masks_dir']):
#         if "seq_" in mask:
#             seqframe_masks_count += 1
#         else:
#             singleframe_masks_count += 1
#     print(f"Singleframe masks count: {singleframe_images_count}, Sequence masks count: {seqframe_images_count}")
#
#
# def class_balance(include_data):
#     train_loader, val_loader, test_loader = DataLoading(include_data=include_data, shuffle=False).get_loaders()
#     for loader in (train_loader, val_loader, test_loader):
#         positive_count, negative_count = 0, 0
#         for img, mask, _ in loader:
#             for single_mask in mask:
#                 if torch.sum(single_mask == 1).item():  # Check for 1's (white pixels) in the mask
#                     positive_count += 1
#                 else:
#                     negative_count += 1
#             # print(mask.shape, img.shape)
#         # print(f'{mode.capitalize()} mode:')
#         print(f"Positive count: {positive_count}, Negative count: {negative_count}")
#         # print(f"Negative count: {negative_count}")


if __name__ == '__main__':
    # Initialize accumulators and pixel count
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_pixels = 0
    if len(sys.argv) > 1: config_name = sys.argv[1]
    
    config = ExperimentLogger.load_config(config_name)
    
    # Get training data loader
    train_loader, _, _ = DataLoading(config).get_loaders(viz=False)
    
    # Initialize accumulators and pixel count
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_pixels = 0
    
    # Iterate over the loader
    for batch in train_loader:
        images = batch[0]  # shape: [B, C, H, W]
        batch_size, channels, height, width = images.shape
        total_pixels += batch_size * height * width
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])
    
    # Compute per-channel mean and std
    mean = channel_sum / total_pixels
    std = torch.sqrt(channel_sum_sq / total_pixels - mean ** 2)
    
    print(f'Mean: {mean}')
    print(f'Standard Deviation: {std}')

    # Custom x-axis ticks
    # max_epoch = 100
    # desired_ticks = [1] + [i for i in range(25, max_epoch + 1, 25)]
    # print(desired_ticks)
    # # xticks = list(range(1, max_epoch + 1))

    # # initialize torch tensor with values from 0 to 255
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tensor = torch.arange(0, 256, dtype=torch.uint8).view(1, 256)
    # print(tensor)
    # tensor = tensor.to(device, dtype=torch.float16)
    # print(tensor)