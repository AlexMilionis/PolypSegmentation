"""
Testing and Analysis Utilities for Datasets:

1. **Purpose:**
   This script provides utility functions for analyzing and validating datasets during training and testing phases.
   It includes functionality to test for overlaps between datasets, count images and masks, and analyze the class balance.

2. **Functions:**

   - `test_for_overlaps_in_datasets()`:
     - Checks for overlapping samples between the training and testing datasets.
     - Uses indices from the DataLoader's Subset objects to identify overlaps.

   - `count_images()`:
     - Counts and reports the number of single-frame and sequence-frame images and masks in the dataset.

   - `class_balance(mode)`:
     - Analyzes the class distribution (positive vs. negative masks) in a specified dataset mode (`train` or `test`).
     - Reports the counts of positive (pixels with value 1) and negative (pixels with value 0) samples.

"""


from torch.utils.data import Subset
from src.data.dataloader import DataLoading
from src.config.constants import Constants
import os
import torch

def test_for_overlaps_in_datasets():
    # Helper function to get dataset indices
    def extract_dataset_indices(dataset):
        if isinstance(dataset, Subset):
            return set(dataset.indices)
        raise ValueError("Dataset must be a Subset instance.")
    # Initialize DataLoaders
    train_loader = DataLoading(mode="train", shuffle=False).get_loaders()
    test_loader = DataLoading(mode="test", shuffle=False).get_loaders()
    # Extract train and test datasets from DataLoading
    train_dataset, test_dataset = DataLoading.split_data(DataLoading(mode="train"))
    # Validate indices
    train_indices = extract_dataset_indices(train_dataset)
    test_indices = extract_dataset_indices(test_dataset)
    # Check for overlap
    overlap = train_indices & test_indices
    if len(overlap) == 0:
        print("No overlap found")
    else:
        print(f"Overlap between train and test sets: {overlap}")


def count_images():
    singleframe_images_count, seqframe_images_count = 0,0
    singleframe_masks_count, seqframe_masks_count   = 0,0
    for img in os.listdir(Constants.IMAGE_DIR):
        if "seq_" in img:
            seqframe_images_count += 1
        else:
            singleframe_images_count += 1
    print(f"Singleframe images count: {singleframe_images_count}, Sequence images count: {seqframe_images_count}")
    for mask in os.listdir(Constants.MASK_DIR):
        if "seq_" in mask:
            seqframe_masks_count += 1
        else:
            singleframe_masks_count += 1
    print(f"Singleframe masks count: {singleframe_images_count}, Sequence masks count: {seqframe_images_count}")


def class_balance(include_data):
    train_loader, val_loader, test_loader = DataLoading(include_data=include_data, shuffle=False).get_loaders()
    for loader in (train_loader, val_loader, test_loader):
        positive_count, negative_count = 0, 0
        for img, mask, _ in loader:
            for single_mask in mask:
                if torch.sum(single_mask == 1).item():  # Check for 1's (white pixels) in the mask
                    positive_count += 1
                else:
                    negative_count += 1
            # print(mask.shape, img.shape)
        # print(f'{mode.capitalize()} mode:')
        print(f"Positive count: {positive_count}, Negative count: {negative_count}")
        # print(f"Negative count: {negative_count}")


if __name__ == '__main__':
    # count_images()
    # test_for_overlaps_in_datasets()
    # for data in ["both", "single_frames", "seq_frames"]:
    #     class_balance("train",include_data=data)
    #     class_balance("test",include_data=data)
    pass