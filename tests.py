from torch.utils.data import Subset
from data.scripts.dataloader import DataLoading
from data.constants import Constants
from hyperparameters import Hyperparameters
import os
import torch

def test_for_overlaps_in_datasets():

    # Helper function to get dataset indices
    def extract_dataset_indices(dataset):
        if isinstance(dataset, Subset):
            return set(dataset.indices)
        raise ValueError("Dataset must be a Subset instance.")

    # Initialize DataLoaders
    train_loader = DataLoading(mode="train", shuffle=False).get_loader()
    test_loader = DataLoading(mode="test", shuffle=False).get_loader()

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

def class_balance(mode):
    loader = DataLoading(mode=mode, include_data='both', shuffle=False).get_loader()

    positive_count, negative_count = 0, 0
    for img, mask, _ in loader:  # Loop through the DataLoader
        # Loop through each sample in the batch
        for single_mask in mask:
            if torch.sum(single_mask == 1).item():  # Check for 1's (white pixels) in the mask
                positive_count += 1
            else:
                negative_count += 1
        # print(mask.shape, img.shape)
    print(f'{mode.capitalize()} mode:')
    print(f"Positive count: {positive_count}")
    print(f"Negative count: {negative_count}")


if __name__ == '__main__':
    class_balance("test")
