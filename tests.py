from torch.utils.data import Subset
from data.scripts.dataloader import DataLoading
from data.constants import Constants
import os

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

if __name__ == '__main__':
    # test_for_overlaps_in_datasets()
    count_images()