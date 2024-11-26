from torch.utils.data import Subset
from data.scripts.dataloader import DataLoading



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

if __name__ == '__main__':
    test_for_overlaps_in_datasets()
