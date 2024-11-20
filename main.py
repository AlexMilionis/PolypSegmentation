from torch.utils.data import DataLoader
from data.scripts.polyp_dataset import PolypDataset
from data.constants import Constants

# TODO: Check that all images and masks are padded with zeros on all 4 dimensions, to get in dataloader batches


if __name__ == '__main__':
    # Create the dataset
    dataset = PolypDataset(
        images_dir=Constants.IMAGE_DIR,
        masks_dir=Constants.MASK_DIR,
    )




    # # Create the DataLoader
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    #
    # # Iterate over the DataLoader
    # for images, masks in train_loader:
    #     print(f"Images batch shape: {images.shape}, Masks batch shape: {masks.shape}")
