import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
from data.scripts.dataset_utils import *

class PolypDataset(Dataset):
    """
    A custom Dataset class for loading images and masks.
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.data_transform   = get_image_and_mask_transforms()
        self.target_transform = get_mask_transforms()
        self.data = create_image_mask_pairs(self.images_dir, self.masks_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the image and mask pair at the specified index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: Transformed image and mask tensors.
        """
        # Get image and mask paths from the pre-aligned data
        img_path, mask_path = self.data[idx]

        # Read images as PIL images
        image = read_image(img_path)
        mask = read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY)


        # Apply transformations
        if self.data_transform:
            image, mask = self.data_transform(image, mask)
        if self.target_transform:
            mask = self.target_transform(mask)


        return image, mask, (img_path, mask_path)


