import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
# from PIL import Image
from data.scripts.dataset_utils import *

class PolypDataset(Dataset):
    """
    A custom Dataset class for loading images and masks.
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_mask_transform = apply_image_and_mask_transforms()
        self.image_transform      = apply_image_transforms()
        self.mask_transform       = apply_mask_transforms()
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

        # Read images as tensors
        image = read_image(img_path)
        mask = read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY)

        # Apply transformations
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        if self.image_mask_transform:
            # Convert to tv_tensors
            # image = tv_tensors.Image(image)
            # mask = tv_tensors.Mask(mask)
            image, mask = self.image_mask_transform(image, mask)
        # print(image.shape, mask.shape)

        return image, mask, (img_path, mask_path)


