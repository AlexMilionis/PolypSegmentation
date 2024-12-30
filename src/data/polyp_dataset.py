"""
This script defines a custom PyTorch Dataset class, `PolypDataset`, for loading and preprocessing
polyp segmentation data, including image-mask pairs. It integrates image and mask transformations
for both training and testing modes.

Main Features:
- Automatically pairs images and their corresponding masks using file naming conventions.
- Supports separate transformations for images and masks, tailored for training or testing.
- Handles different data types: single frames, sequence frames, or a combination of both.

Class: PolypDataset
- Attributes:
  - `images_dir`: Directory containing input images.
  - `masks_dir`: Directory containing ground truth masks.
  - `mode`: Dataset mode, either "train" or "test".
  - `include_data`: Type of data to include ("single_frames", "seq_frames", or "both").
  - `image_mask_transform`: Combined transformations for both images and masks.
  - `image_transform`: Transformations for images only.
  - `mask_transform`: Transformations for masks only.

- Methods:
  1. `__len__`: Returns the total number of samples in the dataset.
  2. `__getitem__`: Retrieves and applies transformations to the image-mask pair at a specified index.
  3. `tensor_to_tv_tensor`: Converts standard tensors to torchvision tensors for images and masks.
"""


import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from src.data.dataset_utils import create_image_mask_pairs, Transforms
from src.config.constants import Constants


class PolypDataset(Dataset):
    def __init__(self, mode, include_data):
        self.images_dir = Constants.IMAGE_DIR
        self.masks_dir  = Constants.MASK_DIR
        self.mode = mode
        self.include_data = include_data
        assert self.include_data in ['single_frames', 'seq_frames', 'both'], "Use single_frames, seq_frames or both!"
        self.data = create_image_mask_pairs(self.images_dir, self.masks_dir, include_data=self.include_data)
        self.image_mask_transform = Transforms.image_and_mask_train_transforms() if self.mode == "train" else Transforms.image_and_mask_val_test_transforms()
        self.image_transform      = Transforms.image_train_transforms() if self.mode == "train" else Transforms.image_val_test_transforms()
        self.mask_transform       = Transforms.mask_train_transforms() if self.mode == "train" else Transforms.mask_val_test_transforms()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # set_seed()
        img_path, mask_path = self.data[idx]
        # Read images as tensors
        image = read_image(img_path)
        mask = read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY)
        # Convert to tv_tensors
        # image, mask = PolypDataset.convert_to_tv_tensor(image, mask)
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        # Combined image and mask transformations
        image, mask = self.image_mask_transform(image, mask)
        # image transformations
        image = self.image_transform(image)
        # mask transformations
        mask = self.mask_transform(mask)

        return image, mask, (img_path, mask_path)





