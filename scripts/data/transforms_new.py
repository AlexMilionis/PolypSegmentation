import albumentations as A
from albumentations.pytorch import ToTensorV2
from scripts.constants import Constants
import numpy as np

class Transforms:

    @staticmethod
    def image_and_mask_train_transforms():
        """
        Returns an Albumentations Compose that:
         - random-resizes/crops, flips, rotates, etc.
         - normalizes images
         - transforms them to torch.Tensor
        """
        return A.Compose([
            # Spatial transforms (apply equally to image & mask)
            A.RandomResizedCrop(size=(512,512), scale=(0.5, 1.0), p=1),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0,
                               rotate_limit=15, border_mode=0, p=0.5),
            A.ElasticTransform(alpha=25, sigma=6, p=0.3),

            # Color / intensity transforms on image only
            A.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),

            # Normalization (assuming 0-255 -> normalized)
            A.Normalize(mean=Constants.MEANS, std=Constants.STDS, max_pixel_value=255.0,),

            # Convert both image & mask to torch tensors
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})

    @staticmethod
    def image_and_mask_val_test_transforms():
        """
        Deterministic resizing & normalization for val/test.
        """
        return A.Compose([
            A.Resize(height=512, width=512),
            # A.Normalize(mean=Constants.MEANS, std=Constants.STDS, max_pixel_value=255.0,),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})

    # Optional: If you want to binarize the mask after loading
    @staticmethod
    def binarize_mask(mask):
        # If your masks are grayscale [0..255], then threshold:
        # return (mask > 128).float()
        return (mask > 128).astype(np.float32)

