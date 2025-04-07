import albumentations as A
from albumentations.pytorch import ToTensorV2
from scripts.constants import Constants
import numpy as np
import cv2


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
            A.RandomResizedCrop(size=(512,512), scale=(0.5, 1.0), interpolation=cv2.INTER_NEAREST, mask_interpolation=cv2.INTER_NEAREST, p=1),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0,
                               rotate_limit=15, border_mode=0, p=0.5),
            A.ElasticTransform(alpha=25, sigma=6, p=0.3),
            
            # Color / intensity transforms on image only
            A.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            
            
            A.Normalize(mean=Constants.TRAIN_DATA_MEANS, std=Constants.TRAIN_DATA_STDS, max_pixel_value=1),


            ToTensorV2(transpose_mask=True), # [height, width, num_channels] -> [num_channels, height, width]
        ])

    @staticmethod
    def image_and_mask_val_test_transforms():

        return A.Compose([
            A.Resize(height=512, width=512, interpolation=cv2.INTER_NEAREST, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=Constants.TRAIN_DATA_MEANS, std=Constants.TRAIN_DATA_STDS, max_pixel_value=1),
            ToTensorV2(transpose_mask=True),    # [height, width, num_channels] -> [num_channels, height, width]
        ])

