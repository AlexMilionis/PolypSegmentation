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
            # ============= Spatial Transformations =============
            # Random crop/scale with area preservation
            A.RandomResizedCrop(size=(512,512), 
                                scale=(0.5, 1.0), 
                                interpolation=cv2.INTER_LINEAR, 
                                mask_interpolation=cv2.INTER_NEAREST, 
                                p=1),
            # Aggressive flips & rotations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=25, 
                p=0.7),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=cv2.BORDER_CONSTANT,
                # mask_value=0,
                p=0.4
            ),
            # A.ElasticTransform(alpha=25, sigma=6, p=0.3),

            # ============= Intensity Transformations =============
            A.RandomBrightnessContrast(
                brightness_limit=0.15, 
                contrast_limit=0.15, 
                p=0.5),
            
            
            # Color / intensity transforms on image only
            # A.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1, p=0.2),

            # A.RandomGamma(gamma_limit=(80, 120), p=0.2),

            # A.CoarseDropout(
            #     max_holes=8,
            #     max_height=32,
            #     max_width=32,
            #     min_holes=2,
            #     fill_value=0,  # Use dataset mean if normalized
            #     mask_fill_value=0,
            #     p=0.4
            # ),
            
            
            A.Normalize(
                mean=Constants.TRAIN_DATA_MEANS, 
                std=Constants.TRAIN_DATA_STDS, 
                max_pixel_value=1),


            ToTensorV2(transpose_mask=True), # [height, width, num_channels] -> [num_channels, height, width]
        ])

    @staticmethod
    def image_and_mask_val_test_transforms():

        return A.Compose([
            A.Resize(height=512, width=512, interpolation=cv2.INTER_NEAREST, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=Constants.TRAIN_DATA_MEANS, std=Constants.TRAIN_DATA_STDS, max_pixel_value=1),
            ToTensorV2(transpose_mask=True),    # [height, width, num_channels] -> [num_channels, height, width]
        ])

