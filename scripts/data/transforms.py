import albumentations as A
from albumentations.pytorch import ToTensorV2
from scripts.constants import Constants
import numpy as np
import cv2


class Transforms:

    @staticmethod
    def train_transforms():
        return A.Compose(
            [
            A.Resize(*Constants.image_size, interpolation=cv2.INTER_LANCZOS4),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(
                p=0.5, brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01
            ),
            A.Affine(
                p=0.5,
                scale=(0.5, 1.5),
                translate_percent=0.125,
                rotate=90,
                interpolation=cv2.INTER_LANCZOS4,
            ),
            A.ElasticTransform(p=0.5, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(
                mean=Constants.DATASET_MEANS,
                std=Constants.DATASET_STDS,
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
        seed=42
        )
    
    @staticmethod
    def val_transforms():
        return A.Compose([
            A.Resize(*Constants.image_size, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(
                mean=Constants.DATASET_MEANS,
                std=Constants.DATASET_STDS,
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
        seed=42
        ) 
    
    @staticmethod
    def test_transforms():
        return A.Compose([
            A.Resize(*Constants.image_size, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(
                mean=Constants.DATASET_MEANS,
                std=Constants.DATASET_STDS,
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
        seed=42
        )
    
# class Transforms:

#     @staticmethod
#     def image_and_mask_train_transforms():

#         return A.Compose([
#             # ============= Spatial Transformations =============
#             # Random crop/scale with area preservation
#             A.RandomResizedCrop(size=(512,512), 
#                                 scale=(0.5, 1.0), 
#                                 interpolation=cv2.INTER_LINEAR, 
#                                 mask_interpolation=cv2.INTER_NEAREST, 
#                                 p=1),
#             # Aggressive flips & rotations
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.ShiftScaleRotate(
#                 shift_limit=0.1, 
#                 scale_limit=0.2, 
#                 rotate_limit=25, 
#                 p=0.5),
#             A.GridDistortion(
#                 num_steps=5,
#                 distort_limit=0.3,
#                 p=0.4
#             ),
#             # A.ElasticTransform(alpha=25, sigma=6, p=0.3),

#             # ============= Intensity Transformations =============
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.15, 
#                 contrast_limit=0.15, 
#                 p=0.5),
#             A.GaussianBlur(blur_limit=(3, 7), p=0.3),

            
#             # Color / intensity transforms on image only
#             # A.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1, p=0.2),
#             # A.RandomGamma(gamma_limit=(80, 120), p=0.2),

#             # A.CoarseDropout(
#             #     num_holes_range=(2, 8),       # Replaces min_holes/max_holes
#             #     hole_height_range=(32, 32),   # Fixed size (32px)
#             #     hole_width_range=(32, 32),    # Fixed size (32px)
#             #     fill=0,                       # Same as fill_value=0
#             #     fill_mask=0,                  # Replaces mask_fill_value
#             #     p=0.4
#             # ),
#             A.Normalize(
#                 mean=Constants.TRAIN_DATA_MEANS, 
#                 std=Constants.TRAIN_DATA_STDS, 
#                 max_pixel_value=1),
#             ToTensorV2(transpose_mask=True), # [height, width, num_channels] -> [num_channels, height, width]
#         ])

#     @staticmethod
#     def image_and_mask_val_test_transforms():

#         return A.Compose([
#             A.Resize(
#                 height=512, 
#                 width=512, 
#                 interpolation=cv2.INTER_LINEAR, 
#                 mask_interpolation=cv2.INTER_NEAREST),
#             A.Normalize(mean=Constants.TRAIN_DATA_MEANS, std=Constants.TRAIN_DATA_STDS, max_pixel_value=1),
#             ToTensorV2(transpose_mask=True),    # [height, width, num_channels] -> [num_channels, height, width]
#         ])
