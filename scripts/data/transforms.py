from torchvision.transforms import v2 as T
from scripts.constants import Constants


class Transforms():

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def binarize_mask(mask):
        return (mask > 128).float()

    @staticmethod
    def convert_to_float(image):
        return image.float()

    @staticmethod
    def convert_to_01_range(image):
        return image / 255.0

    @staticmethod
    def image_and_mask_train_transforms():
        return T.Compose([
            T.RandomResizedCrop(size=(512, 512), scale=(0.5, 2)),
            T.RandomHorizontalFlip(p=0.2),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
            T.ElasticTransform(alpha=25, sigma=6, interpolation=T.InterpolationMode.BILINEAR, fill=0)
        ])

    @staticmethod
    def image_and_mask_val_test_transforms():
        return T.Compose([
            T.Resize(size=(512, 512)),
        ])

    @staticmethod
    def image_train_transforms():

        return T.Compose([
            T.Lambda(Transforms.convert_to_float),
            T.Lambda(Transforms.convert_to_01_range),
            T.ColorJitter(brightness=0.25, contrast=0.25, hue=0.25),
            T.RandomGrayscale(p=0.1),  # convert the image to grayscale
            # T.GaussianBlur(kernel_size=(3, 3)), # blur the image with a 3x3 kernel
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.Normalize(mean=Constants.TRAIN_DATA_MEANS, std=Constants.TRAIN_DATA_STDS),
        ])

    @staticmethod
    def image_val_test_transforms():
        return T.Compose([
            T.Lambda(Transforms.convert_to_01_range),
            T.Normalize(mean=Constants.TRAIN_DATA_MEANS, std=Constants.TRAIN_DATA_STDS),
        ])

    @staticmethod
    def mask_train_transforms():
        return T.Compose([
            T.Lambda(Transforms.binarize_mask),
        ])

    @staticmethod
    def mask_val_test_transforms():
        return T.Compose([
            T.Lambda(Transforms.binarize_mask),
        ])


# # scripts/data/transforms.py
# from torchvision.transforms import v2 as T, InterpolationMode
# import torch
#
# # TODO: understand this, or better use Albumentations
# class Transforms_new:
#     @staticmethod
#     def train_dict_transforms():
#         """
#         Return a Compose that expects a dict:
#         {
#           "input": (C,H,W) image tensor,
#           "mask":  (1,H,W) mask tensor
#         }
#         and returns them with random augmentations applied in sync.
#         """
#         return T.Compose([
#             # RandomResizedCrop applies to both "input" & "mask" by default
#             T.RandomResizedCrop(size=(512, 512),
#                                 scale=(0.5, 1.5),
#                                 interpolation=InterpolationMode.BILINEAR),
#             T.RandomHorizontalFlip(prob=0.2),  # flips both input & mask
#             T.RandomVerticalFlip(prob=0.2),
#             T.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
#             # If you want an elastic transform that handles the mask too:
#             # T.ElasticTransform(...)
#             # etc.
#             T.ElasticTransform(alpha=25, sigma=6, interpolation=T.InterpolationMode.BILINEAR, fill=0)
#         ])
#
#     @staticmethod
#     def val_dict_transforms():
#         """
#         Deterministic resizing for val/test.
#         """
#         return T.Compose([
#             T.Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR),
#         ])
#
#     @staticmethod
#     def image_extra_transform(mode):
#         """
#         Optionally add color, normalization, or other image-only transforms
#         AFTER the dictionary transform.
#         We'll just do a few examples here.
#         """
#         from torchvision.transforms import functional as F
#
#         def _image_only_transform(img_tensor: torch.Tensor) -> torch.Tensor:
#             # Convert to float [0..1]
#             img_tensor = img_tensor.float() / 255.0
#             # Example: color jitter only if train
#             if mode == "train":
#                 # manually do random color ops, or do a T.ColorJitter on single image
#                 # e.g. T.ColorJitter(0.2,0.2,0.2,0.1)(img_tensor)
#                 pass
#             # e.g. normalization
#             # means = [0.485, 0.456, 0.406]
#             # stds  = [0.229, 0.224, 0.225]
#             # ...
#             return img_tensor
#
#         return _image_only_transform  # a function that takes a tensor
#
#     @staticmethod
#     def mask_extra_transform(mode):
#         """
#         For the mask, maybe binarize it or convert to float, etc.
#         """
#         from torchvision.transforms import functional as F
#
#         def _mask_only_transform(msk_tensor: torch.Tensor) -> torch.Tensor:
#             # Already shape (1,H,W), typical mask usage
#             # maybe we want to binarize:
#             return (msk_tensor > 0).float()
#         return _mask_only_transform
