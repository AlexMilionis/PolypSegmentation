import os
from torchvision.transforms import v2 as T
from src.config.constants import Constants
import numpy as np

def _create_tuple(image, base_image_name, image_mask_pairs, images_dir, masks_dir):
    expected_mask_name = f"{base_image_name}_mask.jpg"
    mask_path = os.path.join(masks_dir, expected_mask_name)
    if os.path.exists(mask_path):
        image_path = os.path.join(images_dir, image)
        image_mask_pairs.append((image_path, mask_path))
        return image_mask_pairs
    else:
        raise ValueError(f"No matching mask found for image {image}")


def create_image_mask_pairs(images_dir, masks_dir, include_data="single_frames"):

    # Sorted lists of image and mask filenames
    images = sorted(os.listdir(images_dir))
    masks = sorted(os.listdir(masks_dir))
    # Create list of tuples
    image_mask_pairs = []
    for image in images:
        base_image_name, _ = os.path.splitext(image)
        if include_data == "single_frames":
            if "seq_" not in base_image_name:
                image_mask_pairs = _create_tuple(image, base_image_name, image_mask_pairs, images_dir, masks_dir)
        elif include_data == "seq_frames":
            if "seq_" in base_image_name:
                image_mask_pairs = _create_tuple(image, base_image_name, image_mask_pairs, images_dir, masks_dir)
        else:   # single_frames + seq_frames
            image_mask_pairs = _create_tuple(image, base_image_name, image_mask_pairs, images_dir, masks_dir)


    return image_mask_pairs


class Transforms():
    def __init__(self):
        pass

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
            T.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
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
            T.ColorJitter(),
            T.Lambda(Transforms.convert_to_01_range),
            T.Normalize(mean=Constants.IMAGENET_COLOR_MEANS, std=Constants.IMAGENET_COLOR_STDS),
        ])

    @staticmethod
    def image_val_test_transforms():
        return T.Compose([
            T.Lambda(Transforms.convert_to_01_range),
            T.Normalize(mean=Constants.IMAGENET_COLOR_MEANS, std=Constants.IMAGENET_COLOR_STDS),
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


def unnormalize_image(image):
    mean = np.array(Constants.IMAGENET_COLOR_MEANS)
    std = np.array(Constants.IMAGENET_COLOR_STDS)
    # reverse normalization
    unnormalized_image = (image * std) + mean
    unnormalized_image = np.clip(unnormalized_image, a_min = 0, a_max = 1)
    return unnormalized_image

