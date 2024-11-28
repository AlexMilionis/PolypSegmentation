import os
from torchvision.transforms import v2 as T
from data.constants import Constants
import numpy as np

def _create_tuple(image, base_image_name, image_mask_pairs, images_dir, masks_dir):
    expected_mask_name = f"{base_image_name}_mask.jpg"  # Adjust extension if necessary

    mask_path = os.path.join(masks_dir, expected_mask_name)
    if os.path.exists(mask_path):
        image_path = os.path.join(images_dir, image)
        image_mask_pairs.append((image_path, mask_path))
        return image_mask_pairs
    else:
        raise ValueError(f"No matching mask found for image {image}")


def create_image_mask_pairs(images_dir, masks_dir, include_data="single_frames"):

    # Get sorted lists of image and mask filenames
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
        """
        A no-op transformation that simply returns the input as is.
        """
        return x

    @staticmethod
    def binarize_mask(mask):
        """
        Converts a grayscale mask (tensor) to binary by applying a threshold.

        Args:
            mask (torch.Tensor): Input mask as a tensor with shape [1, H, W].

        Returns:
            torch.Tensor: Binarized mask with values 0 or 1.
        """
        return (mask > 128).float()

    @staticmethod
    def convert_to_01_range(image):

        return image / 255.0

    @staticmethod
    def image_and_mask_train_transforms():
        """
        Combined image and mask transformations for training.
        """
        return T.Compose([
            T.RandomResizedCrop(size=(512, 512), scale=(0.5, 2.0)),
            T.RandomHorizontalFlip(p=0.5),
        ])

    @staticmethod
    def image_train_transforms():
        """
        Image transformations for training.
        """
        return T.Compose([
            T.Lambda(Transforms.convert_to_01_range),
            T.Normalize(mean=Constants.IMAGENET_COLOR_MEANS, std=Constants.IMAGENET_COLOR_STDS),
        ])

    @staticmethod
    def mask_train_transforms():
        """
        Mask transformations for training.
        """
        return T.Compose([
            T.Lambda(Transforms.binarize_mask),
            # T.Lambda(Transforms.identity_transform)
        ])

    @staticmethod
    def image_and_mask_test_transforms():
        """
        Combined image and mask transformations for testing.
        """
        return T.Compose([
            T.Resize(size=(512, 512)),
        ])

    @staticmethod
    def image_test_transforms():
        """
        Image transformations for training.
        """
        return T.Compose([
            T.Lambda(Transforms.convert_to_01_range),
            T.Normalize(mean=Constants.IMAGENET_COLOR_MEANS, std=Constants.IMAGENET_COLOR_STDS),
        ])

    @staticmethod
    def mask_test_transforms():
        """
        Mask transformations for testing.
        """
        return T.Compose([
            T.Lambda(Transforms.binarize_mask),
            # T.Lambda(Transforms.identity_transform)
        ])


def unnormalize_image(image):
    """
    Unnormalize a normalized image tensor for visualization.

    Args:
        image (torch.Tensor): Normalized image tensor (C, H, W).
        mean (list): Mean values used during normalization (per channel).
        std (list): Standard deviation values used during normalization (per channel).

    Returns:
        numpy.ndarray: Unnormalized image in (H, W, C) format for visualization.
    """
    mean = np.array(Constants.IMAGENET_COLOR_MEANS)
    std = np.array(Constants.IMAGENET_COLOR_STDS)

    # reverse normalization
    unnormalized_image = (image * std) + mean
    unnormalized_image = np.clip(unnormalized_image, a_min = 0, a_max = 1)

    return unnormalized_image

