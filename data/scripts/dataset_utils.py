import os
from torchvision.transforms import v2 as T
from data.constants import Constants
import numpy as np

def create_image_mask_pairs(images_dir, masks_dir):
    """
    Creates a list of tuples (image, mask) where the mask corresponds to the image
    based on the naming convention: image name + "_mask" = mask name.

    Args:
        images_dir (str): Directory containing images.
        masks_dir (str): Directory containing masks.

    Returns:
        list: List of tuples [(image_path, mask_path), ...].

    Raises:
        ValueError: If an image does not have a corresponding mask.
    """
    # Get sorted lists of image and mask filenames
    images = os.listdir(images_dir)
    masks = os.listdir(masks_dir)

    # Create list of tuples
    image_mask_pairs = []
    for image in images:
        base_image_name, _ = os.path.splitext(image)
        expected_mask_name = f"{base_image_name}_mask.jpg"  # Adjust extension if necessary

        mask_path = os.path.join(masks_dir, expected_mask_name)
        if os.path.exists(mask_path):
            image_path = os.path.join(images_dir, image)
            image_mask_pairs.append((image_path, mask_path))
        else:
            raise ValueError(f"No matching mask found for image {image}")

    return image_mask_pairs



def identity_transform(x):
    """
    A no-op transformation that simply returns the input as is.
    """
    return x



def binarize_mask(mask):
    """
    Converts a grayscale mask (tensor) to binary by applying a threshold.

    Args:
        mask (torch.Tensor): Input mask as a tensor with shape [1, H, W].

    Returns:
        torch.Tensor: Binarized mask with values 0 or 1.
    """
    return (mask > 128).float()


def convert_to_01_range(image):
    return image/255


def image_transforms():
    return T.Compose([
        T.Lambda(convert_to_01_range),
        T.Normalize(mean = Constants.IMAGENET_COLOR_MEANS,
                    std  = Constants.IMAGENET_COLOR_STDS)
    ])

def mask_transforms():
    return T.Compose([
        T.Lambda(binarize_mask)
    ])

def image_and_mask_transforms():
    return T.Compose([
        T.RandomResizedCrop(size=(512, 512), scale=(0.5, 2.0)),
        T.RandomHorizontalFlip(p=0.5),
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