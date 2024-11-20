import os
from torchvision import transforms
import torchvision.transforms.functional as TF


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


def zero_padding_to_max_size(input, target_size=(1920, 1080)):
    """
    Pad input to the target size on all four sides.

    Args:
        input (torch.Tensor): Input image tensor of shape [C, H, W].
        target_size (tuple): Target dimensions (height, width).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded image and mask tensors.
    """
    h, w = input.shape[1], input.shape[2]  # Assuming [C, H, W] shape
    target_h, target_w = target_size

    # Calculate padding
    pad_h = target_h - h
    pad_w = target_w - w

    # Distribute padding evenly on all sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Apply padding
    padding = [pad_left, pad_top, pad_right, pad_bottom]  # Left, Top, Right, Bottom
    padded_input = TF.pad(input, padding, fill=0)  # Fill with 0 for black borders

    return padded_input



def get_dataset_transforms():
    """
    Define and return the transformations to be applied to the dataset.

    The transformations are split into two categories:
    1. `images`: Transformations to be applied to the input images.
       - In the current setup, no transformation is applied (`Lambda(lambda x: x)`).
         This means the images will be returned as-is without any changes.
    2. `masks`: Transformations to be applied to the ground-truth masks.
       - The `binarize_mask` function is applied to convert grayscale masks into binary masks
         (e.g., 0 or 1 values) for segmentation tasks.

    Returns:
        dict: A dictionary with two keys:
              - "images": Transformations for the input images.
              - "masks": Transformations for the ground-truth masks.
    """
    dataset_transforms = {
        "images": transforms.Compose([
            # No transformation applied; images returned as-is.
            transforms.Lambda(zero_padding_to_max_size)
        ]),
        "masks": transforms.Compose([
            transforms.Lambda(zero_padding_to_max_size),
            # Convert grayscale mask to binary mask.
            transforms.Lambda(binarize_mask),
        ])
    }

    return dataset_transforms

