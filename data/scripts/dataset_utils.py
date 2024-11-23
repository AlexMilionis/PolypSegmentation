import os
import torchvision.transforms.v2 as transformsv2
import torchvision.transforms.functional as TF
from data.constants import Constants


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


# def zero_padding_to_max_size(input):
#     """
#     Pad input to the target size on all four sides.
#
#     Args:
#         input (torch.Tensor): Input image tensor of shape [C, H, W].
#         target_size (tuple): Target dimensions (height, width).
#
#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: Padded image and mask tensors.
#     """
#     h, w = input.shape[1], input.shape[2]  # Assuming [C, H, W] shape
#     target_h, target_w = Constants.INPUT_IMAGE_MAX_SIZE
#
#     # Calculate padding
#     pad_h = target_h - h
#     pad_w = target_w - w
#
#     # Distribute padding evenly on all sides
#     pad_top = pad_h // 2
#     pad_bottom = pad_h - pad_top
#     pad_left = pad_w // 2
#     pad_right = pad_w - pad_left
#
#     # Apply padding
#     padding = [pad_left, pad_top, pad_right, pad_bottom]  # Left, Top, Right, Bottom
#     padded_input = TF.pad(input, padding, fill=0)  # Fill with 0 for black borders
#
#     return padded_input


class PadToMaxSize:
    """
    Custom transformation to pad an image or mask to a specified maximum size.

    Args:
        max_size (tuple): The target size (height, width).
        fill (int, optional): The padding value. Defaults to 0.
    """

    def __init__(self, max_size, fill=0):
        self.max_size = max_size
        self.fill = fill

    def __call__(self, img):
        """
        Apply the padding transformation to the input image or mask.

        Args:
            img (PIL Image or Tensor): The input image or mask.

        Returns:
            Tensor: The padded image or mask.
        """
        # Get current size
        h, w = TF.get_size(img)  # Extract height and width

        # Calculate padding
        target_h, target_w = self.max_size
        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Image size {h}x{w} is larger than the target size {target_h}x{target_w}."
            )

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding using torchvision.transforms.v2.Pad
        return transformsv2.Pad([pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)(img)


class TransformDicts():

    binarize_masks       = transformsv2.Lambda(binarize_mask)
    identity             = transformsv2.Lambda(identity_transform)
    zero_padding         = PadToMaxSize(max_size=Constants.INPUT_IMAGE_MAX_SIZE, fill=0)  # Pad to max size
    randomresizedcrop    = transformsv2.RandomResizedCrop(size=(512, 512), scale=(0.8, 2.0)) # Resize and crop
    randomcrop           = transformsv2.RandomCrop(size=(512, 512))
    randomhorizontalflip = transformsv2.RandomHorizontalFlip(p=1)
    # converttotensor      = transformsv2.ToTensor(),  # Convert image to tensor
    # converttopil              = transformsv2.ToTensor(),
    normalize            = transformsv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize



def get_image_and_mask_transforms():

    data_masks_shared_transforms = transformsv2.Compose([
        transformsv2.RandomCrop(size=(512, 512))
    ])
    return data_masks_shared_transforms



def get_mask_transforms():

    mask_transforms = transformsv2.Compose([
        transformsv2.Lambda(binarize_mask)
    ])

    return mask_transforms

