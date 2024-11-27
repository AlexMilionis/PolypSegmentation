import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
# from PIL import Image
from data.scripts.dataset_utils import *
from torchvision.transforms.functional import to_tensor
from data.constants import Constants

class PolypDataset(Dataset):
    """
    A custom Dataset class for loading images and masks.
    """

    def __init__(self, mode="train", include_data="both"):
        self.images_dir = Constants.IMAGE_DIR
        self.masks_dir  = Constants.MASK_DIR
        self.mode = mode
        self.include_data = include_data
        self.data = create_image_mask_pairs(self.images_dir, self.masks_dir, include_data=self.include_data)
        self.image_mask_transform = self._get_image_and_mask_transforms()
        self.image_transform      = self._get_image_transforms()
        self.mask_transform       = self._get_mask_transforms()

        # if self.include_data not in ["single_frames","seq_frames","both"]:
        #     raise ValueError(f"Invalid data '{self.include_data}'. Use 'single_frames', 'seq_frames' or 'both'.")
        assert self.include_data in ['single_frames','seq_frames','both'], "Use single_frames, seq_frames or both!"

    @staticmethod
    def tensor_to_tv_tensor(image, mask, direct = False):
        if direct:
            image = tv_tensors.Image(image)
            mask = tv_tensors.Mask(mask)
        else:
            image = to_tensor(image)
            mask  = to_tensor(mask)
        return image, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the image and mask pair at the specified index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: Transformed image and mask tensors.
        """
        # Get image and mask paths from the pre-aligned data
        img_path, mask_path = self.data[idx]

        # Read images as tensors
        image = read_image(img_path)
        mask = read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY)


        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        if self.image_mask_transform:
            # Convert to tv_tensors
            image, mask = PolypDataset.tensor_to_tv_tensor(image, mask, direct = True)
            image, mask = self.image_mask_transform(image, mask)
            # image, mask = PolypDataset.tensor_to_tv_tensor(image, mask, direct = False)
        # print(img_path, mask_path)
        return image, mask, (img_path, mask_path)

    def _get_image_transforms(self):
        """
        Return image-only transformations based on the mode.
        """
        if self.mode == "train":
            return image_transforms()
        elif self.mode == "test":
            return image_transforms()
        return None  # No transforms for 'full'

    def _get_mask_transforms(self):
        """
        Return mask-only transformations based on the mode.
        """
        if self.mode == "train":
            return mask_transforms()
        elif self.mode == "test":
            return mask_transforms()
        return None  # No transforms for 'full'

    def _get_image_and_mask_transforms(self):
        """
        Return image-mask pair transformations based on the mode.
        """
        if self.mode == "train":
            return image_and_mask_transforms()
        elif self.mode == "test":
            return None  # Example: No combined transforms in 'test' mode
        return None  # No transforms for 'full'