import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from src.data.transforms import Transforms
from src.data.dataset_utils import CreateDataset

class PolypDataset(Dataset):
    def __init__(self, mode, preload=True):
        # self.images_dir = config['paths']['images_dir']
        # self.masks_dir  = config['paths']['masks_dir']
        self.mode = mode
        # self.include_data = include_data
        # assert self.include_data in ['single_frames', 'seq_frames', 'both'], "Use single_frames, seq_frames or both!"

        self.image_mask_transform = Transforms.image_and_mask_train_transforms() if self.mode == "train" else Transforms.image_and_mask_val_test_transforms()
        self.image_transform      = Transforms.image_train_transforms() if self.mode == "train" else Transforms.image_val_test_transforms()
        self.mask_transform       = Transforms.mask_train_transforms() if self.mode == "train" else Transforms.mask_val_test_transforms()

        #   preload images and masks to memory, and apply transformations
        self.preload = preload
        if self.preload:
            self.data = [
                self._apply_transforms(read_image(img_path), read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY), img_path, mask_path) for img_path, mask_path in CreateDataset.get_image_mask_pairs(self.mode)
            ]
        else:
            self.data = CreateDataset.get_image_mask_pairs(self.mode)

    def _apply_transforms(self, image, mask, img_path, mask_path):
        # Convert to tv_tensors
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        # Combined  transformations
        image, mask = self.image_mask_transform(image, mask)
        # Separate transformations
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        return image, mask, (img_path, mask_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.preload:
            # image, mask, img_path, mask_path = self.preloaded_data[idx]
            return self.data[idx]
        else:
            img_path, mask_path = self.data[idx]
            # Read images as tensors
            image = read_image(img_path)
            mask = read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY)
            return self._apply_transforms(image, mask, img_path, mask_path)
