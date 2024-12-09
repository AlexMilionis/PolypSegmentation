import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from data.scripts.dataset_utils import create_image_mask_pairs, Transforms
from torchvision.transforms.functional import to_tensor
from constants import Constants


class InMemoryPolypDataset(Dataset):
    def __init__(self, mode="train", include_data="both"):
        self.images_dir = Constants.IMAGE_DIR
        self.masks_dir = Constants.MASK_DIR
        self.mode = mode
        self.include_data = include_data
        assert self.include_data in ['single_frames', 'seq_frames', 'both'], "Use single_frames, seq_frames or both!"
        self.data = []

        # Preload all images and masks into memory
        file_paths = create_image_mask_pairs(self.images_dir, self.masks_dir, include_data=self.include_data)
        for img_path, mask_path in file_paths:
            image = read_image(img_path)
            mask = read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY)
            self.data.append((image, mask, (img_path, mask_path)))

        self.image_mask_transform = Transforms.image_and_mask_train_transforms() if self.mode == "train" else Transforms.image_and_mask_test_transforms()
        self.image_transform = Transforms.image_train_transforms() if self.mode == "train" else Transforms.image_test_transforms()
        self.mask_transform = Transforms.mask_train_transforms() if self.mode == "train" else Transforms.mask_test_transforms()


    @staticmethod
    def tensor_to_tv_tensor(image, mask, direct=False):
        if direct:
            image = tv_tensors.Image(image)
            mask = tv_tensors.Mask(mask)
        else:
            image = to_tensor(image)
            mask = to_tensor(mask)
        return image, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask, paths = self.data[idx]
        # Combined image and mask transformations
        image, mask = self.image_mask_transform(InMemoryPolypDataset.tensor_to_tv_tensor(image,mask,direct=True))
        # image transformations
        image = self.image_transform(image)
        # mask transformations
        mask = self.mask_transform(mask)
        return image, mask, paths

