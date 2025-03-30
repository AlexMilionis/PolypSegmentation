import torch, torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from scripts.data.transforms import Transforms
from scripts.data.dataset_utils import CreateDataset

class PolypDataset(Dataset):
    def __init__(self, mode, preload=False):
        # self.images_dir = config['paths']['images_dir']
        # self.masks_dir  = config['paths']['masks_dir']
        self.mode = mode
        # self.include_data = include_data
        # assert self.include_data in ['single_frames', 'seq_frames', 'both'], "Use single_frames, seq_frames or both!"


        if self.mode == "train":
            self.image_mask_transform = Transforms.image_and_mask_train_transforms()
            self.image_transform      = Transforms.image_train_transforms()
            self.mask_transform       = Transforms.mask_train_transforms()
        elif self.mode in ["val", "test"]:
            self.image_mask_transform = Transforms.image_and_mask_val_test_transforms()
            self.image_transform      = Transforms.image_val_test_transforms()
            self.mask_transform       = Transforms.mask_val_test_transforms()

        self.data_pairs = CreateDataset.get_image_mask_pairs(self.mode)
        self.preload = preload

        # if preload True -> images and masks to memory, and apply transformations (static transformations for all training)
        if self.preload:
            self.data = []
            for img_path, mask_path in self.data_pairs:
                img = read_image(img_path)
                mask = read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY)

                # apply the transformations
                img, msk = self.image_mask_transform(img, mask)
                img = self.image_transform(img)
                mask = self.mask_transform(mask)

                # append the transformed image and mask to the list
                self.data.append( (img, mask, (img_path, mask_path)) )


    def __len__(self):
        return len(self.data_pairs) if not self.preload else len(self.data)

    def __getitem__(self, idx):
        if self.preload:    # validation and test data
            return self.data
        else:   # training data
            img_path, mask_path = self.data_pairs[idx]
            img = read_image(img_path)
            mask = read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY)

            # Apply random transformations if train
            img, mask = self.image_mask_transform(img, mask)
            img = self.image_transform(img)
            mask = self.mask_transform(mask)

            return img, mask, (img_path, mask_path)

        # if self.preload:
        #     # image, mask, img_path, mask_path = self.preloaded_data[idx]
        #     return self.data[idx]
        # else:
        #     img_path, mask_path = self.data[idx]
        #     # Read images as tensors
        #     image = read_image(img_path)
        #     mask = read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY)
        #     return self._apply_transforms(image, mask, img_path, mask_path)
