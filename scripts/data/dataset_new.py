import os
from torchvision.io import read_image, ImageReadMode
from scripts.data.transforms import Transforms
import torchvision
from torch.utils.data import Dataset

class PolypDataset(Dataset):

    def __init__(self, images_dir, masks_dir, mode="train", preload=False):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.mode = mode
        self.preload = preload

        # Gather all (image_path, mask_path) pairs
        self.data_pairs = self._collect_image_mask_pairs()

        # Create transformations
        # "train" => random data augmentations
        # "val"/"test" => deterministic transforms
        if self.mode == "train":
            self.image_mask_transform = Transforms.image_and_mask_train_transforms()
            self.image_transform = Transforms.image_train_transforms()
            self.mask_transform = Transforms.mask_train_transforms()
        elif self.mode in ["val", "test"]:
            self.image_mask_transform = Transforms.image_and_mask_val_test_transforms()
            self.image_transform = Transforms.image_val_test_transforms()
            self.mask_transform = Transforms.mask_val_test_transforms()

        # If preload => load & transform entire dataset here
        if self.preload:
            self.preloaded_data = []
            for (img_path, msk_path) in self.data_pairs:
                img, msk = self._read_and_transform(img_path, msk_path)
                self.preloaded_data.append( (img, msk, (img_path, msk_path)) )

    def _collect_image_mask_pairs(self):
        """Collect all (img_path, mask_path) from the directories."""
        img_files = sorted(os.listdir(self.images_dir))
        data_pairs = []
        for fn in img_files:
            img_path = os.path.join(self.images_dir, fn)
            base_name = os.path.splitext(fn)[0]
            mask_fn = f"{base_name}_mask.jpg"  # or some other pattern
            msk_path = os.path.join(self.masks_dir, mask_fn)
            if os.path.exists(msk_path):
                data_pairs.append( (img_path, msk_path) )
        return data_pairs


    # def _read_and_transform(self, img_path, msk_path):
    #     # read images and masks
    #     img = read_image(img_path)
    #     msk = read_image(msk_path, mode=torchvision.io.ImageReadMode.GRAY)
    #     # transform images and masks
    #     img, msk = self.image_mask_transform(img, msk)
    #     img = self.image_transform(img)
    #     msk = self.mask_transform(msk)
    #     return img, msk

    def _read_and_transform(self, img_path, msk_path):
        # Read images
        img = read_image(img_path, mode=ImageReadMode.RGB)  # shape (3,H,W)
        msk = read_image(msk_path, mode=ImageReadMode.GRAY)  # shape (1,H,W)

        # Convert them into the dict format
        sample = {"input": img, "mask": msk}

        # 1) Apply the dictionary-based transform
        sample = self.image_mask_transform(sample)
        # => This should resize/crop/flip both 'input' & 'mask' identically

        # 2) Retrieve them back
        img, msk = sample["input"], sample["mask"]

        # 3) (Optional) Additional transforms on just the image or mask
        img = self.image_transform(img)
        msk = self.mask_transform(msk)


    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if self.preload:
            # Return from memory
            return self.preloaded_data[idx]
        else:
            # Read+transform now
            img_path, msk_path = self.data_pairs[idx]
            img, msk = self._read_and_transform(img_path, msk_path)
            return img, msk, (img_path, msk_path)


