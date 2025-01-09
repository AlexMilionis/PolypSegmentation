import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from src.data.dataset_utils import _create_image_mask_pairs, Transforms


class PolypDataset(Dataset):
    def __init__(self, config, mode, include_data, preload=True):
        self.images_dir = config['paths']['images_dir']
        self.masks_dir  = config['paths']['masks_dir']
        self.mode = mode
        self.include_data = include_data
        assert self.include_data in ['single_frames', 'seq_frames', 'both'], "Use single_frames, seq_frames or both!"

        self.data = _create_image_mask_pairs(self.images_dir, self.masks_dir, include_data=self.include_data)
        self.log_data_to_file(self.data, 'image_mask_log.txt')

        self.image_mask_transform = Transforms.image_and_mask_train_transforms() if self.mode == "train" else Transforms.image_and_mask_val_test_transforms()
        self.image_transform      = Transforms.image_train_transforms() if self.mode == "train" else Transforms.image_val_test_transforms()
        self.mask_transform       = Transforms.mask_train_transforms() if self.mode == "train" else Transforms.mask_val_test_transforms()

        self.preload = preload
        if self.preload:
            self.preloaded_data = [(read_image(img_path), read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY), img_path, mask_path) for img_path, mask_path in self.data]

    def log_data_to_file(self, data, filename):
        # Write the image-mask pairs to a text file
        with open(filename, 'w') as file:
            for img_path, mask_path in data:
                file.write(f"{img_path},{mask_path}\n")  # Format each pair as: image_path, mask_path


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if self.preload:
            image, mask, img_path, mask_path = self.preloaded_data[idx]
        else:
            img_path, mask_path = self.data[idx]
            # Read images as tensors
            image = read_image(img_path)
            mask = read_image(mask_path, mode=torchvision.io.ImageReadMode.GRAY)


        # Convert to tv_tensors
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        # Combined  transformations
        image, mask = self.image_mask_transform(image, mask)
        # Separate transformations
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        return image, mask, (img_path, mask_path)





