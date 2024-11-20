
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import os
from PIL import Image
from torchvision.transforms import ToTensor
from PIL import Image
import torch


# TODO: Inside dataset, apply the transformations
# TODO: Create custom dataloader
# TODO: check where should Dataset, DataLoader go
# TODO: Misaligned names for images,masks

class PolypDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.img_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(self.img_dir)
        self.masks = os.listdir(self.masks_dir)

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)
        return image



