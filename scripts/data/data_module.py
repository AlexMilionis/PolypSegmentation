import os
from torch.utils.data import DataLoader
from scripts.seed import set_generator
from scripts.data.transformations import Transforms
from torch.utils.data import Dataset
import cv2

class PolypDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.transform = transform
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        try:
            
            image = cv2.imread(self.pairs[idx][0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.pairs[idx][1], cv2.IMREAD_GRAYSCALE)
            mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

            image_path = self.pairs[idx][0].split("\\")[-1].split(".")[0]
            mask_path = self.pairs[idx][1].split("\\")[-1].split(".")[0]
            
            return image, mask.long().unsqueeze(0), (image_path, mask_path)
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            print(f"Image: {self.pairs[idx][0]}, Mask: {self.pairs[idx][1]}")
            
            return None, None, (None, None)


class DataLoading:
    def __init__(self, config, mode="cloud"):

        self.mode = mode
        if self.mode=="cloud":
            self.num_workers = config['num_workers']
            self.batch_size = config['batch_size']
        elif self.mode=="local":
            self.num_workers = 4
            self.batch_size = config['batch_size']
        elif self.mode=="overfit":
            self.num_workers = 4
            self.batch_size = 5
            self.overfit_samples = 5

    @staticmethod
    def _create_image_mask_pairs(images, masks):
        image_to_mask_pairs = []
        mask_dict = {}
        for mask_path in masks:
            # Extract the base name (removing '_mask' suffix)
            mask_filename = os.path.basename(mask_path)
            base_name = mask_filename.split("_mask")[0]
            mask_dict[base_name] = mask_path
        # Match each image with its mask
        for image_path in images:
            image_filename = os.path.basename(image_path)
            base_name = os.path.splitext(image_filename)[0]  # Remove extension
            
            if base_name in mask_dict:
                image_to_mask_pairs.append((image_path, mask_dict[base_name]))
            else:
                print(f"No matching mask found for: {image_path}")
        return image_to_mask_pairs

    def _build_datasets(self):
         
        # Load all images and masks from the directories
        train_images = os.listdir("data/train/images")
        train_masks = os.listdir("data/train/masks")
        train_images = [os.path.join("data/train/images", img) for img in train_images]
        train_masks = [os.path.join("data/train/masks", mask) for mask in train_masks]

        val_images = os.listdir("data/val/images")
        val_masks = os.listdir("data/val/masks")
        val_images = [os.path.join("data/val/images", img) for img in val_images]
        val_masks = [os.path.join("data/val/masks", mask) for mask in val_masks]

        test_images = os.listdir("data/test/images")
        test_masks = os.listdir("data/test/masks")
        test_images = [os.path.join("data/test/images", img) for img in test_images]
        test_masks = [os.path.join("data/test/masks", mask) for mask in test_masks]

        # Create pairs of (image_path, mask_path)
        train_pairs = DataLoading._create_image_mask_pairs(train_images, train_masks)
        val_pairs   = DataLoading._create_image_mask_pairs(val_images, val_masks)
        test_pairs  = DataLoading._create_image_mask_pairs(test_images, test_masks)

        # Create datasets
        if self.mode == "overfit":
            overfit_pairs = train_pairs[:self.overfit_samples]
            
            self.train_set = PolypDataset(overfit_pairs, transform=Transforms.val_transforms())
            self.val_set = PolypDataset(overfit_pairs, transform=Transforms.val_transforms())
            self.test_set = PolypDataset(overfit_pairs, transform=Transforms.val_transforms())

        else:
            self.train_set = PolypDataset(train_pairs, transform=Transforms.train_transforms())
            self.val_set = PolypDataset(val_pairs, transform=Transforms.val_transforms())
            self.test_set = PolypDataset(test_pairs, transform=Transforms.test_transforms())

    def build_loaders(self):

        self._build_datasets()

        generator = set_generator()

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=True, 
            shuffle=True if self.mode != "overfit" else False, 
            drop_last=True,
            generator=generator,
            )
        
        
        self.val_loader = DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=True, 
            shuffle=False, 
            drop_last=False,
            generator=generator,
            )
        
        self.test_loader = DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            shuffle=False, 
            drop_last=False,
            generator=generator,
        )

