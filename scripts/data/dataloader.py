import os
from torch.utils.data import DataLoader
from scripts.data.dataset import PolypDataset
from scripts.seed import set_generator
from scripts.data.transforms import Transforms

class DataLoading:
    def __init__(self, config, mode="cloud"):
        
        if mode=="cloud":
            self.num_workers = config['num_workers']
            self.batch_size = config['batch_size']
        elif mode=="local":
            self.num_workers = 4
            self.batch_size = config['batch_size']
        elif mode=="overfit":
            self.num_workers = 4
            self.batch_size = 4
    
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


    def _build_datasets(self, mode="cloud"):
         
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
        if mode == "overfit":
            # For overfitting test, we can limit the dataset size
            self.train_set = PolypDataset(train_pairs[:3], transform=Transforms.train_transforms())
            self.val_set = PolypDataset(train_pairs[:3], transform=Transforms.val_transforms())
            self.test_set = PolypDataset(train_pairs[:3], transform=Transforms.test_transforms())

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
            shuffle=True, 
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


        

# class DataLoading:


#     def __init__(self, config):
#         self.config = config
#         # self.num_workers = config['num_workers']
#         # self.pin_memory = config['pin_memory']
#         # self.persistent_workers = config['persistent_workers']


#     def get_dataloaders(self, run_in_colab):

#         full_dataset = PolypDataset(
#             images_dir = Constants.TRAIN_VAL_IMAGES_DIR,
#             masks_dir = Constants.TRAIN_VAL_MASKS_DIR,
#             mode="train",
#             preload=False
#         )
#         # print("Full dataset length:", len(full_dataset))  # Debug print
#         # print("Images dir:", Constants.TRAIN_VAL_IMAGES_DIR)
#         # print("Masks dir:", Constants.TRAIN_VAL_MASKS_DIR)

#         val_split = self.config["val_split"]
#         # random_split
#         full_len = len(full_dataset)
#         val_len = int(full_len * val_split)
#         train_len = full_len - val_len

#         train_subset, val_subset = random_split(full_dataset, [train_len, val_len], generator=set_generator(42))

#         # Override val_subset to have val transforms & preload
#         # We'll just re-wrap val_subset.dataset in a new PolypDataset so we can preload
#         # 1) get all the image indices from val_subset
#         val_indices = val_subset.indices  # list of indices
#         # 2) gather the actual paths
#         val_data_pairs = [full_dataset.data_pairs[i] for i in val_indices]
#         # 3) build a new PolypDataset with "val" mode, preload=True
#         #    we artificially pass the same data but different mode
#         val_dataset = PolypDataset(
#             images_dir=Constants.TRAIN_VAL_IMAGES_DIR,
#             masks_dir=Constants.TRAIN_VAL_MASKS_DIR,
#             mode="val",    # deterministic transforms
#             preload=True
#         )
#         # However, we only want the specific subset of data. We'll store them in val_dataset, ignoring the default collect.
#         val_dataset.data_pairs = val_data_pairs
#         # re-run preload step for that subset
#         val_dataset.preloaded_data = []
#         for (img_path, msk_path) in val_data_pairs:
#             img, msk = val_dataset._read_and_transform(img_path, msk_path)
#             val_dataset.preloaded_data.append((img, msk, (img_path, msk_path)))

#         # Similarly for test
#         test_dataset = PolypDataset(
#             images_dir=Constants.TEST_IMAGES_DIR,
#             masks_dir=Constants.TEST_MASKS_DIR,
#             mode="test",     # deterministic transforms
#             preload=True     # preload
#         )

#         # train_subset is still a Subset -> wrap in a DataLoader
#         # val_dataset is a direct Dataset

#         if run_in_colab:
#             train_loader = DataLoader(
#                 train_subset, 
#                 batch_size=self.config['batch_size'], 
#                 num_workers=12, 
#                 pin_memory=True, 
#                 persistent_workers=True, 
#                 shuffle=True,
#                 drop_last=True,
#                 )
#             val_loader = DataLoader(
#                 val_dataset,  
#                 batch_size=self.config['batch_size'], 
#                 num_workers=12,
#                 pin_memory=True, 
#                 persistent_workers=True, 
#                 shuffle=False,
#                 drop_last=False,
#                 )
#             test_loader = DataLoader(
#                 test_dataset, 
#                 batch_size=self.config['batch_size'], 
#                 num_workers=12,
#                 pin_memory=True, 
#                 shuffle=False,
#                 drop_last=False,
#                 )
            
#         else:   # local execution
#             train_loader = DataLoader(
#                 train_subset, 
#                 batch_size=self.config['batch_size'], 
#                 num_workers=4, 
#                 pin_memory=True, 
#                 persistent_workers=True, 
#                 shuffle=True,
#                 drop_last=True,
#                 )
#             val_loader = DataLoader(
#                 val_dataset,  
#                 batch_size=self.config['batch_size'], 
#                 num_workers=4,
#                 pin_memory=True, 
#                 persistent_workers=True, 
#                 shuffle=False,
#                 drop_last=False,
#                 )
#             test_loader = DataLoader(
#                 test_dataset, 
#                 batch_size=self.config['batch_size'], 
#                 num_workers=12,
#                 pin_memory=True, 
#                 shuffle=False,
#                 drop_last=False,
#                 )

#         return train_loader, val_loader, test_loader
