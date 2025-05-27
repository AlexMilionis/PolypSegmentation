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

        


# class PolypDataset(Dataset):

#     def __init__(self, images_dir, masks_dir, mode="train", preload=False):
#         super().__init__()
#         self.images_dir = images_dir
#         self.masks_dir  = masks_dir
#         self.mode = mode
#         self.preload = preload

#         # Gather all (image_path, mask_path) pairs
#         self.data_pairs = self._collect_image_mask_pairs()

#         # Create transformations
#         # "train" => random data augmentations
#         # "val"/"test" => deterministic transforms
#         if self.mode == "train":
#             # self.image_mask_transform = Transforms.image_and_mask_train_transforms()
#             # self.image_transform = Transforms.image_train_transforms()
#             # self.mask_transform = Transforms.mask_train_transforms()
#             self.transform = Transforms.image_and_mask_train_transforms()
#         elif self.mode in ["val", "test"]:
#             # self.image_mask_transform = Transforms.image_and_mask_val_test_transforms()
#             # self.image_transform = Transforms.image_val_test_transforms()
#             # self.mask_transform = Transforms.mask_val_test_transforms()
#             self.transform = Transforms.image_and_mask_val_test_transforms()

#         # If preload => load & transform entire dataset here
#         if self.preload:
#             self.preloaded_data = []
#             for (img_path, msk_path) in self.data_pairs:
#                 img, msk = self._read_and_transform(img_path, msk_path)
#                 self.preloaded_data.append( (img, msk, (img_path, msk_path)) )


#     def _collect_image_mask_pairs(self):
#         """Collect all (img_path, mask_path) from the directories."""
#         img_files = sorted(os.listdir(self.images_dir))
#         data_pairs = []
#         for fn in img_files:
#             img_path = os.path.join(self.images_dir, fn)
#             base_name = os.path.splitext(fn)[0]
#             mask_fn = f"{base_name}_mask.jpg"
#             msk_path = os.path.join(self.masks_dir, mask_fn)
#             if os.path.exists(msk_path):
#                 data_pairs.append( (img_path, msk_path) )
#         return data_pairs


#     def _read_and_transform(self, img_path, msk_path):
#         # read images and masks
#         # img = read_image(img_path)
#         # msk = read_image(msk_path, mode=torchvision.io.ImageReadMode.GRAY)

#         image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # shape (H,W,3)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask  = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)  # shape (H,W)

#         if image is None or mask is None:
#             print(f"[WARNING] Could not read {img_path} or {msk_path}. Returning None.")
#             return None
        
#         # mask shape (H,W) => (H,W,1)
#         mask = np.expand_dims(mask, axis=-1)  # shape (H,W,1)
        
#         # # 2) Convert BGR -> RGB if you prefer
#         # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#         # masks convert to binary, convert to float
#         mask = (mask > 128).astype(np.float32)

#         # convert images to unit range [0,1]
#         image = (image / 255.0).astype(np.float32) 

#         # 3) Apply Albumentations
#         augmented = self.transform(image=image, mask=mask)

#         # 4) Extract results
#         # aug_img = augmented["image"]  # torch.Tensor shape (C,H,W) converted to int type  # torch.Tensor shape (C,H,W)
#         # aug_mask = augmented["mask"].unsqueeze(0) # torch.Tensor shape (1,H,W)
#         aug_img, aug_mask = augmented["image"], augmented["mask"]

#         # # print maximum values, minimum values for debugging
#         # print(f"aug_img min: {aug_img.min()}, aug_img max:, {aug_img.max()}")
#         # print(f"aug_mask min: {aug_mask.min()}, aug_mask max:, {aug_mask.max()}")

     
#         return aug_img, aug_mask


#     def __len__(self):
#         return len(self.data_pairs)

#     def __getitem__(self, idx):
#         if self.preload:
#             # Return from memory
#             return self.preloaded_data[idx]
#         else:
#             # Read+transform now
#             img_path, msk_path = self.data_pairs[idx]
#             img, msk = self._read_and_transform(img_path, msk_path)
            
#             return img, msk, (img_path, msk_path)


