import os
import shutil
from typing import List
from PIL import Image
from torchvision.transforms import v2 as T
from src.config.constants import Constants
import numpy as np
import random


class CreateDataset:
    @staticmethod
    def _create_folders(target_path: str, new_dirs: List[str]) -> None:
        for item in new_dirs:
            os.makedirs(os.path.join(target_path, item), exist_ok=True)

    @staticmethod
    def _create_black_mask(image_path: str, save_path: str) -> None:
        img = Image.open(image_path)
        black_mask = Image.new("L", img.size, 0)  # Create black mask
        black_mask.save(save_path)

    @staticmethod
    def create_initial_dataset(config, include_seq_frames: bool = False) -> None:
        dataset_path = config['paths']['raw_dataset_dir']
        dst_path = config['paths']['initial_dataset_dir']
        # Define paths for initial data
        processed_dirs = ["AllImages", "AllMasks"]
        CreateDataset._create_folders(dst_path, processed_dirs)
        # Define paths to target folders
        all_images_path = os.path.join(dst_path, "AllImages")
        all_masks_path = os.path.join(dst_path, "AllMasks")
        # Collect single-frame images and masks
        img_folders = [f"data_C{i}/images_C{i}" for i in range(1, 7)]
        mask_folders = [f"data_C{i}/masks_C{i}" for i in range(1, 7)]
        # Copy single-frame images
        for folder in img_folders:
            data_path = os.path.join(dataset_path, folder)
            for img in os.listdir(data_path):
                img_path = os.path.join(data_path, img)
                shutil.copy(img_path, os.path.join(all_images_path, img))
        # Copy single-frame masks
        for folder in mask_folders:
            data_path = os.path.join(dataset_path, folder)
            for mask in os.listdir(data_path):
                mask_path = os.path.join(data_path, mask)
                shutil.copy(mask_path, os.path.join(all_masks_path, mask))
        # Collect sequence-frame images and masks if required
        if include_seq_frames:
            # Process positive sequences
            seq_positive_path = os.path.join(dataset_path, "sequenceData", "positive")
            for root, dirs, files in os.walk(seq_positive_path):
                # Check for `images_seqi` and `masks_seqi` subfolders
                if "images_seq" in root:
                    for file in files:
                        if file.endswith((".jpg", ".png")):  # Process images
                            file_path = os.path.join(root, file)
                            new_name = "seq_" + file
                            shutil.copy(file_path, os.path.join(all_images_path, new_name))
                elif "masks_seq" in root:
                    for file in files:
                        if file.endswith("_mask.jpg"):  # Process masks
                            file_path = os.path.join(root, file)
                            new_name = "seq_" + file
                            shutil.copy(file_path, os.path.join(all_masks_path, new_name))

            # Process negative sequences
            seq_negative_path = os.path.join(dataset_path, "sequenceData", "negativeOnly")
            for root, _, files in os.walk(seq_negative_path):
                for file in files:
                    if file.endswith((".jpg", ".png")):  # Process images
                        file_path = os.path.join(root, file)
                        new_name = "seq_" + file
                        shutil.copy(file_path, os.path.join(all_images_path, new_name))
                        # Create a black mask for each negative image
                        black_mask_name = "seq_" + os.path.splitext(file)[0] + "_mask.jpg"
                        black_mask_path = os.path.join(all_masks_path, black_mask_name)
                        CreateDataset._create_black_mask(file_path, black_mask_path)

        print(f"Dataset creation complete. Check the '{dst_path}' folder.")


    @staticmethod
    def create_processed_datasets(config):
        assert config['dataset']['train_ratio'] + config['dataset']['val_ratio'] + config['dataset']['test_ratio'] == 1.0, "Ratios must sum to 1.0"
        split_dirs = ["train/images", "train/masks",
                  "validation/images", "validation/masks",
                  "test/images", "test/masks"]
        for split_dir in split_dirs:
            os.makedirs(os.path.join(config['paths']['processed_dataset_dir'], split_dir), exist_ok=True)

        all_images_dir = config['paths']['images_dir']
        all_masks_dir = config['paths']['masks_dir']
        image_files = os.listdir(all_images_dir)
        mask_files = os.listdir(all_masks_dir)

        image_mask_pairs = _create_image_mask_pairs(config['paths']['images_dir'], config['paths']['masks_dir'], config['dataset']['include_data'])
        # image_mask_pairs = [(img, os.path.join(all_images_dir, img), os.path.join(all_masks_dir, img))
        #                     for img in image_files if img in mask_files]

        random.shuffle(image_mask_pairs)
        train_count = int(len(image_mask_pairs) * config['dataset']['train_ratio'])
        val_count = int(len(image_mask_pairs) * config['dataset']['val_ratio'])

        train_data = image_mask_pairs[:train_count]
        val_data = image_mask_pairs[train_count:train_count + val_count]
        test_data = image_mask_pairs[train_count + val_count:]

        for split, data in zip(["train", "validation", "test"], [train_data, val_data, test_data]):
            for img_path, mask_path in data:
                img_name = os.path.basename(img_path)
                mask_name = os.path.basename(mask_path)
                shutil.copy(img_path, os.path.join(config['paths']['processed_dataset_dir'], f"{split}/images", img_name))
                shutil.copy(mask_path, os.path.join(config['paths']['processed_dataset_dir'], f"{split}/masks", mask_name))

        print(f"Dataset split complete. Check the '{config['paths']['processed_dataset_dir']}' folder.")


def _create_image_mask_pairs(images_dir, masks_dir, include_data="single_frames"):
    images = sorted(os.listdir(images_dir))
    masks_set = set(os.listdir(masks_dir))
    image_mask_pairs = []

    for image in images:
        base_image_name, ext = os.path.splitext(image)
        if ext.lower() not in ['.jpg', '.png', '.jpeg']:
            continue
        if include_data == "single_frames" and "seq_" in base_image_name:
            continue
        elif include_data == "seq_frames" and "seq_" not in base_image_name:
            continue

        expected_mask_name = f"{base_image_name}_mask.jpg"
        if expected_mask_name in masks_set:
            image_path = os.path.join(images_dir, image)
            mask_path = os.path.join(masks_dir, expected_mask_name)
            image_mask_pairs.append((image_path, mask_path))
        else:
            raise ValueError(f"No matching mask found for image {image}")

    return image_mask_pairs



class Transforms():
    def __init__(self):
        pass

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def binarize_mask(mask):
        return (mask > 128).float()

    @staticmethod
    def convert_to_float(image):
        return image.float()

    @staticmethod
    def convert_to_01_range(image):
        return image / 255.0

    @staticmethod
    def image_and_mask_train_transforms():
        return T.Compose([
            T.RandomResizedCrop(size=(512, 512), scale=(0.5, 2.0)),
            T.RandomHorizontalFlip(p=0.5),
        ])

    @staticmethod
    def image_and_mask_val_test_transforms():
        return T.Compose([
            T.Resize(size=(512, 512)),
        ])

    @staticmethod
    def image_train_transforms():
        return T.Compose([
            T.Lambda(Transforms.convert_to_float),
            T.ColorJitter(),
            T.Lambda(Transforms.convert_to_01_range),
            T.Normalize(mean=Constants.IMAGENET_COLOR_MEANS, std=Constants.IMAGENET_COLOR_STDS),
        ])

    @staticmethod
    def image_val_test_transforms():
        return T.Compose([
            T.Lambda(Transforms.convert_to_01_range),
            T.Normalize(mean=Constants.IMAGENET_COLOR_MEANS, std=Constants.IMAGENET_COLOR_STDS),
        ])

    @staticmethod
    def mask_train_transforms():
        return T.Compose([
            T.Lambda(Transforms.binarize_mask),
        ])

    @staticmethod
    def mask_val_test_transforms():
        return T.Compose([
            T.Lambda(Transforms.binarize_mask),
        ])




