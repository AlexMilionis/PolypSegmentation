"""
This script preprocesses a dataset for semantic segmentation tasks, organizing it into a standardized
structure suitable for training. It supports single-frame and sequence-frame data and includes
functionality to generate black masks for images without ground truth.

Main Features:
- Copies and organizes single-frame images and masks from specified dataset folders.
- Optionally processes sequence-frame data, prefixing related files with 'seq_'.
- Automatically generates black masks for sequence images that lack corresponding masks.
- Ensures the output dataset is stored in a unified structure under specified directories.

Output Structure:
- AllImages: Contains all processed images.
- AllMasks: Contains all processed masks, including generated black masks for negatives.

Functions:
1. `create_folders`: Ensures required directories exist in the target path.
2. `create_black_mask`: Generates black masks for images without corresponding ground truth masks.
3. `create_dataset`: Combines single-frame and optional sequence-frame data into a unified structure.
"""


import os
import shutil
from typing import List
from PIL import Image
from constants import Constants


def create_folders(target_path: str, new_dirs: List[str]) -> None:
    for item in new_dirs:
        os.makedirs(os.path.join(target_path, item), exist_ok=True)


def create_black_mask(image_path: str, save_path: str) -> None:
    img = Image.open(image_path)
    black_mask = Image.new("L", img.size, 0)  # Create black mask
    black_mask.save(save_path)


def create_dataset(dataset_path: str, dst_path: str, include_seq_frames: bool = False) -> None:
    # Define paths for processed data
    processed_dirs = ["AllImages", "AllMasks"]
    create_folders(dst_path, processed_dirs)
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
                    create_black_mask(file_path, black_mask_path)

    print(f"Dataset creation complete. Check the '{dst_path}' folder.")


if __name__ == "__main__":
    create_dataset(Constants.DATASET_PATH, Constants.DST_PATH, include_seq_frames=True)
