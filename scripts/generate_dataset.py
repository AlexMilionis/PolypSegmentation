import os
import shutil
from typing import List
from PIL import Image
from constants import Constants

def create_folders(target_path: str, new_dirs: List[str]) -> None:
    """
    Create directories for the dataset if they don't already exist.

    Args:
        target_path (str): Base path where directories will be created.
        new_dirs (List[str]): List of directory names to create.
    """
    for item in new_dirs:
        os.makedirs(os.path.join(target_path, item), exist_ok=True)



def create_black_mask(image_path: str, save_path: str) -> None:
    """
    Create a black mask for a given image.

    Args:
        image_path (str): Path to the input image.
        save_path (str): Path to save the generated black mask.
    """
    img = Image.open(image_path)
    black_mask = Image.new("L", img.size, 0)  # Create black mask
    black_mask.save(save_path)



def create_dataset(dataset_path: str, include_seq_frames: bool = False) -> None:
    """
    Create dataset structure by combining single-frame and optional sequence data.

    Args:
        dataset_path (str): Base path where dataset will be created.
        include_seq_frames (bool): If True, include sequence frames in the dataset.
    """
    target_path = os.path.join(dataset_path, "Generated")
    new_dirs = ["AllImages", "AllMasks"]
    create_folders(target_path, new_dirs)

    # Define paths to target folders
    all_images_path = os.path.join(target_path, "AllImages")
    all_masks_path = os.path.join(target_path, "AllMasks")

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
                        shutil.copy(file_path, os.path.join(all_images_path, file))
            elif "masks_seq" in root:
                for file in files:
                    if file.endswith("_mask.jpg"):  # Process masks
                        file_path = os.path.join(root, file)
                        shutil.copy(file_path, os.path.join(all_masks_path, file))


        # Process negative sequences
        seq_negative_path = os.path.join(dataset_path, "sequenceData", "negativeOnly")
        for root, _, files in os.walk(seq_negative_path):
            for file in files:
                if file.endswith((".jpg", ".png")):  # Process images
                    file_path = os.path.join(root, file)
                    shutil.copy(file_path, os.path.join(all_images_path, file))
                    # Create a black mask for each negative image
                    black_mask_name = os.path.splitext(file)[0] + "_mask.jpg"
                    black_mask_path = os.path.join(all_masks_path, black_mask_name)
                    create_black_mask(file_path, black_mask_path)

    print("Dataset creation complete. Check the 'Generated' folder.")

create_dataset(Constants.DATASET_PATH, include_seq_frames = False)
