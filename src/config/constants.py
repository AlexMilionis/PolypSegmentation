"""
This script defines the `Constants` class, which contains key configuration paths and
constants for the polyp segmentation project. These constants streamline the management
of file paths and normalization parameters for the dataset.

Class: Constants
- Attributes:
  - `ABS_PATH`: Absolute path to the project's root directory.
  - `DATASET_PATH`: Path to the raw dataset directory.
  - `DST_PATH`: Path to the directory where processed data will be stored.
  - `IMAGE_DIR`: Path to the directory containing processed images.
  - `MASK_DIR`: Path to the directory containing processed masks.
  - `IMAGENET_COLOR_MEANS`: Mean pixel values for each channel (RGB) based on ImageNet statistics.
  - `IMAGENET_COLOR_STDS`: Standard deviation for each channel (RGB) based on ImageNet statistics.
"""


import os


class Constants():
    PROJECT_PATH = "//"
    ABS_PATH     = os.path.join(PROJECT_PATH, "data/")
    DATASET_PATH = os.path.join(ABS_PATH, "raw/")
    DST_PATH     = os.path.join(ABS_PATH, "processed/")
    IMAGE_DIR = os.path.join(DST_PATH, "AllImages")
    MASK_DIR = os.path.join(DST_PATH, "AllMasks")
    IMAGENET_COLOR_MEANS = [0.485, 0.456, 0.406]
    IMAGENET_COLOR_STDS = [0.229, 0.224, 0.225]
    MODEL_CHECKPOINT_DIR = os.path.join(PROJECT_PATH, "models/")