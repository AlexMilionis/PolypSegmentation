import os


class Constants():
    # PROJECT_PATH = "D:/repos/MScThesis/github_repo/"
    # # ABS_PATH     = os.path.join(PROJECT_PATH, "data/")
    # RAW_DATASET_PATH = os.path.join(PROJECT_PATH, "data/raw/")
    # PROCESSED_DATASET_PATH     = os.path.join(PROJECT_PATH, "data/initial/")
    # IMAGE_DIR = os.path.join(PROCESSED_DATASET_PATH, "AllImages")
    # MASK_DIR = os.path.join(PROCESSED_DATASET_PATH, "AllMasks")
    TRAIN_VAL_IMAGES_DIR = os.path.join("data", "data", "train_val", "images")
    TRAIN_VAL_MASKS_DIR = os.path.join("data", "data", "train_val", "masks")
    TEST_IMAGES_DIR = os.path.join("data", "data", "test", "images")
    TEST_MASKS_DIR = os.path.join("data", "data", "test", "masks")
    IMAGENET_COLOR_MEANS = [0.485, 0.456, 0.406]
    IMAGENET_COLOR_STDS = [0.229, 0.224, 0.225]
    TRAIN_DATA_MEANS = [0.5543, 0.3644, 0.2777]#[0.2783, 0.3649, 0.5551] #[ 0.0821, -0.5092, -0.6079]
    TRAIN_DATA_STDS  = [0.2840, 0.2101, 0.1770]#[0.1766, 0.2094, 0.2836] #[1.3483, 1.0123, 0.8642]
    # RESULTS_DIR = os.path.join(PROJECT_PATH, "results/")
    # CONFIG_DIR = os.path.join(PROJECT_PATH, "scripts/configurations/configurations/")