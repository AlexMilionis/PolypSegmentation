import os

class Constants():
    ABS_PATH     = "D:/repos/MScThesis/github_repo/data"
    DATASET_PATH = os.path.join(ABS_PATH, "raw")
    DST_PATH     = "D:/repos/MScThesis/github_repo/data/processed"
    IMAGE_DIR = os.path.join(DST_PATH, "AllImages")
    MASK_DIR = os.path.join(DST_PATH, "AllMasks")
    # INPUT_IMAGE_MAX_SIZE = (2048, 2048) #(1080,1920)
    IMAGENET_COLOR_MEANS = [0.485, 0.456, 0.406]
    IMAGENET_COLOR_STDS = [0.229, 0.224, 0.225]