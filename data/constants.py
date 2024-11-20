import os

class Constants():
    ABS_PATH     = "D:/repos/MScThesis/github_repo/data"
    DATASET_PATH = os.path.join(ABS_PATH, "raw")
    DST_PATH     = "D:/repos/MScThesis/github_repo/data/processed"
    IMAGE_DIR = os.path.join(DST_PATH, "AllImages")
    MASK_DIR = os.path.join(DST_PATH, "AllMasks")