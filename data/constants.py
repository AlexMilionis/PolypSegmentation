import os

class Constants():
    ABS_PATH     = "D:/repos/MScThesis/github_repo/data"
    DATASET_PATH = os.path.join(ABS_PATH, "raw")
    DST_PATH     = "D:/repos/MScThesis/github_repo/data/processed"
    IMAGE_DIR = os.path.join(DST_PATH, "AllImages")
    MASK_DIR = os.path.join(DST_PATH, "AllMasks")
    INPUT_IMAGE_MAX_SIZE = (1080, 1920)
    # VISUALIZATION_DATA_PATH = os.path.join("D:/repos/MScThesis/github_repo/notebooks/visualization_data",)
