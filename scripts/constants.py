import os


class Constants():
    
    # TRAIN_VAL_IMAGES_DIR = os.path.join("data", "data", "train_val", "images")
    # TRAIN_VAL_MASKS_DIR = os.path.join("data", "data", "train_val", "masks")
    # TEST_IMAGES_DIR = os.path.join("data", "data", "test", "images")
    # TEST_MASKS_DIR = os.path.join("data", "data", "test", "masks")
    IMAGENET_COLOR_MEANS = [0.485, 0.456, 0.406]
    IMAGENET_COLOR_STDS = [0.229, 0.224, 0.225]
    DATASET_MEANS = (0.5543, 0.3644, 0.2777)
    DATASET_STDS  = (0.2840, 0.2101, 0.1770)
    image_size = (512, 512) 
    