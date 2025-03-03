from torchvision.transforms import v2 as T
from scripts.constants import Constants


class Transforms():

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
