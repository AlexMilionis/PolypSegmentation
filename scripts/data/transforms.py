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
            T.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.2)),
            T.RandomHorizontalFlip(p=0.2),
            # T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
        ])

    @staticmethod
    def image_and_mask_val_test_transforms():
        return T.Compose([
            T.Resize(size=(512, 512)),
        ])

    @staticmethod
    def image_train_transforms():
        # return T.Compose([
        #     T.Lambda(Transforms.convert_to_float),
        #     # T.ColorJitter(brightness=0.1, contrast=0.1),
        #     T.RandomGrayscale(p=0.1),   # convert the image to grayscale
        #     # T.GaussianBlur(kernel_size=(3, 3)), # blur the image with a 3x3 kernel
        #     T.RandomApply([T.GaussianBlur(3)], p=0.2),
        #     T.Lambda(Transforms.convert_to_01_range),
        #     T.Normalize(mean=Constants.MEANS, std=Constants.STDS),
        # ])
        return T.Compose([
            T.Lambda(Transforms.convert_to_float),
            T.Lambda(Transforms.convert_to_01_range),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.RandomGrayscale(p=0.1),  # convert the image to grayscale
            # T.GaussianBlur(kernel_size=(3, 3)), # blur the image with a 3x3 kernel
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.Normalize(mean=Constants.MEANS, std=Constants.STDS),
        ])

    @staticmethod
    def image_val_test_transforms():
        return T.Compose([
            T.Lambda(Transforms.convert_to_01_range),
            T.Normalize(mean=Constants.MEANS, std=Constants.STDS),
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
