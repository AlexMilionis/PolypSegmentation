#%%
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
#%%
# from segmentation_models_pytorch.encoders import get_encoder_names
# encoders = get_encoder_names()
# print(encoders)

#%%
class UNet(nn.Module):
    """
    A wrapper class for a ResNet-based U-Net segmentation model using `segmentation_models_pytorch`.

    Args:
        encoder_name (str): Name of the encoder (default: "resnet18").
        encoder_weights (str): Pretrained weights for the encoder (default: "imagenet").
        in_channels (int): Number of input channels (default: 3 for RGB).
        classes (int): Number of output classes (default: 2 for binary segmentation).
    """
    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1):
        super(UNet, self).__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes

        # Build the U-Net model
        self.model = self._build_model()

    def _build_model(self):
        # Use the segmentation_models_pytorch library to build the U-Net
        return smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.classes,
        )

    def forward(self, x):
        # Forward pass
        return self.model(x)

    def summary(self):
        # Log a summary of the model architecture
        print(self.model)


# if __name__ == "__main__":
#     model = UNet(encoder_name="resnet18", encoder_weights="imagenet")
#     model.summary()