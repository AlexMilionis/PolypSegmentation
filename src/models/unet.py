"""
This script defines the `UNet` class, a wrapper around the U-Net architecture
from the `segmentation_models_pytorch` library. The class is designed for binary
or multi-class semantic segmentation tasks and allows for flexible configuration
of the encoder backbone.

Class: UNet
- Inherits: `torch.nn.Module`
- Purpose: Builds a U-Net model with customizable encoder and parameters.

Attributes:
- `encoder_name` (str): Name of the encoder backbone.
- `encoder_weights` (str): Pretrained weights for the encoder.
- `in_channels` (int): Number of input channels (default is 3 for RGB images).
- `classes` (int): Number of output channels (e.g., 1 for binary segmentation).

Methods:
- `_build_model()`: Constructs the U-Net using the `segmentation_models_pytorch` library.
- `forward(x)`: Defines the forward pass for input tensor `x`.
- `summary()`: Logs a detailed summary of the model architecture.
"""


import segmentation_models_pytorch as smp
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, transfer_learning, encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1):
        super(UNet, self).__init__()
        self.name = "UNet"
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.transfer_learning = transfer_learning
        self.model = self._build_model()


    def _build_model(self):
        model = smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.classes,
        )
        if self.transfer_learning:
            # Freeze the encoder weights for transfer learning
            for param in model.encoder.parameters():
                param.requires_grad = False
        return model


    def forward(self, x):
        return self.model(x)


    def summary(self):
        print(self.model)