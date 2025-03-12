import monai.networks.nets as monai_nets
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config['model']['input_channels']
        self.classes = config['model']['num_classes']
        self.model = self._build_model()

    def _build_model(self):
        return monai_nets.UNet(
            spatial_dims=2,  # <-- 2D mode
            in_channels=self.in_channels,
            out_channels=self.classes,
            channels=(16, 32, 64, 128, 256),  # Example channel progression
            strides=(2, 2, 2, 2),  # Downsampling steps
            num_res_units=2,  # Residual blocks per stage
            norm="BATCH",  # BatchNorm by default
            act="RELU",  # Activation function
        )

    def forward(self, x):
        return self.model(x)