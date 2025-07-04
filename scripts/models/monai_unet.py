import monai.networks.nets as monai_nets
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_dims = config['model']['spatial_dims']
        self.in_channels = config['model']['in_channels']
        self.classes = config['model']['out_channels']
        self.channels = config['model']['channels']
        self.strides = config['model']['strides']
        self.num_res_units = config['model']['num_res_units']
        self.norm = config['model']['norm']
        self.activation = config['model']['act']
        self.dropout = config['model']['dropout']
        self.bias = config['model']['bias']
        self.model = self._build_model()


    def _build_model(self):
        return monai_nets.UNet(
            spatial_dims=self.spatial_dims,  # <-- 2D mode
            in_channels=self.in_channels,
            out_channels=self.classes,
            channels=self.channels,  # Example channel progression
            strides=self.strides,  # Downsampling steps
            num_res_units=self.num_res_units,  # Residual blocks per stage
            norm=self.norm,  # BatchNorm by default
            bias = self.bias,
            act=self.activation,  # Activation function
            dropout=self.dropout
        )

    def forward(self, x):
        return self.model(x)