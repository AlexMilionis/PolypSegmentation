import segmentation_models_pytorch as smp
import torch.nn as nn

class DiceAndBCE(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')  # For binary segmentation
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, outputs, targets):
        dice_loss = self.dice_loss(outputs, targets)
        bce_loss = self.bce_loss(outputs, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss