import torch.nn as nn
import monai.losses as monai_losses



class Dice_CE_Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss_fn = monai_losses.DiceCELoss(
            sigmoid = config['loss']['sigmoid'],  # Apply sigmoid to logits (binary segmentation)
            include_background=config['loss']['include_background'],  # First channel = foreground (polyps)
            lambda_dice=config['loss']['lambda_dice'],
            lambda_ce=config['loss']['lambda_ce'],
            # weight=torch.tensor(config["loss"]["weight"]).cuda()  # Optional class balancing
        )

    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tensor): Raw logits from model (shape: B, 1, H, W)
            targets (Tensor): Ground truth masks (shape: B, 1, H, W) with values 0/1
        """
        return self.loss_fn(outputs, targets)
