import torch
from torch.cuda.amp import autocast
from scripts.visualization_utils import visualize_outputs

class Trainer:
    def __init__(self, config, model, optimizer, criterion, scaler, device):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.device = device

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for images, masks, _ in train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss
        return total_loss / len(train_loader)
        # return total_loss.item() / len(train_loader)


    def validate_one_epoch(self, loader, metrics, to_visualize=False):
        self.model.eval()
        total_val_loss = 0
        already_visualized = False
        threshold = 0.5
        with torch.no_grad():
            for images, masks, paths in loader:
                images, masks = images.to(self.device), masks.to(self.device)
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                metrics.add_batch(preds, masks)
                total_val_loss += loss
                if to_visualize and not already_visualized:
                    visualize_outputs(self.config, images, masks, preds, paths)
                    already_visualized = True
        return total_val_loss, metrics
