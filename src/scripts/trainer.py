import torch
from torch.cuda.amp import autocast
from src.scripts.metrics import Metrics


class Trainer:
    def __init__(self, model, optimizer, criterion, scaler, device):
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
            total_loss += loss.item()
        return total_loss / len(train_loader)


    def validate_one_epoch(self, loader):
        self.model.eval()
        total_val_loss = 0
        val_metrics = Metrics()
        with torch.no_grad():
            for images, masks, _ in loader:
                images, masks = images.to(self.device), masks.to(self.device)
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_metrics.add_batch(preds, masks)
                total_val_loss += loss.item()
        return total_val_loss, val_metrics