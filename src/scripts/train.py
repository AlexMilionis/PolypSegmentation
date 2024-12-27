import torch
from tqdm import tqdm
from src.scripts.metrics import Metrics
from src.scripts.utils import ExperimentLogger
from src.experiments.exp1.run_unet import ExperimentImplementation
from torch.cuda.amp import autocast, GradScaler


class Trainer(ExperimentImplementation):
    def __init__(self, train_loader, val_loader):
        super().__init__(train_loader, val_loader)


    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for images, masks, _ in self.train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            #   Forward pass with mixed precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def _validate_one_epoch(self):
        pass

    def train(self):
        pass

    #   save model