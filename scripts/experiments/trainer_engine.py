import torch
from torch.amp import autocast
from scripts.visualizations.visualization_utils import visualize_outputs
from torch import optim
from torch.optim import lr_scheduler


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
            images = images.to(self.device)
            masks  = masks.to(self.device)
            self.optimizer.zero_grad()
            # Mixed precision forward pass
            with autocast('cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss
        return total_loss / len(train_loader)


    def validate_one_epoch(self, loader, metrics, to_visualize=False):
        self.model.eval()
        total_val_loss = 0
        already_visualized = False
        threshold = 0.5
        with torch.no_grad():
            for images, masks, paths in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                metrics.add_batch(preds, masks)
                total_val_loss += loss
                if to_visualize and not already_visualized: 
                    visualize_outputs(self.config, images, masks, preds, paths)
                    already_visualized = True
        return total_val_loss/len(loader), metrics
    

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


class Optimizer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        optimizer_type = getattr(optim, self.config['optimizer'])
        self.optimizer = optimizer_type(
            self.model.parameters(), 
            lr=float(self.config['learning_rate']), 
            weight_decay=self.config['weight_decay']
            )
        

        if self.config["scheduler"] == "CosineAnnealingLR":
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-5,
            )

        elif self.config["scheduler"] == "ReduceLROnPlateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode="min",
                patience=5,
                factor=0.5,
                verbose=True,
                # min_lr=1e-6
            )

        elif self.config["scheduler"] == "CosineAnnealingWarmRestarts":
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=10,  # Number of iterations for the first restart
                T_mult=2,  # Factor by which to increase T_0 after each restart
                eta_min=1e-6,  # Minimum learning rate
            )


        elif self.config["scheduler"] in [None, "None"]:
            self.scheduler = None

        else:
            raise ValueError(f"Unknown scheduler: {self.config['scheduler']}") 
        

    def scheduler_step(self, val_loss):
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        elif isinstance(self.scheduler, lr_scheduler.CosineAnnealingLR):
            self.scheduler.step()

