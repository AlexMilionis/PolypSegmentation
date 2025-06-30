import torch
import numpy as np
from torch.amp import autocast
from scripts.visualizations.visualization_utils import visualize_outputs
from torch import optim
from torch.optim import lr_scheduler
from monai import metrics as mm

class Trainer:
    def __init__(self, config, model, optimizer, criterion, scaler, device):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.device = device

        self.mean_dice = mm.DiceMetric(include_background=False, reduction="mean")
        self.mean_iou = mm.MeanIoU(include_background=False, reduction="mean")
        self.precision = mm.ConfusionMatrixMetric(include_background=False, metric_name="precision")
        self.recall = mm.ConfusionMatrixMetric(include_background=False, metric_name="sensitivity")
        # self.get_f1_score = mm.ConfusionMatrixMetric(include_background=False, metric_name="f1_score")
        self.accuracy = mm.ConfusionMatrixMetric(include_background=False, metric_name="accuracy")


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
                # print(f"Output shape: {outputs.shape}, Mask shape: {masks.shape}")
                # print(f"Outputs: {outputs}, Masks: {masks}")
                loss = self.criterion(outputs, masks) 
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss
            # print(f"Loss: {loss}")

        return total_loss / len(train_loader)

    def validate_one_epoch(self, loader, metrics, test=False):
        self.model.eval()
        total_loss = 0
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
                total_loss += loss

                # Update metrics for all batches
                self.mean_iou(y_pred=preds, y=masks)
                self.mean_dice(y_pred=preds, y=masks)
                self.precision(y_pred=preds, y=masks)
                self.recall(y_pred=preds, y=masks)
                self.accuracy(y_pred=preds, y=masks)

                if test and not already_visualized:
                    visualize_outputs(self.config, images, masks, preds, paths)
                    already_visualized = True

            # Compute final metrics after processing all batches
            computed_loss = total_loss/len(loader)
            
            if test:
                # Only append one test entry after processing all batches
                metrics.append({
                    "epoch": -1,
                    "train_loss": np.nan,
                    "val_loss": computed_loss,
                    "meanIoU": self.mean_iou.aggregate(),
                    "meanDice": self.mean_dice.aggregate(),
                    "precision": self.precision.aggregate(),
                    "recall": self.recall.aggregate(),
                    "accuracy": self.accuracy.aggregate()
                })
            else:   # validation
                metrics[-1]["val_loss"] = computed_loss
                metrics[-1]["meanIoU"] = self.mean_iou.aggregate()
                metrics[-1]["meanDice"] = self.mean_dice.aggregate()
                metrics[-1]["precision"] = self.precision.aggregate()
                metrics[-1]["recall"] = self.recall.aggregate()
                metrics[-1]["accuracy"] = self.accuracy.aggregate()

            # Reset metrics after aggregation
            self.mean_iou.reset()
            self.mean_dice.reset()
            self.precision.reset()
            self.recall.reset()
            self.accuracy.reset()

        return computed_loss, metrics
    

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
                T_max=self.config["epochs"],
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

