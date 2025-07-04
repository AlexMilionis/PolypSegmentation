import torch
import numpy as np
from scripts.models.model_utils import ModelManager
from tqdm import tqdm
# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler, autocast
from scripts.experiments.experiment_utils import ExperimentLogger
from scripts.experiments.trainer_engine import Trainer, EarlyStopping, Optimizer
# from scripts.experiments.metrics import Metrics
from scripts.models.loss import Dice_CE_Loss

from scripts.visualizations.visualization_utils import plot_loss_curves
from monai import metrics as mm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scripts.visualizations.visualization_utils import visualize_outputs



class Experiment:
    def __init__(self, train_loader, val_loader, test_loader, config):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = GradScaler('cuda')  # mixed precision training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ModelManager.load_model(self.config).to(self.device)
        self.num_epochs = config['epochs']
        self.criterion = Dice_CE_Loss(self.config)

        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'], 
            weight_decay=self.config['weight_decay'],
            )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.num_epochs, 
            eta_min=1e-5,
        )
        
        self.trainer = Trainer(self.config, self.model, self.opt_object.optimizer, self.criterion, self.scaler, self.device)
        self.mean_dice = mm.DiceMetric(include_background=False, reduction="mean")
        self.mean_iou = mm.MeanIoU(include_background=False, reduction="mean")
        self.precision = mm.ConfusionMatrixMetric(include_background=False, metric_name="precision")
        self.recall = mm.ConfusionMatrixMetric(include_background=False, metric_name="sensitivity")
        # self.get_f1_score = mm.ConfusionMatrixMetric(include_background=False, metric_name="f1_score")
        self.accuracy = mm.ConfusionMatrixMetric(include_background=False, metric_name="accuracy")


        self.metrics = []

    def execute_training(self, load_checkpoint=False):
        with tqdm(range(self.num_epochs), desc="Training Epochs") as pbar:
            # early_stopping = EarlyStopping(patience=15)
            # self.metrics = Metrics(self.device)

            for epoch in pbar:                    
                # train_loss = self.trainer.train_one_epoch(self.train_loader)

                # Train Loop
                self.model.train()
                total_train_loss = 0
                for images, masks, _ in self.train_loader:
                    images = images.to(self.device)
                    masks  = masks.to(self.device)
                    self.optimizer.zero_grad()
                    # Mixed precision forward pass
                    with autocast('cuda'):
                        outputs = self.model(images)
                        # print(f"Output shape: {outputs.shape}, Mask shape: {masks.shape}")
                        # print(f"Outputs: {outputs}, Masks: {masks}")
                        batch_loss = self.criterion(outputs, masks) 
                        
                    self.scaler.scale(batch_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    total_train_loss += batch_loss

                # Update metrics on train epoch end
                self.metrics.append({
                    'epoch': epoch+1,
                    'train_loss': total_train_loss/len(self.train_loader),
                    'val_loss': np.nan,
                    'meanIoU': np.nan,
                    'meanDice': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'accuracy': np.nan,
                })

                # Validation Loop
                self.model.eval()
                total_val_loss = 0
                threshold = 0.5
                with torch.no_grad():
                    for images, masks, _ in self.val_loader:
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        with autocast('cuda'):
                            outputs = self.model(images)
                            batch_loss = self.criterion(outputs, masks)
                        probs = torch.sigmoid(outputs)
                        preds = (probs > threshold).float()
                        total_val_loss += batch_loss

                        # Update metrics for all batches
                        self.mean_iou(y_pred=preds, y=masks)
                        self.mean_dice(y_pred=preds, y=masks)
                        self.precision(y_pred=preds, y=masks)
                        self.recall(y_pred=preds, y=masks)
                        self.accuracy(y_pred=preds, y=masks)

                    # Update metrics on validation epoch end
                    self.metrics[-1]["val_loss"] = total_val_loss/len(self.val_loader)
                    self.metrics[-1]["meanIoU"] = self.mean_iou.aggregate()
                    self.metrics[-1]["meanDice"] = self.mean_dice.aggregate()
                    self.metrics[-1]["precision"] = self.precision.aggregate()
                    self.metrics[-1]["recall"] = self.recall.aggregate()
                    self.metrics[-1]["accuracy"] = self.accuracy.aggregate()

                    # Reset metrics after aggregation
                    self.mean_iou.reset()
                    self.mean_dice.reset()
                    self.precision.reset()
                    self.recall.reset()
                    self.accuracy.reset()

                # # Early stopping check
                # early_stopping.check_early_stop(val_loss)
                # if early_stopping.stop_training:
                #     print(f"Early stopping at epoch {epoch}")
                #     break

                self.scheduler.step()

        # save final model
        ModelManager.save_checkpoint(self.model, self.config)


    def execute_evaluation(self):

        # test_loss, self.metrics = self.trainer.validate_one_epoch(self.test_loader, self.metrics, test=True)
        self.model.eval()
        total_test_loss = 0
        threshold = 0.5
        already_visualized = False
        with torch.no_grad():
            for images, masks, paths in self.test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                with autocast('cuda'):
                    outputs = self.model(images)
                    batch_loss = self.criterion(outputs, masks)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                total_test_loss += batch_loss

                # Update metrics for all batches
                self.mean_iou(y_pred=preds, y=masks)
                self.mean_dice(y_pred=preds, y=masks)
                self.precision(y_pred=preds, y=masks)
                self.recall(y_pred=preds, y=masks)
                self.accuracy(y_pred=preds, y=masks)

                if not already_visualized:
                    visualize_outputs(self.config, images, masks, preds, paths)
                    already_visualized = True

            # Compute final metrics after processing all batches
            test_loss = total_test_loss/len(self.test_loader)
            
            # Only append one test entry after processing all batches
            self.metrics.append({
                "epoch": -1,
                "train_loss": np.nan,
                "val_loss": test_loss,
                "meanIoU": self.mean_iou.aggregate(),
                "meanDice": self.mean_dice.aggregate(),
                "precision": self.precision.aggregate(),
                "recall": self.recall.aggregate(),
                "accuracy": self.accuracy.aggregate()
                })


            # Reset metrics after aggregation
            self.mean_iou.reset()
            self.mean_dice.reset()
            self.precision.reset()
            self.recall.reset()
            self.accuracy.reset()


        # metrics.compute_metrics(test_loss = test_loss, mode="test")
        ExperimentLogger.log_metrics(self.config, self.metrics)
        plot_loss_curves(self.config)
