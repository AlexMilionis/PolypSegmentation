import torch
import numpy as np
from scripts.models.model_utils import ModelManager
from tqdm import tqdm
# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler
from scripts.experiments.experiment_utils import ExperimentLogger
from torch import nn, optim
from torch.optim import lr_scheduler
from scripts.experiments.trainer_engine import Trainer, EarlyStopping, Optimizer
# from scripts.experiments.metrics import Metrics
from scripts.models.loss import Dice_CE_Loss

from scripts.visualizations.visualization_utils import plot_loss_curves


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

        self.opt_object = Optimizer(self.config, self.model)
        
        self.trainer = Trainer(self.config, self.model, self.opt_object.optimizer, self.criterion, self.scaler, self.device)
        self.metrics = []

    def execute_training(self, load_checkpoint=False):
        with tqdm(range(self.num_epochs), desc="Training Epochs") as pbar:
            # early_stopping = EarlyStopping(patience=15)
            # self.metrics = Metrics(self.device)

            for epoch in pbar:                    
                train_loss = self.trainer.train_one_epoch(self.train_loader)

                self.metrics.append({
                    'epoch': epoch+1,
                    'train_loss': train_loss,
                    'val_loss': np.nan,
                    'meanIoU': np.nan,
                    'meanDice': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'accuracy': np.nan,
                })
                val_loss, self.metrics = self.trainer.validate_one_epoch(self.val_loader, self.metrics)

                # self.metrics.compute_metrics(epoch = epoch+1, train_loss = train_loss, val_loss = val_loss)

                # # Save model checkpoint
                # ModelManager.save_model_checkpoint(self.model, self.config, self.metrics, epoch)

                # # Early stopping check
                # early_stopping.check_early_stop(val_loss)
                # if early_stopping.stop_training:
                #     print(f"Early stopping at epoch {epoch}")
                #     break

                self.opt_object.scheduler_step(val_loss)

        # save final model
        ModelManager.save_checkpoint(self.model, self.config)


    def execute_evaluation(self):

        test_loss, self.metrics = self.trainer.validate_one_epoch(self.test_loader, self.metrics, test=True)
        # metrics.compute_metrics(test_loss = test_loss, mode="test")
        ExperimentLogger.log_metrics(self.config, self.metrics)
        plot_loss_curves(self.config)
