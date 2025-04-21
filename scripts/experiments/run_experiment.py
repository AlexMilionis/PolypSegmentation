import torch

from scripts.models.model_utils import ModelManager
from tqdm import tqdm
# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler
from scripts.experiments.experiment_utils import ExperimentLogger
from torch import nn, optim
from torch.optim import lr_scheduler
from scripts.experiments.trainer_engine import Trainer, EarlyStopping, Optimizer
from scripts.experiments.metrics import Metrics
from scripts.experiments.loss import Dice_CE_Loss

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


    def execute_training(self, load_checkpoint=False):
        with tqdm(range(self.num_epochs), desc="Training Epochs") as pbar:
            # Initialize early stopping 
            early_stopping = EarlyStopping(patience=15)
            # Initialize metrics
            self.metrics = Metrics(self.device, self.config)
            for epoch in pbar:                    
                train_loss = self.trainer.train_one_epoch(self.train_loader)
                val_loss, self.metrics = self.trainer.validate_one_epoch(self.val_loader, self.metrics)
                self.metrics.compute_metrics(epoch = epoch+1, train_loss = train_loss, val_loss = val_loss)
                # Early stopping check
                early_stopping.check_early_stop(val_loss)
                if early_stopping.stop_training:
                    print(f"Early stopping at epoch {epoch}")
                    break

                self.opt_object.scheduler_step(val_loss)

                # Save model checkpoint
                ModelManager.save_model_checkpoint(self.model, self.config, self.metrics, epoch)
        return self.metrics


    def execute_evaluation(self, metrics):

        test_loss, metrics = self.trainer.validate_one_epoch(self.test_loader, metrics, to_visualize=True)
        metrics.compute_metrics(test_loss = test_loss, mode="test")
        ExperimentLogger.log_metrics(self.config, metrics.metrics)
        plot_loss_curves(self.config)
