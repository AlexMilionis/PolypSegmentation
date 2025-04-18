import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.models.model_utils import ModelManager
from tqdm import tqdm
# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler
from scripts.experiments.experiment_utils import ExperimentLogger
from torch import nn, optim
from torch.optim import lr_scheduler
from scripts.experiments.trainer import Trainer
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
        optimizer_type = getattr(optim, self.config['optimizer'])
        self.optimizer = optimizer_type(self.model.parameters(), lr=float(self.config['learning_rate']), weight_decay=self.config['weight_decay'])

        # scheduler_type = getattr(lr_scheduler, self.config['optimizer']['scheduler'])
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
                min_lr=1e-6
            )
        elif self.config["scheduler"] in [None, "None"]:
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.config['scheduler']}")

        self.trainer = Trainer(self.config, self.model, self.optimizer, self.criterion, self.scaler, self.device)


    def execute_training(self):
        with tqdm(range(self.num_epochs), desc="Training Epochs") as pbar:
            for epoch in pbar:
                if epoch==0:
                    metrics = Metrics(self.device, self.config)
                train_loss = self.trainer.train_one_epoch(self.train_loader)
                val_loss, metrics = self.trainer.validate_one_epoch(self.val_loader, metrics)
                metrics.compute_metrics(epoch = epoch+1, train_loss = train_loss, val_loss = val_loss)

                # Step the scheduler
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif isinstance(self.scheduler, CosineAnnealingLR):
                    self.scheduler.step()


        ModelManager.save_checkpoint(self.model, self.config)
        return metrics


    def execute_evaluation(self, metrics):

        test_loss, metrics = self.trainer.validate_one_epoch(self.test_loader, metrics, to_visualize=True)

        metrics.compute_metrics(test_loss = test_loss, mode="test")
        ExperimentLogger.log_metrics(self.config, metrics.metrics)

        plot_loss_curves(self.config)
