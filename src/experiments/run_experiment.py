import torch
from src.config.hyperparameters import Hyperparameters
from src.models.unet import UNet
from src.models.model_utils import ModelCheckpoint
import warnings
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from src.scripts.experiment_utils import ExperimentLogger
from src.scripts.metrics import Metrics
from torch import nn, optim
from torch.nn import BCEWithLogitsLoss

warnings.filterwarnings('ignore')


class Experiment:
    def __init__(self, train_loader, val_loader, config):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.scaler = GradScaler()  # mixed precision training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = None

        self.model = self._load_model()

        self.num_epochs = config['training']['epochs']
        self.criterion = getattr(nn, self.config['training']['criterion'])()
        optimizer_type = getattr(optim, self.config['training']['optimizer']['type'])
        self.optimizer = optimizer_type(self.model.parameters(),
                                        lr=self.config['training']['optimizer']['learning_rate'],
                                        )


    def _load_model(self):
        model = UNet(self.config).to(self.device)
        return model


    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for images, masks, _ in self.train_loader:
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
        return total_loss / len(self.train_loader)


    def _validate_one_epoch(self):
        self.model.eval()
        total_val_loss = 0
        val_metrics = Metrics()
        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_metrics.add_batch(preds, masks)
                total_val_loss += loss.item()
        return total_val_loss, val_metrics


    def train(self):
        with tqdm(range(self.num_epochs), desc="Training Epochs") as pbar:
            for epoch in pbar:
                torch.cuda.empty_cache()  # Clear GPU memory
                total_train_loss = self._train_one_epoch()
                # total_train_loss = Trainer.train_one_epoch(self)
                total_val_loss, val_metrics = self._validate_one_epoch()
                # total_val_loss, val_metrics = Trainer.validate_one_epoch(self)
                val_metrics_dict = val_metrics.compute_metrics(total_train_loss, len(self.train_loader), total_val_loss, len(self.val_loader))
                if epoch==0:
                    self.logger = ExperimentLogger(experiment_name=self.model.name, metrics=val_metrics_dict)
                self.logger.log_metrics(epoch=epoch, metrics=val_metrics_dict)
                pbar.set_postfix({"Train Loss": val_metrics_dict["Training Loss"],
                                  "Validation Loss": val_metrics_dict["Validation Loss"]})
        ModelCheckpoint.save(self.model, self.logger.experiment_results_dir)
