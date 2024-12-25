import torch
from torch import optim
from src.config.hyperparameters import Hyperparameters
from src.models.unet import UNet
import warnings
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from src.scripts.metrics import Metrics
from src.scripts.utils import ExperimentLogger

warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, train_loader, val_loader, transfer_learning=True):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._load_model()
        self.criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy_with_logits']
        self.optimizer = optim.Adam(self.model.parameters(), lr=Hyperparameters.LEARNING_RATE)
        self.scaler = GradScaler()  #   mixed precision training
        self.num_epochs = Hyperparameters.EPOCHS
        self.logger = None

        self.transfer_learning =  transfer_learning
        if self.transfer_learning:
            # Freeze the encoder weights for transfer learning
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False


    def _load_model(self):
        model = UNet().to(self.device)
        return model


    def _train_loop(self):
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


    def _validation_loop(self):
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
                total_train_loss = self._train_loop()
                total_val_loss, val_metrics = self._validation_loop()
                val_metrics_dict = val_metrics.compute_metrics(total_train_loss, len(self.train_loader), total_val_loss, len(self.val_loader))
                if epoch==0:
                    self.logger = ExperimentLogger(experiment_name=self.model.name, metrics=val_metrics_dict)
                self.logger.log_metrics(epoch=epoch, metrics=val_metrics_dict)
                pbar.set_postfix({"Train Loss": val_metrics_dict["Training Loss"],
                                  "Validation Loss": val_metrics_dict["Validation Loss"]})
        self.logger.save_checkpoint(self.model)
