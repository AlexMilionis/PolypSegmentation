import torch
from torch import optim
from hyperparameters import Hyperparameters
from models.unet import UNet
import warnings
from constants import Constants
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import time
from scripts.metrics import Metrics

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

    def _compute_metrics(self, total_loss, val_metrics):
        metrics = {"loss": total_loss / len(self.val_loader),
                   "recall": val_metrics.recall(),
                   "precision": val_metrics.precision(),
                   "specificity": val_metrics.specificity(),
                   "dice score": val_metrics.dice_score(),
                   "jaccard index": val_metrics.jaccard_index()}
        # print(f"Displaying Validation metrics: {metrics}")
        return metrics


    def _save_model(self):
        #   Save the model checkpoint
        os.makedirs(Constants.MODEL_CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(Constants.MODEL_CHECKPOINT_DIR, self.model.name + "_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path


    def train(self):
        # epoch_bar = tqdm(range(self.num_epochs), desc="Training Epochs", total=self.num_epochs)
        # with epoch_bar:
        #     for epoch in epoch_bar:
        #         torch.cuda.empty_cache()  # Clear GPU memory
        #         total_train_loss = self._train_loop()
        #         # Update epoch progress bar with average loss
        #         avg_train_loss = total_train_loss / len(self.train_loader)
        #         epoch_bar.set_postfix({"Average Loss": f"{avg_train_loss:.4f}"})
        #         total_val_loss, val_metrics = self._validation_loop()
        #         self._display_metrics(total_val_loss, val_metrics)
        for epoch in tqdm(range(self.num_epochs), desc="Training Epochs"):
            torch.cuda.empty_cache()  # Clear GPU memory
            total_train_loss = self._train_loop()
            total_val_loss, val_metrics = self._validation_loop()
            val_metrics = self._compute_metrics(total_val_loss, val_metrics)

            tqdm.write(f"\nEpoch {epoch + 1}:")
            tqdm.write(f"   Train Loss: {total_train_loss / len(self.train_loader):.4f}")
            tqdm.write(f"   Validation Metrics: {val_metrics}")


        self._save_model()
