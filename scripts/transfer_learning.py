import torch
from torch import optim
from hyperparameters import Hyperparameters
from models.unet import UNet
import warnings
from constants import Constants
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, loader, transfer_learning=True):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        self.criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy_with_logits']
        self.optimizer = optim.Adam(self.model.parameters(), lr=Hyperparameters.LEARNING_RATE)
        self.transfer_learning =  transfer_learning
        if self.transfer_learning:
            # Freeze the encoder weights for transfer learning
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False

        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler()
        self.num_epochs = Hyperparameters.EPOCHS


    def _train_loop(self):
        epoch_bar = tqdm(range(self.num_epochs), desc="Training Epochs", total=self.num_epochs)
        # Progress bar for epochs
        with epoch_bar:
            for epoch in epoch_bar:
                self.model.train()
                total_loss = 0

                # Iterate through the DataLoader
                for images, masks, _ in self.loader:
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

                # Update epoch progress bar with average loss
                avg_loss = total_loss / len(self.loader)
                if epoch == self.num_epochs - 1:
                    epoch_bar.set_postfix({"Average Loss": f"{avg_loss:.4f}"})
            epoch_bar.close()


    def _save_model(self):
        #   Save the model checkpoint
        os.makedirs(Constants.MODEL_CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(Constants.MODEL_CHECKPOINT_DIR, self.model.name + "_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        torch.save(self.model.state_dict(), checkpoint_path)
        # tqdm.write(f"Model checkpoint saved to: {checkpoint_path}")
        return checkpoint_path


    def train(self):
        self._train_loop()
        return self._save_model()
