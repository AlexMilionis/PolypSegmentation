import torch
from torch import optim
from hyperparameters import Hyperparameters
from models.unet import UNet
import time
import warnings
from constants import Constants
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

def train_model(train_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    # Define loss function and optimizer (for decoder only)
    criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy_with_logits']
    optimizer = optim.Adam(model.model.decoder.parameters(), lr=Hyperparameters.LEARNING_RATE)
    # Freeze the encoder weights for transfer learning
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    num_epochs = Hyperparameters.EPOCHS
    epoch_bar = tqdm(range(num_epochs), desc="Training Epochs", total=num_epochs)

    # Progress bar for epochs
    with epoch_bar:
        for epoch in epoch_bar:
            model.train()
            total_loss = 0

            # Iterate through the DataLoader
            for images, masks, _ in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                # Mixed precision forward pass
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            # Update epoch progress bar with average loss
            avg_loss = total_loss / len(train_loader)
            if epoch==num_epochs-1:
                epoch_bar.set_postfix({"Average Loss": f"{avg_loss:.4f}"})

    # Save the model checkpoint
    os.makedirs(Constants.MODEL_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(Constants.MODEL_CHECKPOINT_DIR, model.name + "_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    torch.save(model.state_dict(), checkpoint_path)
    # tqdm.write(f"Model checkpoint saved to: {checkpoint_path}")
    return checkpoint_path