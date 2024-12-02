import torch
from torch import optim
from hyperparameters import Hyperparameters
from models.unet import UNet
import time
import warnings
from constants import Constants
import os
warnings.filterwarnings('ignore')

from torch.cuda.amp import autocast, GradScaler


def train_model(train_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model = model.to(device)
    # Define loss function and optimizer (for decoder only)
    criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy_with_logits']
    optimizer = optim.Adam(model.model.decoder.parameters(), lr=Hyperparameters.LEARNING_RATE)
    # Freeze the encoder weights for transfer learning
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    num_epochs = Hyperparameters.EPOCHS
    for epoch in range(num_epochs):
        start_epoch_time = time.time()
        model.train()
        total_loss = 0
        for batch_num, (images, masks, _) in enumerate(train_loader):
            start_batch_time = time.time()
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

                # Backward pass and optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            batch_time = time.time() - start_batch_time
            print(f"Batch {batch_num + 1}/{len(train_loader)}, Time Taken: {batch_time:.2f}s")
            break

        # Epoch time logging
        epoch_time = time.time() - start_epoch_time
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Time Taken: {epoch_time:.2f}s")

    # Save the model checkpoint
    os.makedirs(Constants.MODEL_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(Constants.MODEL_CHECKPOINT_DIR, model.name + "_checkpoint.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")
    return checkpoint_path