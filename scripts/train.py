import torch
from torch import optim
from hyperparameters import Hyperparameters
from models.unet import UNet
import time
import warnings
warnings.filterwarnings('ignore')

def train_model(train_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.is_available())
    # train_loader = DataLoading(mode="train").get_loader()
    model = UNet()
    model = model.to(device)
    criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy_with_logits']
    optimizer = optim.Adam(model.parameters(), lr=Hyperparameters.LEARNING_RATE)

    num_epochs = Hyperparameters.EPOCHS
    for epoch in range(num_epochs):
        start_epoch_time = time.time()  # Time the epoch
        model.train()
        total_loss = 0
        batch_num = 0
        for images, masks, _ in train_loader:
            start_batch_time = time.time()
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)#.squeeze(1)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss = loss.item()
            batch_num += 1
            batch_time = time.time() - start_batch_time
            print(f"Batch {batch_num}/{len(train_loader)}, Time Taken: {batch_time:.2f}s")

            # if batch_num > 2: break

        epoch_time = time.time() - start_epoch_time  # Calculate elapsed time for epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Time Taken: {epoch_time:.2f}s")


    torch.save(model.state_dict(), "D:/repos/MScThesis/github_repo/models/model_checkpoint.pth")

# import torch
# from torch import optim
# from hyperparameters import Hyperparameters
# from models.unet import UNet
# import time
#
#
# def train_model(train_loader):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     model = UNet()
#     model = model.to(device)
#
#     criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy_with_logits']
#     optimizer = optim.Adam(model.parameters(), lr=Hyperparameters.LEARNING_RATE)
#
#     # Initialize GradScaler for mixed precision
#     scaler = torch.cuda.amp.GradScaler()
#
#     num_epochs = Hyperparameters.EPOCHS
#     for epoch in range(num_epochs):
#         start_time = time.time()  # Time the epoch
#         model.train()
#         total_loss = 0
#
#         for images, masks, _ in train_loader:
#             images, masks = images.to(device), masks.to(device)
#
#             optimizer.zero_grad()
#
#             # Forward pass with mixed precision
#             with torch.cuda.amp.autocast():
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)
#
#             # Backward pass with scaled gradients
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#             total_loss += loss.item()
#
#         epoch_time = time.time() - start_time  # Calculate elapsed time for epoch
#         avg_loss = total_loss / len(train_loader)
#
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
#
#     # Save the model
#     torch.save(model.state_dict(), "D:/repos/MScThesis/github_repo/models/model_checkpoint.pth")
