import torch
from data.scripts.dataloader import DataLoading
from torch import optim
from scripts.hyperparameters import Hyperparameters
from models.unet import UNet

def train_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\
    train_loader = DataLoading(mode="train").get_loader()
    model = UNet(encoder_name="resnet18", encoder_weights="imagenet")
    model = model.to(device)
    criterion = None
    optimizer = optim.Adam(model.parameters(), lr=Hyperparameters.LEARNING_RATE)

    num_epochs = 10 #Hyperparameters.EPOCHS
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss = loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")