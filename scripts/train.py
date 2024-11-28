import torch
from torch import optim
from hyperparameters import Hyperparameters
from models.unet import UNet

def train_model(train_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    # train_loader = DataLoading(mode="train").get_loader()
    model = UNet()
    model = model.to(device)
    criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy']
    optimizer = optim.Adam(model.parameters(), lr=Hyperparameters.LEARNING_RATE)

    num_epochs = 1 #Hyperparameters.EPOCHS
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