from torch.utils.data import random_split
from scripts.hyperparameters import Hyperparameters

def train_test(dataset):
    print(dataset.__len__())
    train_size = int(Hyperparameters.TRAIN_PCT * dataset.__len__())
    print(train_size)
