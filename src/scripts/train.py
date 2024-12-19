import torch
from tqdm import tqdm
from src.scripts.metrics import Metrics
from src.scripts.utils import ExperimentLogger

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
        pass

    def _train_one_epoch(self):
        pass

    def _validate_one_epoch(self):
        pass

    def train(self):
        pass

    #   save model