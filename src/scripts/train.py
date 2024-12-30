import torch
from tqdm import tqdm
from src.scripts.metrics import Metrics
from src.scripts.train_utils import ExperimentLogger
from src.experiments.exp1.run_experiment import ExperimentImplementation
from torch.cuda.amp import autocast, GradScaler


class Trainer(ExperimentImplementation):
    def __init__(self, train_loader, val_loader):
        super().__init__(train_loader, val_loader)


    def _train_one_epoch(self):
        pass

    def _validate_one_epoch(self):
        pass

    def train(self):
        pass

    #   save model