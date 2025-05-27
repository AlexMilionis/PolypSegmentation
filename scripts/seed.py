import torch
import random
import numpy as np
from torch import Generator


def set_seed(seed=42):
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For current GPU
    torch.cuda.manual_seed_all(seed)  # For all GPUs (if using multi-GPU)
    random.seed(seed)  # For Python's built-in random module
    np.random.seed(seed)  # For NumPy

    # Ensures deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_generator(seed=42):
    return Generator().manual_seed(seed)
