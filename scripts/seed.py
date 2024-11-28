import torch
import random
import numpy as np
from torch import Generator


def set_seed(seed=42):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For current GPU
    torch.cuda.manual_seed_all(seed)  # For all GPUs (if using multi-GPU)
    random.seed(seed)  # For Python's built-in random module
    np.random.seed(seed)  # For NumPy

    # Ensures deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print(f"Global seed set to: {seed}")


def worker_init_fn(worker_id):
    """
    Initialize the seed for each worker.

    Args:
        worker_id (int): Worker ID provided by the DataLoader.
    """
    seed = torch.initial_seed() % (2 ** 32)  # Derive seed from PyTorch initial seed
    np.random.seed(seed)
    random.seed(seed)


def set_generator(seed=42):
    return Generator().manual_seed(seed)
