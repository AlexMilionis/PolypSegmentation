"""
Seed Initialization for Reproducibility:

1. **Purpose:**
   This script ensures reproducibility by setting global and worker-level random seeds for PyTorch, NumPy, and Python's random module. It also provides a deterministic behavior for PyTorch operations.

2. **Functions:**

   - `set_seed(seed=42)`:
     - Sets global seeds for PyTorch (CPU and GPU), Python's `random`, and NumPy to ensure consistent results across runs.
     - Ensures deterministic behavior in PyTorch by disabling CUDNN benchmarking and enabling deterministic mode.

   - `worker_init_fn(worker_id)`:
     - Initializes the seed for each DataLoader worker in PyTorch.
     - Ensures that each worker's seed is derived from the global PyTorch seed, providing reproducible behavior in multi-threaded data loading.

   - `set_generator(seed=42)`:
     - Returns a PyTorch `Generator` initialized with the specified seed.
     - Used to ensure deterministic splitting of datasets (e.g., in `random_split`).

3. **Usage:**
   - Call `set_seed()` at the start of your training script to globally set seeds.
   - Pass `worker_init_fn` to the `worker_init_fn` argument in PyTorch `DataLoader` to ensure reproducibility in data loading.
   - Use `set_generator()` when performing dataset splits or operations requiring a PyTorch `Generator`.
"""



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

#
# def worker_init_fn(worker_id):
#     seed = torch.initial_seed() % (2 ** 32)  # Derive seed from PyTorch initial seed
#     np.random.seed(seed)
#     random.seed(seed)


def set_generator(seed=42):
    return Generator().manual_seed(seed)
