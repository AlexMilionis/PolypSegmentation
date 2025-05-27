# import torch
# from torch.utils.data import DataLoader, Subset
# from scripts.data.dataset import PolypDataset
# from scripts.constants import Constants

# from scripts.experiments.experiment_utils import ExperimentLogger
# from scripts.experiments.run_experiment import Experiment
# import warnings
# from scripts.seed import set_seed
# from scripts.visualizations.visualization_utils import visualize_data

# warnings.filterwarnings("ignore")

# def run_overfit_test(num_samples=10):
#     """
#     Run an overfitting test using a small subset of data.
    
#     Args:
#         num_samples: Number of samples to use for overfitting
#         epochs: Number of epochs to train
#     """
    
#     # 1. Create the dataset with just a few samples
#     full_dataset = PolypDataset(
#         images_dir=Constants.TRAIN_VAL_IMAGES_DIR,
#         masks_dir=Constants.TRAIN_VAL_MASKS_DIR,
#         # mode="train",
#         mode="val",  # Use "val" mode to get deterministic transforms
#         preload=False
#     )
    
#     # 2. Select just the first few samples
#     if num_samples > len(full_dataset):
#         num_samples = len(full_dataset)
#         print(f"Warning: Requested more samples than available. Using {num_samples} samples.")
    
#     # Get indices of the samples we want to use
#     indices = list(range(num_samples))
#     overfit_dataset = Subset(full_dataset, indices)

#     print(overfit_dataset)
    
#     # 3. Create dataloaders - use the SAME data for both training and testing
#     overfit_loader = DataLoader(
#         overfit_dataset,
#         batch_size=4,  # Use small batch size
#         shuffle=True,  # Still shuffle during training for better convergence
#         num_workers=4,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         overfit_dataset,
#         batch_size=4,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # Same data, but no shuffling for testing
#     test_loader = DataLoader(
#         overfit_dataset,
#         batch_size=4,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )

#     return overfit_loader, val_loader, test_loader
    
# set_seed()
# if __name__ == "__main__":

#     config = ExperimentLogger.parse_arguments()

#     overfit_loader, val_loader, test_loader = run_overfit_test(num_samples=3)

#     # print the overfit_loader batch
#     for img, mask, paths in overfit_loader:
#         print(f"images shape: {img.shape}")
#         print(f"masks shape: {mask.shape}")     
#         # print the different pixel values in the mask
#         print(f"masks unique values: {torch.unique(mask)}")
        

#     visualize_data(config, overfit_loader, num_samples=3)

#     exp = Experiment(overfit_loader, val_loader, test_loader, config)

#     metrics = exp.execute_training()

#     exp.execute_evaluation(metrics)

