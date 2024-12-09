"""
Visualization Utilities for Dataset and Predictions:

1. **Purpose:**
   This script contains utility functions to visualize data samples and model predictions, aiding in debugging and understanding model behavior.

2. **Functions:**

   - `plot_image(ax, image, title, cmap=None)`:
     - Plots a single image on the specified axis with a title and optional colormap.

   - `process_image(image_tensor)`:
     - Converts a PyTorch image tensor to a format suitable for visualization.
     - Unnormalizes the image using the dataset's normalization parameters and rearranges it to (H, W, C).

   - `visualize_data(dataloader, num_samples=2)`:
     - Visualizes a specified number of random image-mask pairs from the 9th batch of the provided DataLoader.
     - Displays side-by-side images and their corresponding masks with proper titles.

   - `visualize_predictions(images, masks, predictions, num_samples)`:
     - Visualizes a specified number of samples showing input images, ground truth masks, and predicted masks.
     - Arranges the results in a grid with three columns: Input Image, Ground Truth Mask, and Predicted Mask.
"""


import os
import matplotlib.pyplot as plt
import torch
import random
from data.scripts.dataset_utils import unnormalize_image


def plot_image(ax, image, title, cmap=None):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


def process_image(image_tensor):
    image = image_tensor.permute(1, 2, 0).numpy()  # Convert (C, H, W) -> (H, W, C)
    return unnormalize_image(image)


def visualize_data(dataloader, num_samples=2):
    # Fetch the N=9th batch from the DataLoader
    dl = iter(dataloader)
    for _ in range(1):
        image_batch, mask_batch, paths_batch = next(dl)
    # Limit samples to the batch size if necessary
    num_samples = min(num_samples, len(image_batch))
    random_indices = random.sample(range(len(image_batch)), num_samples)
    # Create a figure for visualizing the images and masks
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))
    for i, idx in enumerate(random_indices):
        image = process_image(image_batch[idx])
        mask = mask_batch[idx].squeeze().numpy()  # Convert (1, H, W) -> (H, W)
        img_path, mask_path = paths_batch[0][idx], paths_batch[1][idx]
        # Plot the image and mask
        plot_image(axes[i][0], image, f"Image: {os.path.basename(img_path)}")
        plot_image(axes[i][1], mask, f"Mask: {os.path.basename(mask_path)}", cmap="gray")

    # Display the plots
    plt.tight_layout()
    plt.show()


def visualize_predictions(images, masks, predictions, num_samples):
    # Create a figure for visualizing the input images, masks, and predictions
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 6 * num_samples))
    for i in range(min(num_samples, len(images))):
        image = process_image(images[i])
        mask = masks[i].squeeze().numpy()  # Convert (1, H, W) -> (H, W)
        predicted_mask = (predictions[i].squeeze().numpy() > 0.5).astype(int)  # Binarize predictions
        # Plot the input image, ground truth mask, and predicted mask
        plot_image(axes[i][0], image, "Input Image")
        plot_image(axes[i][1], mask, "Ground Truth Mask", cmap="gray")
        plot_image(axes[i][2], predicted_mask, "Predicted Mask", cmap="gray")
    # Display the plots
    plt.tight_layout()
    plt.show()
