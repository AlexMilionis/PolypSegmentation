import os.path
import matplotlib.pyplot as plt
import torch
import random


def visualize_samples_from_random_batch(dataloader, num_samples=2):
    """
    Visualizes a specified number of randomly selected images and their corresponding masks
    from a single batch in a DataLoader. Images and masks are displayed side-by-side in a
    grid with dimensions `num_samples x 2`.

    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader object containing the dataset.
        num_samples (int): Number of random samples to visualize from the batch.
                           Defaults to 2. If the number exceeds the batch size,
                           it is capped at the batch size.

    Visualization:
        - Each row of the grid corresponds to a randomly selected image-mask pair.
        - The first column contains the images.
        - The second column contains the corresponding masks.
        - Each subplot is titled with the respective file name of the image or mask.

    Returns:
        None. Displays the visualization as a matplotlib figure.
    """
    # Fetch the first batch from the dataloader
    dl = iter(dataloader)
    for _ in range(9):
        image_batch, mask_batch, paths_batch = next(dl)

    # Max number of samples becomes batch size
    if num_samples > len(image_batch):
        num_samples = len(image_batch)

    # Randomly select  indices from the batch
    batch_size = image_batch.shape[0]
    random_indices = random.sample(range(batch_size), num_samples)

    # Create a figure for num_samples x 2 visualization
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))

    # If only one sample, axes is not a 2D array
    if num_samples == 1:
        axes = [axes]  # Wrap it in a list for consistent indexing

    for i, idx in enumerate(random_indices):
        image = image_batch[idx].permute(1, 2, 0).numpy()  # Convert image to (H, W, C)
        mask = mask_batch[idx].squeeze().numpy()  # Convert mask to (H, W)

        img_path, mask_path = paths_batch[0][idx], paths_batch[1][idx]

        # Plot the image
        axes[i][0].imshow(image)
        axes[i][0].set_title(f"Image: {os.path.basename(img_path)}")
        axes[i][0].axis("off")

        # Plot the mask
        axes[i][1].imshow(mask, cmap="gray")
        axes[i][1].set_title(f"Mask: {os.path.basename(mask_path)}")
        axes[i][1].axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()

