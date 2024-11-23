import os.path

import matplotlib.pyplot as plt
import torch


# def visualize_batch_from_loader(dataloader):
#
#     image, mask, path = next(iter(dataloader))
#     print(path)
#     # image_tensor = image[0].permute(2, 1, 0).numpy()
#     image_tensor = image[0].permute(1,2,0).numpy()
#
#     plt.imshow(image_tensor)
#     plt.title(os.path.basename(path[0][0]))
#     plt.axis("off")
#     plt.show()



def visualize_batch_from_loader(dataloader):
    """
    Visualizes the first image and its corresponding mask from a DataLoader side by side.

    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader object.
    """
    # Fetch the first batch from the dataloader
    dl = iter(dataloader)
    for _ in range(10):
        image_batch, mask_batch, paths_batch = next(dl)

    # Extract the Nth image, mask, and paths
    N = -1
    image = image_batch[N].permute(1, 2, 0).numpy()  # Convert to (H, W, C)
    mask = mask_batch[N].squeeze().numpy()           # Convert to (H, W)

    img_path, mask_path = paths_batch[0][N], paths_batch[1][N]

    # Create a figure for side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the image
    axes[0].imshow(image)
    axes[0].set_title(f"Image: {os.path.basename(img_path)}")
    axes[0].axis("off")

    # Plot the mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Mask: {os.path.basename(mask_path)}")
    axes[1].axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()
