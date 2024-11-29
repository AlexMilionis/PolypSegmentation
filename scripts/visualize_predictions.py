import matplotlib.pyplot as plt
import os
from data.scripts.visualization import unnormalize_image

def visualize_predictions(images, masks, predictions, num_samples):
    """
    Helper function to visualize predictions.

    Args:
        images (list): List of input images.
        masks (list): List of ground truth masks.
        predictions (list): List of predicted masks.
        num_samples (int): Number of samples to visualize.
    """
    # os.makedirs(output_dir, exist_ok=True)

    # Plot the image, ground truth mask, and predicted mask
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 6 * num_samples))

    # If only one sample, axes is not a 2D array
    if num_samples == 1:
        axes = [axes]  # Wrap it in a list for consistent indexing

    for i in range(min(num_samples, len(images))):
        image = images[i].permute(1, 2, 0).numpy()
        image = unnormalize_image(image)
        # image = images[i].permute(1, 2, 0).numpy()  # Convert (C, H, W) -> (H, W, C)
        mask = masks[i].squeeze().numpy()  # Convert (1, H, W) -> (H, W)
        predicted_mask = (predictions[i].squeeze().numpy() > 0.5).astype(int)  # Binarize predictions

        axes[i][0].imshow(image)
        axes[i][0].set_title("Input Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(mask, cmap="gray")
        axes[i][1].set_title("Ground Truth Mask")
        axes[i][1].axis("off")

        axes[i][2].imshow(predicted_mask, cmap="gray")
        axes[i][2].set_title("Predicted Mask")
        axes[i][2].axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()

        # # Save the plot
        # output_path = os.path.join(output_dir, f"sample_{i + 1}.png")
        # plt.savefig(output_path)
        # plt.close()
        # print(f"Saved visualization to {output_path}")