
import os
import matplotlib.pyplot as plt
import numpy as np
from scripts.constants import Constants
import seaborn as sns
import pandas as pd


def plot_image(ax, image, title, cmap=None):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


def process_image(image_tensor):
    image = image_tensor.permute(1, 2, 0).numpy()  # Convert (C, H, W) -> (H, W, C)
    return unnormalize_image(image)


def unnormalize_image(image):
    mean = np.array(Constants.IMAGENET_COLOR_MEANS)
    std = np.array(Constants.IMAGENET_COLOR_STDS)
    # reverse normalization
    unnormalized_image = (image * std) + mean
    unnormalized_image = np.clip(unnormalized_image, a_min = 0, a_max = 1)
    return unnormalized_image


def visualize_data(config, dataloader, num_samples=3, outputs=False):
    batch_images, batch_masks, batch_paths = next(iter(dataloader))
    num_samples = min(num_samples, len(batch_images))
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))
    for i in range(num_samples):
        image = process_image(batch_images[i].cpu())
        mask = batch_masks[i].squeeze().cpu().numpy()  # Convert (1, H, W) -> (H, W)
        img_path, mask_path = batch_paths[0][i], batch_paths[1][i]
        # Plot the image and mask
        plot_image(axes[i][0], image, f"Image: {os.path.basename(img_path)}")
        plot_image(axes[i][1], mask, f"Mask: {os.path.basename(mask_path)}", cmap="gray")
    plt.tight_layout()
    save_path = os.path.join(config['paths']['data_visualizations_dir'], "visualizations.png")
    plt.savefig(save_path)
    plt.close(fig)


def visualize_outputs(config, batch_images, batch_masks, batch_predictions, batch_paths, num_samples=3):
    num_samples = min(num_samples, len(batch_images))  # Ensure we don't exceed batch size
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 6 * num_samples))
    for i in range(num_samples):
        image = process_image(batch_images[i].cpu())
        mask = batch_masks[i].squeeze().cpu().numpy()
        predicted_mask = (batch_predictions[i].squeeze().cpu().numpy() > 0.5).astype(int)
        img_path, mask_path = batch_paths[0][i], batch_paths[1][i]
        plot_image(axes[i][0], image, f"Image: {os.path.basename(img_path)}")
        plot_image(axes[i][1], mask, f"Mask: {os.path.basename(mask_path)}", cmap="gray")
        plot_image(axes[i][2], predicted_mask, "Predicted Mask", cmap="gray")
    plt.tight_layout()

    save_path = os.path.join(config['paths']['results_dir'], config['experiment_name'], "results_visualizations.png")
    plt.savefig(save_path)
    plt.close(fig)


def plot_loss_curves(config):
    """
    Creates training/validation error plots from experiment CSV data.

    Args:
        csv_path (str): Path to CSV file with training metrics
        save_path (str): Optional path to save the plot image
    """
    # Load data and handle -1 values
    csv_path = os.path.join(config['paths']['results_dir'], config['experiment_name'], "experiment_results.csv")
    df = pd.read_csv(csv_path)

    # Separate test results from epoch data
    test_results = df[df['epoch'] == -1].copy()
    epoch_data = df[df['epoch'] != -1].copy()

    # Clean data
    epoch_data.replace(-1, pd.NA, inplace=True)
    test_results.replace(-1, pd.NA, inplace=True)

    # Compute error (1 - accuracy)
    # epoch_data['train_error'] = 1 - epoch_data['accuracy']
    # epoch_data['val_error'] = 1 - epoch_data[
    #     'accuracy']  # Assuming validation accuracy is the same as training accuracy


    # Melt dataframe for seaborn
    plot_df = epoch_data.melt(id_vars='epoch',
                              value_vars=['train_loss', 'val_loss'],
                              var_name='loss_type',
                              value_name='loss_value')

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plot training/validation curves
    ax = sns.lineplot(data=plot_df, x='epoch', y='loss_value',
                      hue='loss_type', palette=['#1f77b4', '#ff7f0e'],
                      linewidth=2.5)

    # Add test error marker if available
    if not test_results.empty and not pd.isna(test_results['test_loss'].iloc[0]):
        test_epoch = epoch_data['epoch'].max() + 1  # Place test after last epoch
        ax.scatter(x=test_epoch, y=test_results['test_loss'].iloc[0],
                   color='#2ca02c', s=150, label='Test Loss', zorder=5,
                   edgecolors='black', linewidth=1.5)

    # Style plot
    plt.title("Training and Validation Loss", fontsize=14, pad=20)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(title='Loss Type', loc='upper right')

    # Custom x-axis ticks
    max_epoch = epoch_data['epoch'].max()
    xticks = list(range(0, max_epoch + 1, max(1, max_epoch // 10)))
    if not test_results.empty:
        xticks.append(test_epoch)
    plt.xticks(xticks)

    # Save
    save_path = os.path.join(config['paths']['results_dir'], config['experiment_name'], "loss_curves.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
