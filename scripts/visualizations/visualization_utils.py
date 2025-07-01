import os
import matplotlib.pyplot as plt
import numpy as np
from scripts.constants import Constants
import seaborn as sns
import pandas as pd
import torch
from PIL import Image 


def plot_image(ax, image, title, cmap=None):
    # ax.imshow(image, cmap=cmap, interpolation='nearest')
    ax.imshow(image, cmap=cmap, interpolation='nearest', aspect='equal')
    ax.set_title(title)
    ax.axis("off")


def process_image(image_tensor):
    image = image_tensor.permute(1, 2, 0).numpy()  # Convert (C, H, W) -> (H, W, C)
    return unnormalize_image(image)


def unnormalize_image(image):
    mean = np.array(Constants.DATASET_MEANS)
    std = np.array(Constants.DATASET_STDS)

    unnormalized_image = (image * std) + mean
    
    unnormalized_image = np.clip(unnormalized_image, 0, 1)

    return unnormalized_image


def visualize_data(config, dataloader, num_samples=3):
    print(f"Visualizing {num_samples} samples from the dataset...")
    batch = next(iter(dataloader))
    # batch_images, batch_masks, batch_paths = batch
    batch_images, batch_masks, paths = batch
    num_samples = min(num_samples, len(batch_images))

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))
    for i in range(num_samples):
        # Process image tensor
        
        image = process_image(batch_images[i].cpu())
        # Process mask tensor
        mask = batch_masks[i].squeeze().cpu().numpy() # .astype(np.uint8)  #changed

        # img_path = os.path.basename(batch_paths[0][i])
        # mask_path = os.path.basename(batch_paths[1][i])
        img_path, mask_path = paths[0][i], paths[1][i]  
        # print(f"Image path: {img_path}, Mask path: {mask_path}")

        plot_image(axes[i][0], image, f"Image: {img_path}")
        plot_image(axes[i][1], mask, f"Mask: {mask_path}", cmap="gray")

    plt.tight_layout()
    save_path = os.path.join(config['paths']['data_visualizations_dir'], "input_data_visualizations.png")
    plt.savefig(save_path)
    plt.close(fig)



def visualize_outputs(config, batch_images, batch_masks, batch_predictions, batch_paths, num_samples=5):
    num_samples = min(num_samples, len(batch_images))
    
    # Calculate figure size to preserve 512x512 resolution per image
    dpi = 100  # Adjust DPI for your display/saving needs
    fig_width = 3 * 512 / dpi  # 3 columns (image, mask, prediction)
    fig_height = num_samples * 512 / dpi
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(fig_width, fig_height), dpi=dpi)
    
    for i in range(num_samples):
        # Process image and masks (your existing code)
        image = process_image(batch_images[i].cpu())
        mask = batch_masks[i].squeeze().cpu().numpy()
        predicted_mask = (batch_predictions[i] * 255).squeeze().cpu().numpy().astype(int)
        
        # Plot with NO interpolation and fixed aspect ratio
        plot_image(axes[i][0], image, f"Image: {os.path.basename(batch_paths[0][i])}")
        plot_image(axes[i][1], mask, f"Mask: {os.path.basename(batch_paths[1][i])}", cmap="gray")
        plot_image(axes[i][2], predicted_mask, "Predicted Mask", cmap="gray")
    
    plt.subplots_adjust(wspace=0, hspace=0)  # Remove spacing between subplots
    plt.tight_layout()
    
    os.makedirs(os.path.join(config['paths']['results_dir'], config['experiment_name']), exist_ok=True)
    save_path = os.path.join(config['paths']['results_dir'], config['experiment_name'], "results_visualizations.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Avoid padding
    plt.close(fig)


def plot_loss_curves(config):
    # Load data
    csv_path = os.path.join(config['paths']['results_dir'], config['experiment_name'], "experiment_results.csv")
    df = pd.read_csv(csv_path)

    # Separate test results from epoch data (filter for regular epochs)
    train_data = df[df['epoch'] != -1].copy()  # Select all rows except test row
    test_row = df[df['epoch'] == -1].copy()

    # Melt dataframe for seaborn
    plot_df = train_data.melt(id_vars='epoch',
                              value_vars=['train_loss', 'val_loss'],
                              var_name='loss_type',
                              value_name='loss_value')

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")

    # Plot training/validation curves
    ax = sns.lineplot(data=plot_df, x='epoch', y='loss_value',
                      hue='loss_type', palette=['#1f77b4', '#ff7f0e'],
                      linewidth=2.5)

    # Add test error marker if available
    if not test_row.empty and not pd.isna(test_row['val_loss'].iloc[0]):
        test_epoch = train_data['epoch'].max()  # Place test at the last epoch
        ax.scatter(x=test_epoch, y=test_row['val_loss'].iloc[0], color='black', label='Test Loss', marker='o')

    # Set y-axis limits based on data
    y_max = plot_df['loss_value'].max() * 1.1  # Add 10% padding
    ax.set_ylim(0, min(y_max, 2))  # Cap at 2 but adjust to data

    # Style plot
    plt.title("Training and Validation Loss", fontsize=14, pad=20)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(title='Loss Type', loc='upper right')

    # Custom x-axis ticks
    max_epoch = train_data['epoch'].max()
    xticks = [1] + [i for i in range(20, max_epoch + 1, 20)]

    if not test_row.empty:
        xticks.append(test_epoch)
    plt.xticks(xticks)

    # Save
    save_path = os.path.join(config['paths']['results_dir'], config['experiment_name'], "loss_curves.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()