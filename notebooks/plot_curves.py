import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss_curves(csv_path, save_path=None):
    """
    Creates training/validation error plots from experiment CSV data.

    Args:
        csv_path (str): Path to CSV file with training metrics
        save_path (str): Optional path to save the plot image
    """
    # Load data and handle -1 values
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

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# Usage example:
# plot_error_curves("results.csv", save_path="error_plot.png")