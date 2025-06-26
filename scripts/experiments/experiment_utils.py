import os, shutil
import yaml
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import pandas as pd
import torch
import sys

class ExperimentLogger:

    @staticmethod
    def convert_tensor(value):
        """Ensures PyTorch tensors are converted to CPU floats before saving."""
        if isinstance(value, torch.Tensor):
            return value.cpu().item()  # Moves to CPU and extracts the numerical value
        return value


    @staticmethod
    def log_metrics(config, metrics):
        """
        Logs metrics into a CSV file without overwriting previous values.

        :param config: Configuration dictionary containing experiment paths.
        :param metrics: Dictionary containing lists of metric values per epoch.
        """
        experiment_results_dir = os.path.join(config['paths']['results_dir'], config['experiment_name'])
        os.makedirs(experiment_results_dir, exist_ok=True)  # Ensure directory exists
        csv_path = os.path.join(experiment_results_dir, 'experiment_results.csv')

        # # Convert tensors to CPU floats before saving
        # cleaned_metrics = {
        #     key: [ExperimentLogger.convert_tensor(value) for value in values]
        #     for key, values in metrics.items()
        # }

        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.applymap(lambda x: x.cpu().item() if torch.is_tensor(x) and x.numel() == 1 else x)
        mask = metrics_df["epoch"].notna()
        metrics_df.loc[mask, "epoch"] = metrics_df.loc[mask, "epoch"].astype(int)
        metrics_df.to_csv(csv_path, index=False)

        # # 🔹 Convert dictionary to a DataFrame where each row is an epoch
        # metrics_df = pd.DataFrame.from_dict(cleaned_metrics)
        #
        # # Write the entire DataFrame at once (ensuring proper row-wise format)
        # metrics_df.to_csv(csv_path, index=False)
        #
        # # print(f"Metrics logged to {csv_path}")

    @staticmethod
    def load_config():
        if len(sys.argv) < 2:
            sys.exit(1)
        config_path = os.path.join("configurations", sys.argv[1])
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config


    # def use_profiler(self, trainer, train_loader, epoch):
    #     # log_dir = './log'
    #     epoch_log_dir = os.path.join(self.experiment_results_dir, "logs", f"epoch_{epoch}")
    #     os.makedirs(epoch_log_dir, exist_ok=True)
    #
    #     log_file_path = os.path.join(epoch_log_dir, "profiling.log")  # Store log file in the same directory
    #     trace_dir = epoch_log_dir  # TensorBoard traces also go in the same directory
    #
    #     # Start profiling for the current epoch
    #     with profile(
    #             activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #             on_trace_ready=tensorboard_trace_handler(trace_dir),
    #             record_shapes=True,
    #             profile_memory=True,
    #     ) as prof:
    #         total_train_loss = trainer.train_one_epoch(train_loader)
    #
    #     # Save profiling results to a .log file
    #     with open(log_file_path, 'w') as log_file:
    #         log_file.write(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    #
    #     return total_train_loss