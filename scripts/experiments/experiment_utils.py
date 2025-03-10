import os, shutil
import yaml
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import pandas as pd
import torch


class ExperimentLogger:

    @staticmethod
    # def convert_tensor(value):
    #     if isinstance(value, torch.Tensor):
    #         return value.detach().cpu().item()  # Move to CPU and convert to Python float
    #     elif isinstance(value, list):  # If it's a list, apply conversion recursively
    #         return [ExperimentLogger.convert_tensor(v) for v in value]
    #     return value  # Return as is if not a tensor

    def convert_tensor(value):
        """Ensures PyTorch tensors are converted to CPU floats before saving."""
        if isinstance(value, torch.Tensor):
            return value.cpu().item()  # Moves to CPU and extracts the numerical value
        return value


    # @staticmethod
    # def log_metrics(config, metrics):
    #     print(metrics)
    #     experiment_results_dir = os.path.join(config['paths']['results_dir'], config['experiment_name'])
    #     if os.path.exists(experiment_results_dir):
    #         shutil.rmtree(experiment_results_dir)  # Removes directory even if not empty
    #     os.makedirs(experiment_results_dir, exist_ok=True)  # Re-create clean directory
    #     csv_path = os.path.join(experiment_results_dir, 'experiment_results.csv')
    #     cleaned_metrics = {key: ExperimentLogger.convert_tensor(value) for key, value in metrics.items()}
    #     metrics_df = pd.DataFrame(cleaned_metrics)
    #     metrics_df.to_csv(csv_path, index=False)

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

        # Convert tensors to CPU floats before saving
        cleaned_metrics = {
            key: [ExperimentLogger.convert_tensor(value) for value in values]
            for key, values in metrics.items()
        }

        # 🔹 Convert dictionary to a DataFrame where each row is an epoch
        metrics_df = pd.DataFrame.from_dict(cleaned_metrics)

        # ✅ Write the entire DataFrame at once (ensuring proper row-wise format)
        metrics_df.to_csv(csv_path, index=False)

        print(f"Metrics logged to {csv_path}")

    #
    # def log_metrics(self, epoch, metrics, mode="train"):
    #     """
    #     Logs training or test metrics into the appropriate sheet in 'experiment_results.xlsx'.
    #     :param epoch: The current epoch (integer) or None for test metrics.
    #     :param metrics: A dictionary containing metric values.
    #     :param mode: "train" for training metrics, "test" for test metrics.
    #     """
    #
    #     # Ensure tensors remain on the GPU while extracting numerical values
    #     cleaned_metrics = {
    #         k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()
    #     }
    #     # Convert dictionary into DataFrame (single-row)
    #     df = pd.DataFrame([{"Epoch": epoch, **cleaned_metrics}])
    #     # Determine sheet name
    #     sheet_name = "Metrics" if mode == "train" else "Test Metrics"
    #     # Read current Excel file
    #     with pd.ExcelWriter(self.csv_path, mode="a", if_sheet_exists="replace", engine="openpyxl") as writer:
    #         existing_data = pd.read_excel(self.csv_path, sheet_name=sheet_name)
    #         updated_df = pd.concat([existing_data, df], ignore_index=True)  # Append new data to the existing DataFrame
    #         updated_df.to_excel(writer, sheet_name=sheet_name, index=False) # Write back the entire sheet
    #         # df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    #


    @staticmethod
    def load_config(config_name, config_dir="experiments/"):
        if not(config_name.endswith(".yaml")):
            config_name = config_name + '.yaml'
        try:
            config_path = os.path.join(config_dir, config_name)
            # print(config_path)
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print("Configuration not found.")
            exit()



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