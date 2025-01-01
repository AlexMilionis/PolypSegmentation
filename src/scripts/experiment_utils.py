import csv
from src.config.constants import Constants
import os
import torch
import yaml

# class X:
#     def save_checkpoint():
#         pass
#     def load_checkpoint:
#         pass
#     def load_config:
#         pass

# class X:
#     def __init__(self):
#         pass
#     def _create_experiment_directory:
#         pass
#     def _init_metrics_csv:
#         pass
#     def log_metrics:
#         pass
#     def log_experiment:
#         pass

class ExperimentLogger:
    def __init__(self, experiment_name, metrics):
        self.experiment_name = experiment_name
        self.experiment_results_dir = os.path.join(Constants.RESULTS_DIR, self.experiment_name)
        self._create_experiment_directory()
        self.metrics_path = os.path.join(self.experiment_results_dir, "metrics.csv")
        self._init_metrics_csv(metrics)
        #   logs initialization
        self.logs_path = os.path.join(Constants.RESULTS_DIR, self.experiment_name, "logs.log")



    def _create_experiment_directory(self):
        #   delete directory files from previous experiment, if they exist
        if os.path.exists(self.experiment_results_dir):
            for filename in os.listdir(self.experiment_results_dir):
                file_path = os.path.join(self.experiment_results_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            #   delete empty directory from previous experiment
            os.rmdir(self.experiment_results_dir)
        #   create directory
        os.makedirs(self.experiment_results_dir, exist_ok=True)


    def _init_metrics_csv(self, metrics):
        with open(self.metrics_path, "w", newline='') as f:
            writer = csv.writer(f)
            cols = ["Epoch"]
            cols.extend(list(metrics.keys()))
            writer.writerow(cols)


    def log_metrics(self, epoch, metrics):
        # Append metrics to the log file
        with open(self.metrics_path, "a", newline='') as f:
            writer = csv.writer(f)
            cols = [epoch]
            cols.extend(list(metrics.values()))
            writer.writerow(cols)


    def log_experiment(self, details):
        # os.makedirs(self.logs_dir, exist_ok=True)

        #   log format
        """
        Experiment ID: experiment_1
        Model: UNet
        Encoder: resnet34
        Learning Rate: 0.0001
        Batch Size: 32
        Epochs: 20

        Epoch 1/20:
            Training Loss: 0.589
            Validation Loss: 0.612
            Dice Score: 0.71
            Time Taken: 45s

        Epoch 2/20:
            Training Loss: 0.421
            Validation Loss: 0.459
            Dice Score: 0.78
            Time Taken: 42s

        GPU Utilization: 75% average during training.

        Experiment Completed: 2024-12-18 14:23:15
        """

    @staticmethod
    def load_config(config_name):
        """
        Loads the experiment's configuration file from path.
        """
        config_name = config_name + '.yaml'
        try:
            config_path = os.path.join(Constants.CONFIG_DIR, config_name)
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print("Configuration not found.")
        # if config_name not in os.listdir(Constants.CONFIG_DIR):
        #     print("Config not found")
        #     return None
        # else:
        #     config_path = os.path.join(Constants.CONFIG_DIR, config_name)
        #     with open(config_path, 'r') as f:
        #         return yaml.safe_load(f)


