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
        self.exp_res_dir = os.path.join(Constants.RESULTS_DIR, self.experiment_name)
        self._create_experiment_directory()

        self.metrics_path = os.path.join(self.exp_res_dir, "metrics.csv")
        self._init_metrics_csv(metrics)

        #   logs initialization
        self.logs_path = os.path.join(Constants.RESULTS_DIR, self.experiment_name, "logs.log")
        self.config_dir = os.path.join(Constants.CONFIG_DIR, self.experiment_name)

    def _create_experiment_directory(self):
        #   delete directory files from previous experiment, if they exist
        if os.path.exists(self.exp_res_dir):
            for filename in os.listdir(self.exp_res_dir):
                file_path = os.path.join(self.exp_res_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            #   delete empty directory from previous experiment
            os.rmdir(self.exp_res_dir)
        #   create directory
        os.makedirs(self.exp_res_dir, exist_ok=True)

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


    def save_checkpoint(self, model):
        # os.makedirs(Constants.RESULTS_DIR ,exist_ok=True)
        # checkpoint_path = os.path.join(Constants.MODEL_CHECKPOINT_DIR, model.name + "_checkpoint.pth")

        checkpoint_path = os.path.join(self.exp_res_dir, "checkpoint.pth")
        # if os.path.exists(checkpoint_path):
        #     os.remove(checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)
        return checkpoint_path


    def load_checkpoint(self, model):
        """
        Loads the model's state dictionary from a checkpoint file.
        """
        checkpoint_path = os.path.join(self.exp_res_dir, "checkpoint.pth")
        try:
            model.load_state_dict(torch.load(checkpoint_path))
        except FileNotFoundError:
            print(f"Checkpoint not found at {checkpoint_path}")
        return model


    def load_config(self, config_name):
        """
        Loads the experiment's configuration file from path.
        """
        config_path = os.path.join(self.config_dir, config_name)
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)