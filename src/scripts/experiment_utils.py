import csv
from src.config.constants import Constants
import os
import yaml


class ExperimentLogger:

    def __init__(self, config, metrics):
        self.experiment_results_dir = os.path.join(config['paths']['results_dir'], config['experiment_name'])
        self._create_experiment_directory(self.experiment_results_dir)
        self.metrics_path = os.path.join(self.experiment_results_dir, "metrics.csv")
        self._init_metrics_csv(self.metrics_path, metrics)  # metrics initialization
        # self.logs_path = os.path.join(self.experiment_results_dir, "logs.log")    # logs initialization

    @staticmethod
    def _create_experiment_directory(experiment_results_dir):
        #   delete directory files from previous experiment, if they exist
        if os.path.exists(experiment_results_dir):
            for filename in os.listdir(experiment_results_dir):
                file_path = os.path.join(experiment_results_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(experiment_results_dir)    #   delete empty directory from previous experiment
        os.makedirs(experiment_results_dir, exist_ok=True)  #   create directory

    @staticmethod
    def _init_metrics_csv(metrics_path, metrics):
        with open(metrics_path, "w", newline='') as f:
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

    @staticmethod
    def log_test_metrics(config, metrics):
        metrics_path = os.path.join(config['paths']['results_dir'], config['experiment_name'], "metrics_test.csv")
        with open(metrics_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics.keys())
            writer.writerow(metrics.values())

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
        if not(config_name.endswith(".yaml")):
            config_name = config_name + '.yaml'
        try:
            config_path = os.path.join(Constants.CONFIG_DIR, config_name)
            # print(config_path)
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print("Configuration not found.")
            exit()