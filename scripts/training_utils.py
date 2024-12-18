import csv
from constants import Constants
import os
import torch


def save_model(model):
    os.makedirs(Constants.MODEL_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(Constants.MODEL_CHECKPOINT_DIR, model.name + "_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


class TrainLogger:
    def __init__(self, model_name, metrics, log_dir="logs"):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, f"{model_name}_train_log.csv")
        self._init_logs(metrics)


    def _init_logs(self, metrics):
        # Create directory and log file with headers if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_path, "w", newline='') as f:
            writer = csv.writer(f)
            cols = ["Epoch"]
            cols.extend(list(metrics.keys()))
            writer.writerow(cols)


    def log_epoch_metrics(self, epoch, metrics):
        # Append metrics to the log file
        with open(self.log_path, "a", newline='') as f:
            writer = csv.writer(f)
            cols = [epoch]
            cols.extend(list(metrics.values()))
            writer.writerow(cols)
