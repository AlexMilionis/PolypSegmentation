import os, shutil
import yaml
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import pandas as pd
import torch
import sys

class ExperimentLogger:

    @staticmethod
    def log_metrics(config, metrics):
        
        experiment_results_dir = os.path.join(config['paths']['results_dir'], config['experiment_name'])
        os.makedirs(experiment_results_dir, exist_ok=True)  # Ensure directory exists
        csv_path = os.path.join(experiment_results_dir, 'experiment_results.csv')

        converted_metrics = []

        for metric in metrics:
            converted_metric = {
                'epoch': metric['epoch'],
                'train_loss': metric['train_loss'].item() if isinstance(metric['train_loss'], torch.Tensor) else metric['train_loss'],
                'val_loss': metric['val_loss'].item(),
                'meanIoU': metric['meanIoU'][0].item(),
                'meanDice': metric['meanDice'][0].item(),
                'precision': metric['precision'][0][0].item(),
                'recall': metric['recall'][0][0].item(),
                'accuracy': metric['accuracy'][0][0].item()
            }
            converted_metrics.append(converted_metric)

        metrics_df = pd.DataFrame(converted_metrics)

        metrics_df.to_csv(csv_path, index=False)



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