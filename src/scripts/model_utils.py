import torch
import os
import importlib


class ModelManager:

    @staticmethod
    def save_checkpoint(model, config):
        checkpoint_path = os.path.join(config['paths']['results_dir'], config['experiment_name'], "checkpoint.pth")
        torch.save(model.state_dict(), checkpoint_path)
        return checkpoint_path

    @staticmethod
    def load_checkpoint(model, config):
        checkpoint_path = os.path.join(config['paths']['results_dir'], config['experiment_name'], "checkpoint.pth")
        try:
            model.load_state_dict(torch.load(checkpoint_path))
        except FileNotFoundError:
            print(f"Checkpoint not found at {checkpoint_path}")
        return model

    @staticmethod
    def load_model(config):
        model_filename = config['model']['filename']
        model_dir = config['paths']['model_dir']
        for filename in os.listdir(model_dir):
            if filename == model_filename:
                module_name = f"src.models.{os.path.splitext(filename)[0]}"  # remove .py
                module = importlib.import_module(module_name)
                class_name = config['model']['class_name']
                model_class = getattr(module, class_name)
                return model_class(config)
