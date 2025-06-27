import torch
import os
import importlib


class ModelManager:

    @staticmethod
    def save_checkpoint(model, config):
        save_dir  = os.path.join(config['paths']['results_dir'], config['experiment_name'])
        checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        # return checkpoint_path
    
    # @staticmethod
    # def save_model_checkpoint(model, config, metrics, epoch):
    #     # if epoch<100 -> don't save model, to avoid saving too many models
    #     # if epoch=100 -> save best model from 100 first epochs, to avoid early stopping with no saved model
    #     if epoch+1==100:
    #         ModelManager._save_checkpoint(model, config)
    #         print(f"Checkpoint saved at epoch {epoch+1}")
    #     # if epoch>100 -> save model if val_loss < min(val_loss)
    #     if epoch+1>100:
    #         if metrics.metrics["val_loss"][-1] < min(metrics.metrics["val_loss"][:-1]):
    #             ModelManager._save_checkpoint(model, config)
    #             print(f"Checkpoint saved at epoch {epoch+1}")

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
                module_name = f"scripts.models.{os.path.splitext(filename)[0]}"  # remove .py
                module = importlib.import_module(module_name)
                class_name = config['model']['class_name']
                model_class = getattr(module, class_name)
                return model_class(config)
