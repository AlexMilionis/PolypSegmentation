import torch
import os


class ModelCheckpoint:

    @staticmethod
    def save(model, experiment_results_dir):
        """
        Saves the model's state dictionary to a checkpoint file.
        """
        #   TODO: fix next line
        checkpoint_path = os.path.join(experiment_results_dir, "checkpoint.pth")
        # if os.path.exists(checkpoint_path):
        #     os.remove(checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)
        return checkpoint_path

    @staticmethod
    def load(model, experiment_results_dir):
        """
        Loads the model's state dictionary from a checkpoint file.
        """
        #   TODO: fix next line
        checkpoint_path = os.path.join(experiment_results_dir, "checkpoint.pth")
        try:
            model.load_state_dict(torch.load(checkpoint_path))
        except FileNotFoundError:
            print(f"Checkpoint not found at {checkpoint_path}")
        return model