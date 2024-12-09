import torch
import time
import os
from hyperparameters import Hyperparameters
from models.unet import UNet
from scripts.visualization_utils import visualize_predictions
import warnings
from torch.cuda.amp import autocast
from tqdm import tqdm
from constants import Constants

warnings.filterwarnings('ignore')


class Evaluator:
    def __init__(self, loader, visualize_results = True, num_samples=3):
        self.loader = loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = UNet()
        self.model_checkpoint_path = os.path.join(Constants.MODEL_CHECKPOINT_DIR, self.model.name + "_checkpoint.pth")
        self.model.load_state_dict(torch.load(self.model_checkpoint_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy_with_logits']
        self.visualize_results = visualize_results
        self.num_samples = num_samples

    def _eval_loop(self):
        total_loss = 0
        total_time = 0

        # # Storage for visualization
        # input_images, ground_truths, predictions = [], [], []

        batch_bar = tqdm(self.loader, desc="Evaluating Batches", total=len(self.loader))
        with torch.no_grad():
            # with batch_bar:
            for batch_idx, (images, masks, _) in enumerate(batch_bar):
                start_time = time.time()
                images, masks = images.to(self.device), masks.to(self.device)

                # Mixed precision inference
                with autocast():
                    outputs = self.model(images)  # Raw logits
                    loss = self.criterion(outputs, masks)  # BCE with logits
                    print(f"outputs shape: {outputs.shape}")

                total_loss += loss.item()
                total_time += time.time() - start_time

                if self.visualize_results:
                    self.visualizer(images, masks, outputs)

                # Update postfix dynamically for the progress bar
                avg_loss = total_loss / (batch_idx + 1)

                batch_bar.set_postfix({"Average Loss": f"{avg_loss:.4f}"})
        batch_bar.close()

        # if self.visualize_results:
        #     return visualize_predictions(input_images, ground_truths, predictions, self.num_samples)


    def visualizer(self, images, masks, outputs):
        # Storage for visualization
        input_images, ground_truths, predictions = [], [], []
        batch_idx = 1
        # Collect samples for visualization if required
        if self.visualize_results and batch_idx == 1 and len(input_images) < self.num_samples:
            for i in range(min(self.num_samples - len(input_images), images.size(0))):
                input_images.append(images[i].cpu())
                ground_truths.append(masks[i].cpu())
                predictions.append(torch.sigmoid(outputs[i].cpu()))  # Apply sigmoid for probabilities


    def evaluate(self):
        return self._eval_loop()

