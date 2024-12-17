import torch
import os
from hyperparameters import Hyperparameters
from models.unet import UNet
from scripts.metrics import Metrics
import warnings
from torch.cuda.amp import autocast
from constants import Constants
from scripts.visualization_utils import visualize_outputs

warnings.filterwarnings('ignore')


class Evaluator:
    def __init__(self, loader: torch.utils.data.DataLoader, visualize_predictions: bool = False, num_samples: int = 5):
        self.loader = loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.visualize_predictions = visualize_predictions
        self.num_samples = num_samples
        self.model = self._load_model()
        self.criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy_with_logits']


    def _load_model(self):
        model = UNet().to(self.device)
        checkpoint_path = os.path.join(Constants.MODEL_CHECKPOINT_DIR, model.name + "_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        return model


    def _eval_loop(self):
        total_loss = 0
        eval_metrics = Metrics()
        with torch.no_grad():
            for batch_idx, (images, masks, _) in enumerate(self.loader):
                images, masks = images.to(self.device), masks.to(self.device)
                # Mixed precision inference
                with autocast():
                    outputs = self.model(images)  # Raw logits
                    loss = self.criterion(outputs, masks)  # BCE with logits

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                eval_metrics.add_batch(preds, masks)
                total_loss += loss.item()
                # visualizing the first batch
                if self.visualize_predictions and batch_idx == 0:
                    visualize_outputs(images, masks, preds, num_samples=self.num_samples)

        return total_loss, eval_metrics


    def _display_metrics(self, total_loss, eval_metrics):
        metrics = {"loss": total_loss / len(self.loader),
                   "recall": eval_metrics.recall(),
                   "precision": eval_metrics.precision(),
                   "specificity": eval_metrics.specificity(),
                   "dice score": eval_metrics.dice_score(),
                   "jaccard index": eval_metrics.jaccard_index()}
        print(metrics)


    def evaluate(self):
        total_loss, eval_metrics = self._eval_loop()
        self._display_metrics(total_loss, eval_metrics)
