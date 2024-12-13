import torch
import os
from hyperparameters import Hyperparameters
from models.unet import UNet
from scripts.metrics import Metrics
import warnings
from torch.cuda.amp import autocast
from constants import Constants

warnings.filterwarnings('ignore')


class Evaluator:
    def __init__(self, loader, visualize_results = True, num_samples=3):
        self.loader = loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy_with_logits']
        self.visualize_results = visualize_results
        self.num_samples = num_samples

        self.model_checkpoint_path = os.path.join(Constants.MODEL_CHECKPOINT_DIR, self.model.name + "_checkpoint.pth")
        self.model.load_state_dict(torch.load(self.model_checkpoint_path, map_location=self.device))
        self.model.eval()

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

        metrics = {"loss": total_loss / len(self.loader),
                   "recall": eval_metrics.recall(),
                   "precision": eval_metrics.precision(),
                   "specificity": eval_metrics.specificity(),
                   "dice score": eval_metrics.dice_score(),
                   "jaccard index": eval_metrics.jaccard_index(),}
        print(metrics)


    def evaluate(self):
        self._eval_loop()
        if self.visualize_results:
            pass

