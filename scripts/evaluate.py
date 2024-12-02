import torch
import time
from hyperparameters import Hyperparameters
from models.unet import UNet
from scripts.visualization_utils import visualize_predictions
import warnings
from torch.cuda.amp import autocast
from tqdm import tqdm
warnings.filterwarnings('ignore')


def evaluate_model(test_loader, model_checkpoint_path, visualize_results=False, num_samples=3, output_dir="visualizations"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = Hyperparameters.LOSS_FUNCTIONS['binary_crossentropy_with_logits']
    total_loss = 0
    total_time = 0

    # Storage for visualization
    input_images, ground_truths, predictions = [], [], []

    batch_bar = tqdm(test_loader, desc="Evaluation Batches", total=len(test_loader))
    with torch.no_grad():
        with batch_bar:
            for batch_idx, (images, masks, _) in enumerate(batch_bar):
                start_time = time.time()
                images, masks = images.to(device), masks.to(device)

                # Mixed precision inference
                with autocast():
                    outputs = model(images)  # Raw logits
                    loss = criterion(outputs, masks)  # BCE with logits

                total_loss += loss.item()
                total_time += time.time() - start_time

                # Collect samples for visualization if required
                if visualize_results and batch_idx == 9 and len(input_images) < num_samples:
                    for i in range(min(num_samples - len(input_images), images.size(0))):
                        input_images.append(images[i].cpu())
                        ground_truths.append(masks[i].cpu())
                        predictions.append(torch.sigmoid(outputs[i].cpu()))  # Apply sigmoid for probabilities

            avg_loss = total_loss / len(test_loader)
        batch_bar.set_postfix({"Average Loss": f"{avg_loss:.4f}"})

    if visualize_results:
        visualize_predictions(input_images, ground_truths, predictions, num_samples)
