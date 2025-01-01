import torch

class Metrics():
    def __init__(self):
        self.true_positives  = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.true_negatives  = 0

    def add_batch(self, batch_predictions, batch_ground_truths):
        predictions = (batch_predictions > 0.5).float()
        ground_truths = batch_ground_truths.float()
        # Calculate metrics efficiently
        self.true_positives  += torch.sum((predictions == 1) & (ground_truths == 1)).item()
        self.false_negatives += torch.sum((predictions == 0) & (ground_truths == 1)).item()
        self.false_positives += torch.sum((predictions == 1) & (ground_truths == 0)).item()
        self.true_negatives  += torch.sum((predictions == 0) & (ground_truths == 0)).item()

    def recall(self):
        return self.true_positives / (self.true_positives + self.false_negatives)

    def precision(self):
        return self.true_positives / (self.true_positives + self.false_positives)

    def specificity(self):
        return self.true_negatives / (self.true_negatives + self.false_positives)

    def f1(self):
        return 2*(self.precision() * self.recall())/(self.precision() + self.recall())

    def dice_score(self):
        return 2*self.true_positives / (2*self.true_positives + self.false_positives + self.false_negatives)

    def jaccard_index(self):
        return self.true_positives / (self.true_positives + self.false_positives + self.false_negatives)

    def compute_metrics(self, test_mode, total_train_loss, len_train_loader, total_val_loss, len_val_loader):
        metrics = {}
        if test_mode:
            metrics['Test Loss'] = None
        metrics = {
            "Training Loss": total_train_loss / len_train_loader,
            "Validation Loss": total_val_loss / len_val_loader,
            "Recall": self.recall(),
            "Precision": self.precision(),
            "Specificity": self.specificity(),
            "Dice Score": self.dice_score(),
            "Jaccard Index": self.jaccard_index()
        }
        return metrics


