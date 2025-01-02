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

    def _recall(self):
        return self.true_positives / (self.true_positives + self.false_negatives)

    def _precision(self):
        return self.true_positives / (self.true_positives + self.false_positives)

    def _specificity(self):
        return self.true_negatives / (self.true_negatives + self.false_positives)

    def _f1_score(self):
        return 2*(self._precision() * self._recall())/(self._precision() + self._recall())

    def _dice_score(self):
        return 2*self.true_positives / (2*self.true_positives + self.false_positives + self.false_negatives)

    def _jaccard_index(self):
        return self.true_positives / (self.true_positives + self.false_positives + self.false_negatives)

    def compute_metrics(self, test_mode, **kwargs):
        metrics_config = kwargs.get("config_metrics", [])
        metrics = {}
        if test_mode:
            metrics['TestLoss'] = kwargs.get("test_loss", 0) / kwargs.get("len_test_loader", 1)
        else:
            metrics["TrainLoss"] = kwargs.get("train_loss", 0) / kwargs.get("len_train_loader", 1)
            metrics["ValLoss"] = kwargs.get("val_loss", 0) / kwargs.get("len_val_loader", 1)

        for metric_name in metrics_config:
            metric_name = '_'+ metric_name
            if hasattr(self,  metric_name):
                metric_method = getattr(self, metric_name)
                metrics[metric_name[1:].title().replace('_','')] = metric_method()
        return metrics


