import torch

class Metrics():
    def __init__(self, device):
        self.tp = torch.tensor(0, dtype=torch.long, device=device)
        self.fn = torch.tensor(0, dtype=torch.long, device=device)
        self.fp = torch.tensor(0, dtype=torch.long, device=device)
        self.tn = torch.tensor(0, dtype=torch.long, device=device)


    def add_batch(self, batch_predictions, batch_ground_truths):
        predictions = (batch_predictions > 0.5).long()
        ground_truths = batch_ground_truths.long()

        self.tp += torch.sum((predictions == 1) & (ground_truths == 1))
        self.fn += torch.sum((predictions == 0) & (ground_truths == 1))
        self.fp += torch.sum((predictions == 1) & (ground_truths == 0))
        self.tn += torch.sum((predictions == 0) & (ground_truths == 0))

    def _recall(self):
        return self.tp / (self.tp + self.fn)

    def _precision(self):
        return self.tp / (self.tp + self.fp)

    def _specificity(self):
        return self.tn / (self.tn + self.fp)

    def _f1_score(self):
        # return 2*(self._precision() * self._recall())/(self._precision() + self._recall())
        return 2*self.tp / (2 * self.tp + self.fp + self.fn)

    def _f2_score(self):
        return 5*self.tp / (5 * self.tp + self.fp + 4 * self.fn)

    def _jaccard_index(self):
        return self.tp / (self.tp + self.fp + self.fn)

    def _accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def _average_hausdorff_distance(self):
        pass

    def _average_surface_distance(self):
        pass

    def _normalized_surface_distance(self):
        pass

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


