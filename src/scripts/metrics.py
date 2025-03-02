import torch
from torchmetrics.segmentation import HausdorffDistance
from torchmetrics.functional.segmentation.hausdorff_distance import hausdorff_distance

class Metrics():
    def __init__(self, device, config):
        self.device = device
        self.metric_names = config['metrics']
        self.tp = torch.tensor(0, dtype=torch.long, device=self.device)
        self.fn = torch.tensor(0, dtype=torch.long, device=self.device)
        self.fp = torch.tensor(0, dtype=torch.long, device=self.device)
        self.tn = torch.tensor(0, dtype=torch.long, device=self.device)


        # initialize metrics dict
        self._create_metrics()

        # # Store batch predictions and ground truths for Hausdorff Distance
        # self.predictions_list = []
        # self.ground_truths_list = []

    def _create_metrics(self):
        self.col_names = ["epoch", "train_loss", "val_loss", "test_loss"] + self.metric_names
        self.metrics = {name:[] for name in self.col_names}


    def add_batch(self, batch_predictions, batch_ground_truths):
        # predictions = (batch_predictions > 0.5).long()
        predictions = (batch_predictions > 0.5).long().squeeze(1)
        # ground_truths = batch_ground_truths.long()
        ground_truths = batch_ground_truths.long().squeeze(1)

        self.tp += torch.sum((predictions == 1) & (ground_truths == 1))
        self.fn += torch.sum((predictions == 0) & (ground_truths == 1))
        self.fp += torch.sum((predictions == 1) & (ground_truths == 0))
        self.tn += torch.sum((predictions == 0) & (ground_truths == 0))

        # # Store for Hausdorff Distance computation
        # self.predictions_list.append(predictions)
        # self.ground_truths_list.append(ground_truths)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def specificity(self):
        return self.tn / (self.tn + self.fp)

    def f1_score(self):
        # return 2*(self._precision() * self._recall())/(self._precision() + self._recall())
        return 2*self.tp / (2 * self.tp + self.fp + self.fn)

    def f2_score(self):
        return 5*self.tp / (5 * self.tp + self.fp + 4 * self.fn)

    def jaccard_index(self):
        return self.tp / (self.tp + self.fp + self.fn)

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    # def average_hausdorff_distance(self):
    #     if len(self.predictions_list) == 0:
    #         return torch.tensor(0.0, device=self.device)
    #
    #     preds = torch.cat(self.predictions_list, dim=0).cpu()
    #     targets = torch.cat(self.ground_truths_list, dim=0).cpu()
    #
    #     total_hd = 0.0
    #     num_samples = 0
    #
    #     for i in range(preds.shape[0]):
    #         p = preds[i].unsqueeze(0)  # Add batch dimension
    #         t = targets[i].unsqueeze(0)
    #
    #         # Check if either prediction or target is empty for foreground
    #         if p.sum() == 0 or t.sum() == 0:
    #             total_hd += float('inf')
    #             num_samples += 1
    #             continue
    #
    #         # Compute HD for the foreground class (class 1)
    #         hd = hausdorff_distance(p, t, num_classes=2, per_class=True)
    #         hd_foreground = hd[1].item()  # Assuming class 1 is foreground
    #         total_hd += hd_foreground
    #         num_samples += 1
    #
    #     if num_samples == 0:
    #         return torch.tensor(0.0, device=self.device)
    #
    #     avg_hd = total_hd / num_samples
    #     return torch.tensor(avg_hd, device=self.device)


    # def average_surface_distance(self):
    #     pass
    #
    # def normalized_surface_distance(self):
    #     pass

    def compute_metrics(self, mode="train", **kwargs):
        if mode == "train":
            self.metrics["epoch"].append(kwargs.get("epoch", 0))
            self.metrics["train_loss"].append(kwargs.get("train_loss", 0))
            self.metrics["val_loss"].append(kwargs.get("val_loss", 0))
            self.metrics["test_loss"].append(-1)
            # self.metrics["average_hausdorff_distance"].append(-1)

        if mode == "test":
            self.metrics["epoch"].append(-1)
            self.metrics["train_loss"].append(-1)
            self.metrics["val_loss"].append(-1)
            self.metrics["test_loss"].append(kwargs.get("test_loss", 0))


        for metric in self.metric_names:
            if mode=="train" and  metric != "average_hausdorff_distance":
                self.metrics[metric].append(getattr(self, metric)())
            if mode=="test":
                self.metrics[metric].append(getattr(self, metric)())

        # # Reset stored predictions after computation
        # self.predictions_list = []
        # self.ground_truths_list = []