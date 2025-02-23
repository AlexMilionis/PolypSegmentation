import torch

class Metrics():
    def __init__(self, device, config):
        self.metric_names = config['metrics']
        self.tp = torch.tensor(0, dtype=torch.long, device=device)
        self.fn = torch.tensor(0, dtype=torch.long, device=device)
        self.fp = torch.tensor(0, dtype=torch.long, device=device)
        self.tn = torch.tensor(0, dtype=torch.long, device=device)
        # initialize metrics dict
        self._create_metrics()


    def _create_metrics(self):
        self.col_names = ["epoch", "train_loss", "val_loss", "test_loss"] + self.metric_names
        self.metrics = {name:[] for name in self.col_names}


    def add_batch(self, batch_predictions, batch_ground_truths):
        predictions = (batch_predictions > 0.5).long()
        ground_truths = batch_ground_truths.long()

        self.tp += torch.sum((predictions == 1) & (ground_truths == 1))
        self.fn += torch.sum((predictions == 0) & (ground_truths == 1))
        self.fp += torch.sum((predictions == 1) & (ground_truths == 0))
        self.tn += torch.sum((predictions == 0) & (ground_truths == 0))

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

    def average_hausdorff_distance(self):
        pass

    def average_surface_distance(self):
        pass

    def normalized_surface_distance(self):
        pass


    def compute_metrics(self, mode="train", **kwargs):
        if mode == "train":
            self.metrics["epoch"].append(kwargs.get("epoch", 0))
            self.metrics["train_loss"].append(kwargs.get("train_loss", 0))
            self.metrics["val_loss"].append(kwargs.get("val_loss", 0))
            self.metrics["test_loss"].append(-1)

        if mode == "test":
            self.metrics["epoch"].append(-1)
            self.metrics["train_loss"].append(-1)
            self.metrics["val_loss"].append(-1)
            self.metrics["test_loss"].append(kwargs.get("test_loss", 0))


        # self.metrics["jaccard_index"].append(self.jaccard_index())
        # self.metrics["f1_score"].append(self.f1_score())
        # self.metrics["f2_score"].append(self.f2_score())
        # self.metrics["f2_score"].append(self.f2_score())
        # self.metrics["precision"].append(self.precision())
        # self.metrics["recall"].append(self.recall())
        # self.metrics["specificity"].append(self.specificity())

        for metric in self.metric_names:
            self.metrics[metric].append(getattr(self, metric)())