import torch

class Metrics():
    def __init__(self, device, config):
        self.device = device
        self.metric_names = config['metrics']
        self.tp = torch.tensor(0, dtype=torch.long, device=self.device)
        self.fn = torch.tensor(0, dtype=torch.long, device=self.device)
        self.fp = torch.tensor(0, dtype=torch.long, device=self.device)
        self.tn = torch.tensor(0, dtype=torch.long, device=self.device)
        # initialize metrics dict
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

    def mDice(self):
        # return 2*(self._precision() * self._recall())/(self._precision() + self._recall())
        return 2*self.tp / (2 * self.tp + self.fp + self.fn) # DEN EINAI MEAN

    def f2_score(self):
        return 5*self.tp / (5 * self.tp + self.fp + 4 * self.fn)

    def mIoU(self):
        return self.tp / (self.tp + self.fp + self.fn)  # DEN EINAI MEAN

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

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


        for metric in self.metric_names:
            if mode=="train":
                self.metrics[metric].append(getattr(self, metric)())
            if mode=="test":
                self.metrics[metric].append(getattr(self, metric)())


