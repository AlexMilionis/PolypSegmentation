from src.experiments.run_experiment import ExperimentImplementation


class Trainer(ExperimentImplementation):
    def __init__(self, train_loader, val_loader):
        super().__init__(train_loader, val_loader)


    def _train_one_epoch(self):
        pass

    def _validate_one_epoch(self):
        pass

    def train(self):
        pass

    #   save model