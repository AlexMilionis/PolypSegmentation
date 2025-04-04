from scripts.data.dataloader_new import DataLoading
from scripts.seed import set_seed
from scripts.experiments.run_experiment import Experiment
from scripts.experiments.experiment_utils import ExperimentLogger
import sys, yaml


set_seed()
if __name__ == '__main__':

    config = ExperimentLogger.parse_arguments()

    train_loader, val_loader, test_loader = DataLoading(config).get_dataloaders(viz=True)
    # train_loader, val_loader, test_loader = DataLoadingCloud(config).get_loaders(viz=True)

    exp = Experiment(train_loader, val_loader, test_loader, config)

    metrics = exp.execute_training()

    exp.execute_evaluation(metrics)
