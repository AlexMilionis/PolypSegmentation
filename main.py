from scripts.data.dataloader_new import DataLoading
from scripts.seed import set_seed
from scripts.experiments.run_experiment import Experiment
from scripts.experiments.experiment_utils import ExperimentLogger
from scripts.visualizations.visualization_utils import visualize_data
from scripts.data.get_train_mean_std import get_train_mean_std
import sys, yaml
import warnings
warnings.filterwarnings("ignore")


set_seed()
if __name__ == '__main__':

    config = ExperimentLogger.parse_arguments()

    train_loader, val_loader, test_loader = DataLoading(config).get_dataloaders()
    # train_loader, val_loader, test_loader = DataLoadingCloud(config).get_loaders(viz=True)

    visualize_data(config, train_loader, num_samples=5)
    
    exp = Experiment(train_loader, val_loader, test_loader, config)

    metrics = exp.execute_training()

    exp.execute_evaluation(metrics)

