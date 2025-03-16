from scripts.data.dataloader import DataLoadingLocal, DataLoadingCloud
from scripts.seed import set_seed
from scripts.visualizations.visualization_utils import visualize_data, plot_loss_curves
from scripts.experiments.run_experiment import Experiment
from scripts.experiments.experiment_utils import ExperimentLogger
import sys


set_seed()
if __name__ == '__main__':
    # python main.py <experiment_name>
    if len(sys.argv) > 1: config_name = sys.argv[1]

    config = ExperimentLogger.load_config(config_name)
    #
    train_loader, val_loader, test_loader = DataLoadingLocal(config).get_loaders()

    visualize_data(config, train_loader, num_samples = 3)

    exp = Experiment(train_loader, val_loader, test_loader, config)

    metrics = exp.execute_training()

    exp.execute_evaluation(metrics)

    plot_loss_curves(config)