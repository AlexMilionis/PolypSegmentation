from src.data.dataloader import DataLoading
from src.config.seed import set_seed
from src.scripts.visualization_utils import visualize_data
from src.experiments.run_experiment import Experiment
from src.scripts.experiment_utils import ExperimentLogger
import sys

set_seed()
if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_name = sys.argv[1]

    config = ExperimentLogger.load_config(config_name)
    train_loader, val_loader, test_loader = DataLoading(config).get_loaders()
    visualize_data(config, train_loader, num_samples = 3)
    exp = Experiment(train_loader, val_loader, test_loader, config)
    exp.execute_training()
    exp.execute_evaluation()

