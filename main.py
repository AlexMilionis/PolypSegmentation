from scripts.data.data_module import DataLoading
from scripts.seed import set_seed
from scripts.experiments.run_experiment import Experiment
from scripts.experiments.experiment_utils import ExperimentLogger
from scripts.visualizations.visualization_utils import visualize_data
import warnings
warnings.filterwarnings("ignore")

set_seed()
if __name__ == '__main__':

    config = ExperimentLogger.load_config()

    dl = DataLoading(config, mode="overfit")

    dl.build_loaders()
    
    # visualize_data(config, dl.train_loader, num_samples=5)
    
    exp = Experiment(dl.train_loader, dl.val_loader, dl.test_loader, config)

    exp.execute_training()

    exp.execute_evaluation()

