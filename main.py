from src.data.dataloader import DataLoading
from src.config.seed import set_seed
from src.scripts.visualization_utils import visualize_data
from src.experiments.run_experiment import Experiment
from src.scripts.experiment_utils import ExperimentLogger
import sys
from src.data.dataset_utils import CreateDataset

set_seed()
if __name__ == '__main__':
    print('To run the script use: python main.py <experiment_name>\n')
    #   Receive experiment name from command prompt and load configuration
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    config = ExperimentLogger.load_config(config_name)
    #   create initial dataset (allimages, allmasks)
    CreateDataset.create_initial_dataset(config, include_seq_frames=False)
    #   create processed dataset (train, val, test)
    CreateDataset.create_processed_datasets(config)
    #    apply transformations to every dataset separately + create 3 separate data loaders
    train_loader, val_loader, test_loader = DataLoading(config).get_loaders()
    visualize_data(config, train_loader, num_samples = 3)
    exp = Experiment(train_loader, val_loader, test_loader, config)
    exp.execute_training()
    exp.execute_evaluation()

