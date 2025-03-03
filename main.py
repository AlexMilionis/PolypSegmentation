from scripts.data.dataloader import DataLoading
from scripts.seed import set_seed
from scripts.visualizations.visualization_utils import visualize_data
from scripts.experiments.run_experiment import Experiment
from scripts.experiments.experiment_utils import ExperimentLogger
import sys

set_seed()
if __name__ == '__main__':
    # print('To run the script use: python main.py <experiment_name>\n')
    #   Receive experiment name from command prompt and load configuration
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    config = ExperimentLogger.load_config(config_name)
    # TODO:  in the 2 following lines: need to change the directory for initial dataset, raw dataset
    # CreateDataset.create_initial_dataset(config, include_seq_frames=False)  #   create initial dataset (allimages, allmasks)
    # CreateDataset.create_processed_datasets(config) #   create processed dataset (train, val, test)
    train_loader, val_loader, test_loader = DataLoading(config).get_loaders()   #    apply transformations to every dataset separately + create 3 separate data loaders
    visualize_data(config, train_loader, num_samples = 3)
    exp = Experiment(train_loader, val_loader, test_loader, config)
    metrics = exp.execute_training()
    exp.execute_evaluation(metrics)

