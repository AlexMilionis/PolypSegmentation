from scripts.data.dataloader import DataLoadingLocal, DataLoadingCloud
from scripts.seed import set_seed
from scripts.experiments.run_experiment import Experiment
from scripts.experiments.experiment_utils import ExperimentLogger
import sys, yaml


set_seed()
if __name__ == '__main__':
    # python main.py <experiment_name>

    args = ExperimentLogger.parse_arguments()
    if args.config:
        config = ExperimentLogger.load_config(args.config)
    elif args.dict:
        config = yaml.safe_load(args.dict)
    else:
        raise ValueError("Either --config or --dict must be provided.")

    # if len(sys.argv) > 1: config_name = sys.argv[1]
    # config = ExperimentLogger.load_config(config_name)

    # train_loader, val_loader, test_loader = DataLoadingLocal(config).get_loaders(viz=True)
    train_loader, val_loader, test_loader = DataLoadingCloud(config).get_loaders(viz=True)

    exp = Experiment(train_loader, val_loader, test_loader, config)

    metrics = exp.execute_training()

    exp.execute_evaluation(metrics)

