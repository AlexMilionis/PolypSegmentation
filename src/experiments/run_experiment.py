import torch
from src.scripts.model_utils import ModelManager
import warnings
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from src.scripts.experiment_utils import ExperimentLogger
from torch import nn, optim
from src.scripts.trainer import Trainer
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

warnings.filterwarnings('ignore')


class Experiment:
    def __init__(self, train_loader, val_loader, test_loader, config):
        self.config = config

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.scaler = GradScaler()  # mixed precision training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = None

        self.model = ModelManager.load_model(self.config, self.device)
        self.num_epochs = config['epochs']
        self.criterion = getattr(nn, self.config['loss_function'])()
        optimizer_type = getattr(optim, self.config['optimizer']['type'])
        self.optimizer = optimizer_type(self.model.parameters(),
                                        lr=self.config['optimizer']['learning_rate'],
                                        )
        self.trainer = Trainer(self.config, self.model, self.optimizer, self.criterion, self.scaler, self.device)


    def execute_training(self):
        with tqdm(range(self.num_epochs), desc="Training Epochs") as pbar:
            for epoch in pbar:
                # torch.cuda.empty_cache()  # Clear GPU memory
                total_train_loss = ExperimentLogger.use_profiler(self.trainer, self.train_loader, epoch)
                # total_train_loss = self.trainer.train_one_epoch(self.train_loader)
                total_val_loss, val_metrics = self.trainer.validate_one_epoch(self.val_loader)
                val_metrics_dict = val_metrics.compute_metrics(
                    test_mode = False,
                    train_loss = total_train_loss,
                    len_train_loader = len(self.train_loader),
                    val_loss = total_val_loss,
                    len_val_loader = len(self.val_loader),
                    config_metrics = self.config['metrics']
                )
                if epoch==0:
                    self.logger = ExperimentLogger(self.config, metrics=val_metrics_dict)
                self.logger.log_metrics(epoch=epoch, metrics=val_metrics_dict)
                # pbar.set_postfix({"Train Loss": val_metrics_dict["TrainLoss"],
                #                   "Validation Loss": val_metrics_dict["ValLoss"]})
        ModelManager.save_checkpoint(self.model, self.config)


    def execute_evaluation(self):
        print('Evaluating...')
        total_test_loss, test_metrics = self.trainer.validate_one_epoch(self.test_loader, to_visualize=True)
        test_metrics_dict = test_metrics.compute_metrics(
            test_mode = True,
            test_loss = total_test_loss,
            len_test_loader = len(self.test_loader),
            config_metrics = self.config['metrics']
            )
        self.logger.log_test_metrics(self.config, test_metrics_dict)