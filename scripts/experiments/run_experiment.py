import torch
from scripts.models.model_utils import ModelManager
import warnings
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from scripts.experiments.experiment_utils import ExperimentLogger
from torch import nn, optim
from scripts.experiments.trainer import Trainer
from scripts.experiments.metrics import Metrics

warnings.filterwarnings('ignore')


class Experiment:
    def __init__(self, train_loader, val_loader, test_loader, config):
        self.config = config

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.scaler = GradScaler()  # mixed precision training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ModelManager.load_model(self.config).to(self.device)
        # for i in self.model.named_parameters():
        #     print(f"{i[0]} -> {i[1].device}")

        self.num_epochs = config['epochs']
        self.criterion = getattr(nn, self.config['loss_function'])()
        optimizer_type = getattr(optim, self.config['optimizer']['type'])
        self.optimizer = optimizer_type(self.model.parameters(),
                                        lr=self.config['optimizer']['learning_rate'],
                                        weight_decay=self.config['optimizer']['weight_decay']
                                        )

        self.trainer = Trainer(self.config, self.model, self.optimizer, self.criterion, self.scaler, self.device)


    def execute_training(self):
        with tqdm(range(self.num_epochs), desc="Training Epochs") as pbar:
            for epoch in pbar:
                if epoch==0: metrics = Metrics(self.device, self.config)
                # torch.cuda.empty_cache()  # Clear GPU memory
                # total_train_loss = self.logger.use_profiler(self.trainer, self.train_loader, epoch)
                total_train_loss = self.trainer.train_one_epoch(self.train_loader)
                total_val_loss, metrics = self.trainer.validate_one_epoch(self.val_loader, metrics)
                metrics.compute_metrics(
                    epoch = epoch,
                    train_loss = total_train_loss / len(self.train_loader),
                    val_loss = total_val_loss / len(self.val_loader),
                )
                # self.logger.log_metrics(epoch=epoch, metrics=val_metrics_dict)
        ModelManager.save_checkpoint(self.model, self.config)
        return metrics


    def execute_evaluation(self, metrics):
        print('Evaluating...')
        total_test_loss, metrics = self.trainer.validate_one_epoch(self.test_loader, metrics, to_visualize=True)
        metrics.compute_metrics(test_loss = total_test_loss / len(self.test_loader), mode="test")
        ExperimentLogger.log_metrics(self.config, metrics.metrics)