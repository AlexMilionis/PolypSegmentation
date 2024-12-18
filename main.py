from src.data.dataloader import DataLoading
from src.experiments.run_scripts.transfer_learning import Trainer
from src.scripts.evaluate import Evaluator
from src.config.seed import set_seed

set_seed()
if __name__ == '__main__':

    train_loader, val_loader, test_loader = DataLoading(include_data = 'single_frames', shuffle=False).get_loaders()
    # visualize_inputs(train_loader, num_samples = 5)    # Visualize some data
    Trainer(train_loader, val_loader, transfer_learning=True).train()
    Evaluator(test_loader, visualize_predictions=False).evaluate()
