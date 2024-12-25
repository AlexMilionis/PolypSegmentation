from src.data.dataloader import DataLoading
from src.experiments.exp1.run_unet import Trainer
from src.scripts.evaluate import Evaluator
from src.config.seed import set_seed
from src.scripts.visualization_utils import visualize_inputs

set_seed()
if __name__ == '__main__':

    train_loader, val_loader, test_loader = DataLoading(include_data = 'single_frames', shuffle=True).get_loaders()
    visualize_inputs(train_loader, num_samples = 5)    # Visualize some data
    Trainer(train_loader, val_loader, transfer_learning=True).train()
    Evaluator(test_loader, visualize_predictions=True).evaluate()
