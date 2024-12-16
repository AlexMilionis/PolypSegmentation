from scripts.visualization_utils import visualize_inputs
from data.scripts.dataloader import DataLoading
from scripts.transfer_learning import Trainer
from scripts.evaluate import Evaluator
from scripts.seed import set_seed

if __name__ == '__main__':
    set_seed()

    train_loader, val_loader, test_loader = DataLoading(include_data = 'single_frames').get_loaders()
    # visualize_data(train_loader, num_samples = 5)    # Visualize some data
    Trainer(train_loader, transfer_learning=True).train()
    Evaluator(test_loader, visualize_results=True).evaluate()



