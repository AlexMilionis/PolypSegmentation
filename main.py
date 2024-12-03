from scripts.visualization_utils import visualize_data
from data.scripts.dataloader import DataLoading
from scripts.transfer_learning import Trainer
from scripts.evaluate import Evaluator
from scripts.seed import set_seed

if __name__ == '__main__':
    set_seed()

    train_loader = DataLoading(mode="train", include_data = 'single_frames', shuffle=False).get_loader()
    test_loader = DataLoading(mode="test", include_data = 'single_frames',shuffle=False).get_loader()

    # visualize_data(test_loader, num_samples = 3)    # Visualize some data

    Trainer(train_loader, transfer_learning=True).train()
    Evaluator(test_loader, visualize_results=True).evaluate()




