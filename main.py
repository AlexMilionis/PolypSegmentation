from scripts.visualization_utils import visualize_data
from data.scripts.dataloader import DataLoading
from scripts.transfer_learning import train_model
from scripts.evaluate import evaluate_model
from scripts.seed import set_seed

if __name__ == '__main__':
    set_seed()

    train_loader = DataLoading(mode="train", include_data = 'single_frames', shuffle=False).get_loader()
    test_loader = DataLoading(mode="test", include_data = 'single_frames',shuffle=False).get_loader()

    # visualize_data(test_loader, num_samples = 3)    # Visualize some data


    model_checkpoint_path = train_model(train_loader)
    evaluate_model(test_loader, model_checkpoint_path, visualize_results=True)




