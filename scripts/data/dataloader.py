from torch.utils.data import DataLoader
from scripts.data.dataset import PolypDataset
from scripts.visualizations.visualization_utils import visualize_data

class DataLoadingLocal:
    def __init__(self, config, shuffle_train_data=True):
        # set_seed()
        self.config = config
        # self.train_ratio = self.config['dataset']['train_ratio']
        # self.val_ratio = self.config['dataset']['val_ratio']
        self.batch_size = self.config['batch_size']
        self.shuffle_train_data = shuffle_train_data  # after we iterate over all batches the data is shuffled

        self.num_workers = self.config['num_workers']
        # self.worker_init_fn = worker_init_fn
        self.pin_memory = self.config['pin_memory']
        self.persistent_workers = self.config['persistent_workers']


    def get_loaders(self, viz=False):
        # train_dataset = PolypDataset(self.config, mode="train", include_data=self.include_data)
        # val_dataset = PolypDataset(self.config, mode="val", include_data=self.include_data)
        # test_dataset = PolypDataset(self.config, mode="test", include_data=self.include_data)
        train_dataset = PolypDataset(mode="train")
        val_dataset  = PolypDataset(mode="val")
        test_dataset = PolypDataset(mode="test")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train_data, num_workers=self.num_workers, pin_memory=self.pin_memory)
        val_loader   = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        test_loader  = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

        if viz: visualize_data(self.config, train_loader, num_samples=5)
        return train_loader, val_loader, test_loader
#

class DataLoadingCloud:
    def __init__(self, config, shuffle_train_data=True):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.shuffle_train_data = shuffle_train_data  # after we iterate over all batches the data is shuffled

        self.num_workers = 0
        self.pin_memory = True
        self.persistent_workers = False

    def get_loaders(self, viz=False):
        train_dataset = PolypDataset(mode="train")
        val_dataset = PolypDataset(mode="val")
        test_dataset = PolypDataset(mode="test")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train_data,
                                  num_workers=self.num_workers, pin_memory=self.pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        if viz: visualize_data(self.config, train_loader, num_samples=5)

        return train_loader, val_loader, test_loader
