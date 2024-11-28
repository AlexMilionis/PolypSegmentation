from torch.utils.data import DataLoader, random_split
from data.constants import Constants  # Assuming constants have paths
from data.scripts.polyp_dataset import PolypDataset  # Assuming PolypDataset is already implemented
from hyperparameters import Hyperparameters
from scripts.seed import worker_init_fn, set_generator


class DataLoading:
    def __init__(self, mode, include_data, shuffle=True, num_workers=4):
        """
        Initializes the dataset and splits into train/test/full based on the mode.

        Args:
            mode (str): Mode for the dataset ('train', 'test', 'full').
            train_ratio (float): Ratio of data used for training (default 0.8).
            batch_size (int): Batch size for DataLoader.
            shuffle (bool): Whether to shuffle data in DataLoader.
            num_workers (int): Number of workers for data loading.
        """
        self.mode = mode
        self.include_data = include_data
        self.train_ratio = Hyperparameters.TRAIN_RATIO
        self.batch_size = Hyperparameters.BATCH_SIZE
        self.shuffle = shuffle if mode == 'train' else False
        self.num_workers = num_workers
        self.worker_init_fn = worker_init_fn
        self.dataset = PolypDataset(mode=self.mode, include_data=self.include_data)
        self.data_loader = self._create_dataloader()


    def split_data(self):
        # Split dataset into train/test
        train_size = int(self.train_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size], generator=set_generator())
        return train_dataset, test_dataset


    def _create_dataloader(self):

        train_dataset, test_dataset = self.split_data()

        if self.mode == 'train':
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)

        elif self.mode == 'test':
            return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn)

        else:
            # raise ValueError(f"Invalid mode '{self.mode}'. Use 'train' or 'test'.")
            assert self.mode in ['train','test'], "Use train or test mode!"


    def get_loader(self):
        """
        Returns the DataLoader for the selected mode.

        Returns:
            DataLoader: The DataLoader instance.
        """
        return self.data_loader

