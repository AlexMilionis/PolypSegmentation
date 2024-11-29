"""
    DataLoading Class
    -----------------

    This class handles the loading and splitting of datasets into training and testing sets. It creates PyTorch DataLoader objects for efficient data loading during training and evaluation.

    Attributes:
        mode (str): Specifies the mode of operation ('train' or 'test').
        include_data (str): Specifies the data subset to include (e.g., 'single_frames', 'seq_frames', or 'both').
        train_ratio (float): The ratio of the dataset used for training. The remainder is used for testing.
        batch_size (int): Batch size for loading data.
        shuffle (bool): Whether to shuffle the data during loading.
        num_workers (int): Number of subprocesses used for data loading.
        worker_init_fn (function): Function to initialize each worker for consistent seeding.
        dataset (Dataset): The dataset object containing image-mask pairs.
        data_loader (DataLoader): The PyTorch DataLoader instance.

    Methods:
        split_data(): Splits the dataset into training and testing subsets based on `train_ratio`.
        _create_dataloader(): Creates the appropriate DataLoader object for the specified mode.
        get_loader(): Returns the DataLoader object for the selected mode.
    """


from torch.utils.data import DataLoader, random_split
from data.scripts.polyp_dataset import PolypDataset
from hyperparameters import Hyperparameters
from scripts.seed import worker_init_fn, set_generator


class DataLoading:
    def __init__(self, mode, include_data, shuffle=True, num_workers=8, pin_memory=False):
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
            assert self.mode in ['train','test'], "Use train or test mode!"


    def get_loader(self):
        return self.data_loader
