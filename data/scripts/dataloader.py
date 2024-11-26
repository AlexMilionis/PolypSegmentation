from torch.utils.data import DataLoader, random_split
from data.constants import Constants  # Assuming constants have paths
from data.scripts.polyp_dataset import PolypDataset  # Assuming PolypDataset is already implemented
from hyperparameters import Hyperparameters

class DataLoading:
    def __init__(self, mode, shuffle=True, num_workers=4):
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
        self.train_ratio = Hyperparameters.TRAIN_RATIO
        self.batch_size = Hyperparameters.BATCH_SIZE
        self.shuffle = shuffle if mode == 'train' else False
        self.num_workers = num_workers

        # Initialize the full dataset
        self.dataset = PolypDataset(
            images_dir=Constants.IMAGE_DIR,
            masks_dir=Constants.MASK_DIR,
            mode=self.mode,
        )

        self.data_loader = self._create_dataloader()


    def split_data(self):
        # Split dataset into train/test
        train_size = int(self.train_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        return train_dataset, test_dataset


    def _create_dataloader(self):

        train_dataset, test_dataset = DataLoading.split_data(self)

        if self.mode == 'train':
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        elif self.mode == 'test':
            return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Use 'train' or 'test'.")


    def get_loader(self):
        """
        Returns the DataLoader for the selected mode.

        Returns:
            DataLoader: The DataLoader instance.
        """
        return self.data_loader

