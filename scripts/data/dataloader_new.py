from torch.utils.data import DataLoader, random_split
from scripts.data.dataset_new import PolypDataset
from scripts.seed import set_generator
from scripts.constants import Constants


# TODO: CONVERT THIS TO CLASS
class DataLoading:

    def __init__(self, config):
        self.config = config
        # self.num_workers = config['num_workers']
        # self.pin_memory = config['pin_memory']
        # self.persistent_workers = config['persistent_workers']


    def get_dataloaders(self, run_in_colab):

        full_dataset = PolypDataset(
            images_dir = Constants.TRAIN_VAL_IMAGES_DIR,
            masks_dir = Constants.TRAIN_VAL_MASKS_DIR,
            mode="train",
            preload=False
        )
        # print("Full dataset length:", len(full_dataset))  # Debug print
        # print("Images dir:", Constants.TRAIN_VAL_IMAGES_DIR)
        # print("Masks dir:", Constants.TRAIN_VAL_MASKS_DIR)

        val_split = self.config["val_split"]
        # random_split
        full_len = len(full_dataset)
        val_len = int(full_len * val_split)
        train_len = full_len - val_len

        train_subset, val_subset = random_split(full_dataset, [train_len, val_len], generator=set_generator(42))

        # Override val_subset to have val transforms & preload
        # We'll just re-wrap val_subset.dataset in a new PolypDataset so we can preload
        # 1) get all the image indices from val_subset
        val_indices = val_subset.indices  # list of indices
        # 2) gather the actual paths
        val_data_pairs = [full_dataset.data_pairs[i] for i in val_indices]
        # 3) build a new PolypDataset with "val" mode, preload=True
        #    we artificially pass the same data but different mode
        val_dataset = PolypDataset(
            images_dir=Constants.TRAIN_VAL_IMAGES_DIR,
            masks_dir=Constants.TRAIN_VAL_MASKS_DIR,
            mode="val",    # deterministic transforms
            preload=True
        )
        # However, we only want the specific subset of data. We'll store them in val_dataset, ignoring the default collect.
        val_dataset.data_pairs = val_data_pairs
        # re-run preload step for that subset
        val_dataset.preloaded_data = []
        for (img_path, msk_path) in val_data_pairs:
            img, msk = val_dataset._read_and_transform(img_path, msk_path)
            val_dataset.preloaded_data.append((img, msk, (img_path, msk_path)))

        # Similarly for test
        test_dataset = PolypDataset(
            images_dir=Constants.TEST_IMAGES_DIR,
            masks_dir=Constants.TEST_MASKS_DIR,
            mode="test",     # deterministic transforms
            preload=True     # preload
        )

        # train_subset is still a Subset -> wrap in a DataLoader
        # val_dataset is a direct Dataset

        if run_in_colab:
            train_loader = DataLoader(
                train_subset, 
                batch_size=self.config['batch_size'], 
                num_workers=12, 
                pin_memory=True, 
                persistent_workers=True, 
                shuffle=True,
                drop_last=True,
                )
            val_loader = DataLoader(
                val_dataset,  
                batch_size=self.config['batch_size'], 
                num_workers=12,
                pin_memory=True, 
                persistent_workers=True, 
                shuffle=False,
                drop_last=False,
                )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.config['batch_size'], 
                num_workers=12,
                pin_memory=True, 
                shuffle=False,
                drop_last=False,
                )
            
        else:   # local execution
            train_loader = DataLoader(
                train_subset, 
                batch_size=self.config['batch_size'], 
                num_workers=4, 
                pin_memory=True, 
                persistent_workers=True, 
                shuffle=True,
                drop_last=True,
                )
            val_loader = DataLoader(
                val_dataset,  
                batch_size=self.config['batch_size'], 
                num_workers=4,
                pin_memory=True, 
                persistent_workers=True, 
                shuffle=False,
                drop_last=False,
                )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.config['batch_size'], 
                num_workers=12,
                pin_memory=True, 
                shuffle=False,
                drop_last=False,
                )

        return train_loader, val_loader, test_loader
