import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from sklearn.model_selection import StratifiedKFold
from typing import Optional, Iterable, Literal
from ..dataset import Dataset, DatasetWrapper
from ..utils.config import DatasetConfig
import math 
import numpy as np
    
class DataModule(L.LightningDataModule):
    def __init__(
            self,
            config: DatasetConfig,
            data_dir: str = "data/",
            batch_size: int = 16,
            num_workers: int = 8,
            pin_memory: bool = False
        ):
        super().__init__()
        
        self.save_hyperparameters()
        self.config = config
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers 
        self.pin_memory = pin_memory
        self.labels = config.target_labels
        self.collate_tn: callable = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes
    
    @property
    def classnames(self) -> Iterable[str]:
        return self.dataset.labels
    
    @property
    def num_channels(self) -> int:
        num_channels = None
        if self.data_train:
            num_channels = self.feature_dim
        elif self.data_test:
            num_channels = self.data_test[0].shape[1]
        elif self.data_predict:
            num_channels = self.data_predict[0].shape[1]
        return num_channels

    def prepare_data(self):
        self.dataset = DatasetWrapper(self.config.dataset_class).init(
            config=self.config,
            data_dir=self.data_dir,
            labels=self.labels
        )
        self.dataset.setup()
        self.feature_dim = self.dataset.feature_dim
        self.collate_fn = self.dataset.collate_fn

    def get_tensor_x(self, data):
        x = [torch.tensor(
            np.array(x, dtype=float), 
            dtype=torch.float) for (x,y) in data]
        return x

    def setup(
            self, 
            stage: Literal["fit", "test", "predict"]="fit",
            data_test: Optional[Iterable]=None,
            data_predict: Optional[Iterable]=None):
        if stage == "fit":
            self.data_train, self.data_val = self.dataset.get_train_test_split()

        if stage == "test":
            if data_test is None:
                self.data_test = self.data_val
            else:
                self.data_test = data_test

        if stage == "predict":
            if data_predict is None:
                self.data_predict = self.get_tensor_x(self.data_val)
            else:
                self.data_predict = data_predict

    @staticmethod
    def get_suitable_batch_size(
        data_size: int, 
        batch_size: int):
        """Get maximum divisor for data_size that is less than or equal to batch_size."""
        num_range = np.arange(min(math.sqrt(data_size), batch_size))[1:][::-1]
        divisors = [i for i in num_range if data_size % i==0]
        return int(divisors[0])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            collate_fn=self.collate_fn,
            drop_last=True,
            shuffle=True
            )

    def val_dataloader(self):
        self.val_batch_size = self.get_suitable_batch_size(
            data_size=len(self.data_val),
            batch_size=self.batch_size
        )
        return DataLoader(
            dataset=self.data_val, 
            batch_size=self.val_batch_size, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
            )
    
    def test_dataloader(self):
        self.test_batch_size = self.get_suitable_batch_size(
            data_size=len(self.data_test),
            batch_size=self.batch_size
        )
        return DataLoader(
            dataset=self.data_test, 
            batch_size=self.test_batch_size, 
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
            )
    
    def predict_dataloader(self):
        self.predict_batch_size = 1
        return DataLoader(
            dataset=self.data_predict, 
            batch_size=self.predict_batch_size, 
            num_workers=self.num_workers
        )

class DataModuleKFold(DataModule):
    """Arguments: 
        fold_number: int, start from 1 to `num_splits`
        split_seed: needs to be always the same for correct cross validation
        num_splits: int, default: 5
    """
    def __init__(
            self,
            config,
            data_dir: str = "data/",
            fold_number: int = 1, 
            split_seed: int = 42, 
            num_splits: int = 5,
            batch_size: int = 16,
            num_workers: int = 0,
            pin_memory: bool = False
        ):
        super().__init__(
            config,
            data_dir,
            batch_size,
            num_workers,
            pin_memory
        )
        
        self.fold_number = fold_number
        self.split_seed = split_seed
        self.num_splits = num_splits
        self.save_hyperparameters()
        assert 0 <= self.fold_number < self.num_splits, "fold number starts from 0" 

    def get_num_classes(self) -> int:
        return self.num_classes

    def prepare_data(self):
        self.dataset = Dataset(
            config = self.config,
            data_dir = self.data_dir,
            labels=self.labels)
        self.dataset.setup()
        self.feature_dim = self.dataset.feature_dim
        self.collate_fn = self.dataset.collate_fn

    def filter_by_indices(self, data, indices):
        out = [data[i] for i in indices]
        return out

    def setup(
            self, 
            stage: Literal['fit', 'test', 'predict']='fit'):
        if stage == "fit":
            kf = StratifiedKFold(
                n_splits=self.num_splits, 
                shuffle=True, 
                random_state=self.split_seed)
            dataset_full = self.dataset.get_data()
            all_splits = [k for k in kf.split(dataset_full, self.dataset.data_y)]

            train_indices, val_indices = all_splits[self.fold_number]
            self.data_val = self.filter_by_indices(dataset_full, val_indices)
            self.data_train = self.filter_by_indices(dataset_full, train_indices)
        
        if stage == "test":
            self.data_test = self.data_val

        if stage == "predict":
            self.data_predict = self.get_tensor_x(self.data_val)
            
    