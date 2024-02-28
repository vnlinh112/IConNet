import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from sklearn.model_selection import StratifiedKFold
from typing import Optional, Iterable, Literal
from ..dataset import WaveformDataset as Dataset
from ..utils.config import DatasetConfig
import math 
import numpy as np
    
class DataModule(L.DataModule):
    def __init__(
            self,
            config: DatasetConfig,
            data_dir: str = "data/",
            batch_size: int = 16,
            num_workers: int = 0,
            pin_memory: bool = False
        ):
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        self.config = config
        self.collate_tn: callable = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.config.num_classes
    
    @property
    def classnames(self) -> Iterable[str]:
        return self.config.classnames

    def prepare_data(self):
        # download...
        pass

    def setup(
            self, 
            stage: Literal["fit", "test", "predict"]="fit",
            data_test: Optional[Iterable]=None,
            data_predict: Optional[Iterable]=None):
        if stage == "fit":
            ds = Dataset(
                config=self.config,
                data_dir=self.hparams.data_dir,
                labels=self.hparams.labels)
            ds.setup()
            self.data_train, self.data_val = ds.get_train_test_split()
            self.collate_fn = ds.collate_fn

        if stage == "test":
            if data_predict is None:
                self.data_test = self.data_val
            else:
                self.data_test = data_test

        if stage == "predict":
            self.data_predict = data_predict

    @staticmethod
    def get_suitable_batch_size(
        data_size: int, 
        batch_size: int):
        """Get maximum divisor for data_size that is less than or equal to batch_size."""
        num_range = np.arange(min(math.sqrt(data_size), batch_size))[1:][::-1]
        divisors = [i for i in num_range if data_size % i==0]
        return divisors[0]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            collate_fn=self.collate_fn,
            drop_last=True,
            shuffle=True
            )

    def val_dataloader(self):
        self.val_batch_size = self.get_suitable_batch_size(
            data_size=len(self.data_val),
            batch_size=self.hparams.batch_size
        )
        return DataLoader(
            dataset=self.data_val, 
            batch_size=self.val_batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn
            )
    
    def test_dataloader(self):
        self.test_batch_size = self.get_suitable_batch_size(
            data_size=len(self.data_test),
            batch_size=self.hparams.batch_size
        )
        return DataLoader(
            dataset=self.data_test, 
            batch_size=self.test_batch_size, 
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn
            )
    
    def predict_dataloader(self):
        self.predict_batch_size = self.get_suitable_batch_size(
            data_size=len(self.data_test),
            batch_size=self.hparams.batch_size
        )
        return DataLoader(
            dataset=self.data_predict, 
            batch_size=self.predict_batch_size, 
            num_workers=self.hparams.num_workers
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
        super().__init__()
        assert 1 <= self.k <= self.num_splits, "incorrect fold number" 

    def get_num_classes(self) -> int:
        return self.num_classes

    def prepare_data(self):
        self.dataset = Dataset(
            config = self.config,
            feature_name = self.hparams.feature_name,
            data_dir = self.hparams.data_dir)
        self.collate_fn = self.dataset.collate_fn

    def setup(
            self, 
            stage: Literal['fit', 'test', 'predict']='fit'):
        if stage == "fit":
            kf = StratifiedKFold(
                n_splits=self.hparams.num_splits, 
                shuffle=True, 
                random_state=self.hparams.split_seed)
            dataset_full = self.dataset.get_data()
            all_splits = [k for k in kf.split(dataset_full)]

            train_indices, val_indices = all_splits[self.hparams.fold_number]
            self.data_train = dataset_full[train_indices.tolist()] 
            self.data_val = dataset_full[val_indices.tolist()]

        if stage == "test":
            self.data_test = self.dataset.get_data()

        if stage == "predict":
            self.data_predict = self.dataset.get_data()
    