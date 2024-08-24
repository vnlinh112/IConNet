import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import Optional, Iterable, Literal
from ..dataset import Dataset, DatasetWrapper, DEFAULTS
from ..utils.config import DatasetConfig
import math 
import numpy as np

import collections
TensorWithMask = collections.namedtuple(
    "TensorDataLoader", ["tensor", "mask"])
    
class SimpleDataModule(nn.Module):
    def __init__(
            self,
            config: DatasetConfig,
            data_dir: str = DEFAULTS["data_dir"],
            batch_size: int = 16,
            num_workers: int = 8,
            pin_memory: bool = False
        ):
        super().__init__()
        
        self.config = config
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers 
        self.pin_memory = pin_memory
        self.labels = config.target_labels
        self.collate_fn: callable = None
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
    
    @staticmethod
    def get_suitable_batch_size(
        data_size: int, 
        batch_size: int):
        """Get maximum divisor for data_size that is less than or equal to batch_size."""
        num_range = np.arange(min(math.sqrt(data_size), batch_size))[1:][::-1]
        divisors = [i for i in num_range if data_size % i==0]
        if len(divisors) > 0:
            batch_size = int(divisors[0])
        else:
            batch_size = 1
        return batch_size
    
    @staticmethod
    def collate_fn_with_mask(batch) -> tuple[TensorWithMask, Tensor]:
        tensors, targets = [], []
        for feature, label in batch:
            tensors += [torch.tensor(
                np.array(feature, dtype=float), 
                dtype=torch.float32)]
            targets += [torch.tensor(label, dtype=torch.long)]
        mask_tensors = torch.nn.utils.rnn.pad_sequence(
                [torch.ones_like(item.t()) for item in tensors],
                batch_first=True, padding_value=0.).t()
        tensors = torch.nn.utils.rnn.pad_sequence(
                [item.t() for item in tensors],
                batch_first=True, padding_value=0.).t()
        targets = torch.stack(targets)
        return TensorWithMask(tensors, mask_tensors), targets

    def prepare_data(self):
        self.dataset = DatasetWrapper(self.config.dataset_class).init(
            config=self.config,
            data_dir=self.data_dir,
            labels=self.labels
        )
        self.dataset.setup()
        self.feature_dim = self.dataset.feature_dim
        self.collate_fn = self.dataset.collate_fn
        # self.collate_fn = self.collate_fn_with_mask

    def setup(
            self, 
            stage: Literal["fit", "test"]="fit",
            data_test: Optional[Iterable]=None):
        if stage == "fit":
            self.data_train, self.data_val = self.dataset.get_train_test_split()

        if stage == "test":
            if data_test is None:
                self.data_test = self.data_val
            else:
                self.data_test = data_test

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
