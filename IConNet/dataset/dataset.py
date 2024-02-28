import torch
from typing import Optional, Iterable
import numpy as np
import os
from einops import rearrange
from ..utils.config import DatasetConfig, get_valid_path

class PickleDataset:
    def __init__(
            self, config: DatasetConfig, 
            data_dir: str = "data/",
            pickle_file_ext: str = "npy"):
        self.config = config
        self.name = self.config.name
        self.classnames = list(np.array(self.config.classnames).flatten())
        self.label_values = list(np.array(self.config.label_values).flatten())
        self.pickle_file_ext = pickle_file_ext
        self.data_dir = get_valid_path(data_dir) + get_valid_path(self.config.root)
        self.feature_dir = self.data_dir + get_valid_path(self.config.feature_dir)
        assert os.path.isdir(self.data_dir)
        self.data_x: Optional[Iterable] = None
        self.data_y: Optional[Iterable] = None

    @property
    def num_classes(self) -> int:
        return len(self.labels)
    
    def load_feature(self, feature_name: str):
        file_path = f"{self.feature_dir}{self.name}.{feature_name}.{self.pickle_file_ext}"
        data = np.load(file_path, allow_pickle=True)
        return data

    def label_to_index(self, label):
        return self.classnames.index(label)
    
    def index_to_label(self, idx):
        return self.classnames[idx]

    def load(self):
        self.data_y = self.load_feature(self.config.label_name)
        self.data_x = self.load_feature(self.config.feature_name)
    
    @staticmethod
    def collate_fn(batch):
        tensors, targets = [], []
        for feature, label in batch:
            tensors += [torch.tensor(
                np.array(feature, dtype=float), 
                dtype=torch.float32)]
            targets += [torch.tensor(label, dtype=torch.long)]
        tensors = rearrange(
            torch.nn.utils.rnn.pad_sequence(
                [item.t() for item in tensors],
                batch_first=True, padding_value=0.),
            'b n c -> b c n')
        targets = torch.stack(targets)
        return tensors, targets
    
    def setup(
            self, 
            data_x: Iterable,
            data_y: Iterable):       
        self.data_x = data_x
        self.data_y = data_y

    def get_data(self):
        if self.data_x is None or self.data_y is None:
            self.setup()
        return list(zip(self.data_x, self.data_y))
    
    


