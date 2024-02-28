import torch
from typing import Optional, Iterable
import numpy as np
import os
from einops import rearrange

class PickleDataset:
    def __init__(
            self, config, 
            data_dir: str = "data/",
            pickle_file_ext: str = "npy"):
        self.config = config
        self.name = self.config.name
        self.num_classes = self.config.num_classes
        self.classnames = self.config.classnames
        self.pickle_file_ext = pickle_file_ext
        self.data_dir = self.get_valid_path(data_dir) + self.get_valid_path(self.config.root)
        assert os.path.isdir(self.data_dir)
        self.data_x: Optional[Iterable] = None
        self.data_y: Optional[Iterable] = None
    
    @staticmethod
    def get_valid_path(path: str):
        if path.endswith('/'):
            return path 
        return path + '/'
    
    def load_feature(self, feature_name: str):
        file_path = f"{self.data_dir}{self.name}.{feature_name}.{self.pickle_file_ext}"
        data = np.load(file_path, allow_pickle=True)
        return data

    def label_to_index(self, label):
        return self.classnames.index(label)
    
    def index_to_label(self, idx):
        return self.classnames[idx]

    def load(self):
        self.data_y = self.load_feature(self.config.label_file)
        self.data_x = self.load_feature(self.feature_name)
    
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
    
    


