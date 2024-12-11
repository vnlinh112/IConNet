import torch
from typing import Optional, Iterable
import numpy as np
from .dataset import PickleDataset
from sklearn.model_selection import train_test_split
from ..utils.config import DatasetConfig
from einops import rearrange
# import librosa
from torchaudio import functional as aF
from functools import partial
import torchaudio

from .dataset import DEFAULTS

class WaveformDataset(PickleDataset):
    def __init__(
            self, config: DatasetConfig, 
            data_dir: str=DEFAULTS['data_dir'],
            max_feature_size: Optional[int]=DEFAULTS['max_feature_size'],
            labels: Optional[Iterable[str]]=DEFAULTS['labels'],
            sample_rate: int=DEFAULTS['labels']):
        super().__init__(config, data_dir)
        if labels is not None:
            self.labels = labels
        else:
            self.labels = self.classnames
        self.max_feature_size = max_feature_size
        self.sample_rate = sample_rate

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def label_filter(self, idx):
        label = self.classnames[self.label_values.index(idx)]
        return label in self.labels

    def update_label_index(self, idx):
        label = self.classnames[self.label_values.index(idx)]
        return self.label_to_index(label)
    
    def label_to_index(self, label):
        return self.labels.index(label)
    
    def index_to_label(self, idx):
        return self.labels[idx]

    def quality_check(self):
        y_label, y_encoded = np.unique(self.data_y), np.arange(self.num_classes)
        assert np.array_equal(y_label, y_encoded), "data_y is not yet encoded"
        assert len(self.data_x) > 0, "data_x is empty"
        assert len(self.data_x) == len(self.data_y), "data_x and data_y must have the same size"

        self.num_samples = len(self.data_y)
        self.ensure_data_dim()   
        self.ensure_data_size()

    def ensure_data_dim(self):
        # expected: data_x = [(num_channels, seq_length)]
        if len(self.data_x[0].shape) == 1:
            self.data_x = [d[None, :] for d in self.data_x]
        else:
            dim0 = np.unique([d.shape[0] for d in self.data_x[:10]])
            dim1 = np.unique([d.shape[1] for d in self.data_x[:10]])
            assert min(len(dim0), len(dim1)) == 1, "data_x must have the same number of channels"
            if len(dim0) > 1: # [(seq_length, num_channels)]
                self.data_x = [item.t() for item in self.data_x]    
        self.feature_dim = self.data_x[0].shape[0]          

    def ensure_data_size(self):
        # trim data_x seq_length according to 
        # max_feature_size vs (num_channels*seq_length)
        if self.max_feature_size is not None:
            _data = []
            for d in self.data_x:
                if d.shape[0] * d.shape[1] > self.max_feature_size:
                    offset = (d.shape[0]- self.max_feature_size) // self.feature_dim // 2
                    if offset > 0:
                        d = d[:, offset:-offset]
                _data += [d]
            self.data_x = _data

    def filter(self):
        assert self.data_y is not None
        filtered_idx = [self.label_filter(idx) for idx in self.data_y]
        assert sum(filtered_idx) > 0, "empty data after filtered"
        self.data_x = self.data_x[filtered_idx]
        self.data_y = self.data_y[filtered_idx]
        self.data_y = [self.update_label_index(idx) for idx in self.data_y]

    def setup(
            self, 
            data_x: Optional[Iterable] = None,
            data_y: Optional[Iterable] = None):
        if data_x is not None and data_y is not None: 
            self.data_x = data_x
            self.data_y = data_y
        else:
            self.load()
            if self.config.classnames != self.labels:
                self.filter()
        self.quality_check()

    def get_train_test_split(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.data_x, self.data_y, test_size=DEFAULTS['test_size'], 
            random_state=DEFAULTS['random_state'], stratify=self.data_y)
        train_set = list(zip(x_train, y_train))
        test_set = list(zip(x_test, y_test))
        return train_set, test_set


class Waveform2mfccDataset(WaveformDataset):
    def __init__(
            self, config: DatasetConfig, 
            data_dir: str=DEFAULTS['data_dir'],
            max_feature_size: Optional[int]=DEFAULTS['max_feature_size_heart'],
            labels: Optional[Iterable[str]]=DEFAULTS['labels_heart'],
            sample_rate: int=DEFAULTS['sample_rate']):
        super().__init__(
            config, data_dir, 
            max_feature_size, labels, 
            sample_rate)
        
        self.collate_fn = partial(
            self.transform_batch, 
            sample_rate=self.sample_rate,
            new_sample_rate=2000,
            lowpass_hz = 400)

    @staticmethod
    def transform_batch(
        batch, sample_rate, 
        new_sample_rate=2000, 
        lowpass_hz=400):
        extract_mfcc = torchaudio.transforms.MFCC(
            sample_rate = new_sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 512, 
                    "hop_length": 128, 
                    "n_mels": 64, 
                    "center": False
                    })
        
        tensors, targets = [], []
        for waveform, label in batch:
            tensors += [torch.tensor(
                np.array(waveform, dtype=float), 
                dtype=torch.float32)]
            targets += [torch.tensor(label, dtype=torch.long)]
        tensors = rearrange(
            torch.nn.utils.rnn.pad_sequence(
                [item.t() for item in tensors],
                batch_first=True, padding_value=0.),
            'b n c -> b c n')
        tensors = aF.lowpass_biquad(
            tensors, 
            sample_rate=sample_rate, 
            cutoff_freq=lowpass_hz)
        tensors = aF.preemphasis(tensors)
        tensors = aF.resample(
            tensors, 
            orig_freq=sample_rate, 
            new_freq=new_sample_rate)
        tensors = extract_mfcc(tensors)
        delta1 = aF.compute_deltas(tensors)
        delta2 = aF.compute_deltas(delta1)
        tensors = torch.concat([tensors, delta1, delta2], dim=2)
        targets = torch.stack(targets)
        return tensors, targets
    

class SplittedWaveformDataset(WaveformDataset):
    def __init__(
            self, config: DatasetConfig, 
            data_dir: str=DEFAULTS['data_dir'],
            max_feature_size: Optional[int]=DEFAULTS['max_feature_size'],
            labels: Optional[Iterable[str]]=DEFAULTS['labels_heart'],
            sample_rate: int=DEFAULTS['sample_rate_heart']):
        super().__init__(
            config, data_dir, 
            max_feature_size, labels, 
            sample_rate)
        
    def load(self):
        self.y_train = self.load_feature(self.config.label_name + '.train')
        self.y_test = self.load_feature(self.config.label_name + '.test')
        self.x_train = self.load_feature(self.config.feature_name + '.train')
        self.x_test = self.load_feature(self.config.feature_name + '.test')

    def quality_check(self):
        y_label, y_encoded = np.unique(self.y_train), np.arange(self.num_classes)
        assert np.array_equal(y_label, y_encoded), "y_train is not yet encoded"
        assert len(self.x_train) > 0, "x_train is empty"
        assert len(self.x_train) == len(self.y_train), "x_train and y_train must have the same size"

        self.num_samples = len(self.y_train)
        self.ensure_data_dim()   
        self.ensure_data_size()

    def ensure_data_dim(self):
        # expected: data_x = [(num_channels, seq_length)]
        if len(self.x_train[0].shape) == 1:
            self.x_train = [d[None, :] for d in self.x_train]
            self.x_test = [d[None, :] for d in self.x_test]
        else:
            dim0 = np.unique([d.shape[0] for d in self.x_train[:10]])
            dim1 = np.unique([d.shape[1] for d in self.x_train[:10]])
            assert min(len(dim0), len(dim1)) == 1, "data_x must have the same number of channels"
            if len(dim0) > 1: # [(seq_length, num_channels)]
                self.x_train = [item.t() for item in self.x_train]    
                self.x_test = [item.t() for item in self.x_test]    
        self.feature_dim = self.x_train[0].shape[0]         

    def ensure_data_size(self):
        # trim data_x seq_length according to 
        # max_feature_size vs (num_channels*seq_length)
        if self.max_feature_size is not None:
            _data = []
            for d in self.x_train:
                if d.shape[0] * d.shape[1] > self.max_feature_size:
                    offset = (d.shape[0]- self.max_feature_size) // self.feature_dim // 2
                    if offset > 0:
                        d = d[:, offset:-offset]
                _data += [d]
            self.x_train = _data

            _data = []
            for d in self.x_test:
                if d.shape[0] * d.shape[1] > self.max_feature_size:
                    offset = (d.shape[0]- self.max_feature_size) // self.feature_dim // 2
                    if offset > 0:
                        d = d[:, offset:-offset]
                _data += [d]
            self.x_test = _data

    def filter(self):
        assert self.y_train is not None
        filtered_idx = [self.label_filter(idx) for idx in self.y_train]
        assert sum(filtered_idx) > 0, "empty data after filtered"
        self.x_train = self.x_train[filtered_idx]
        self.y_train = self.y_train[filtered_idx]
        self.y_train = [self.update_label_index(idx) for idx in self.y_train]

        assert self.y_test is not None
        filtered_idx = [self.label_filter(idx) for idx in self.y_test]
        assert sum(filtered_idx) > 0, "empty data after filtered"
        self.x_test = self.x_test[filtered_idx]
        self.y_test = self.y_test[filtered_idx]
        self.y_test = [self.update_label_index(idx) for idx in self.y_test]
    
    def get_train_test_split(self):
        train_set = list(zip(self.x_train, self.y_train))
        test_set = list(zip(self.x_test, self.y_test))
        return train_set, test_set