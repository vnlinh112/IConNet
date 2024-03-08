import torch
from typing import Optional, Iterable
import numpy as np
from einops import rearrange
from .waveform import WaveformDataset
from sklearn.model_selection import train_test_split
from ..utils.config import DatasetConfig
import librosa
from torchaudio import functional as aF
from functools import partial

class HeartsoundDataset(WaveformDataset):
    def __init__(
            self, config: DatasetConfig, 
            data_dir: str = "data/",
            max_feature_size: Optional[int]=1e5,
            labels: Optional[Iterable[str]] = 
                ['murmur', 'normal'],
            sample_rate=16000):
        super().__init__(
            config, data_dir, 
            max_feature_size, labels, 
            sample_rate)
        
        self.collate_fn = partial(
            self.transform_batch, 
            sample_rate=self.sample_rate,
            lowpass_hz = 400)
    
    @staticmethod
    def transform_batch(batch, sample_rate, lowpass_hz=400):
        tensors, targets = [], []
        for feature, label in batch:
            tensors += [feature]
            targets += [torch.tensor(label, dtype=torch.long)]
        targets = torch.stack(targets)

        data = np.array(feature, dtype=float)
        data = librosa.feature.mfcc(n_mfcc=13,y=data, sr=sample_rate)
        data1 = librosa.feature.delta(data, order=1)
        data2 = librosa.feature.delta(data, order=2)
        data = np.concatenate((data, data1, data2), axis=2)
        data = torch.from_numpy(data)
        data = aF.lowpass_biquad(data, sample_rate, lowpass_hz)
        data = aF.preemphasis(data)
        
        return tensors, targets