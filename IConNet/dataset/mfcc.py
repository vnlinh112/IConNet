import torch
from typing import Optional, Iterable
import numpy as np
from .waveform import WaveformDataset
from einops import rearrange
import torchaudio

from .dataset import DEFAULTS

class MFCCDataset(WaveformDataset):
    """Not recommended. Use WaveformDataset and an MFCC model instead."""
    def __init__(
            self, config, 
            feature_name: str,
            data_dir: str=DEFAULTS['data_dir'],
            max_feature_size: Optional[int]=DEFAULTS['max_feature_size_heart'],
            labels: Optional[Iterable[str]]=DEFAULTS['labels'],
            sample_rate: int=DEFAULTS['sample_rate']):
        super().__init__(
            config, 
            feature_name, 
            data_dir,
            max_feature_size,
            labels,
            sample_rate)

    @staticmethod
    def collate_fn(batch):
        transform = torchaudio.transforms.MFCC(
            sample_rate = DEFAULTS['sample_rate'],
            n_mfcc=40,
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
        tensors = transform(tensors)
        targets = torch.stack(targets)
        return tensors, targets