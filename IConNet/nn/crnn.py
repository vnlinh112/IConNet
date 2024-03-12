import torch
import torch.nn as nn
from einops import rearrange
import torchaudio
from torchaudio import functional as aF
from functools import partial

class CRNN(nn.Module):
    def __init__(self, config=None, n_input=1, n_output=2):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 39, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=39),
            nn.Conv2d(39, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=64),
        )
        self.n_output_channel = 768 
        self.lstm = nn.LSTM(
            input_size=self.n_output_channel, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True)
        h0 = torch.zeros(2,64)
        c0 = torch.zeros_like(h0)
        self.register_buffer("h0", h0)
        self.register_buffer("c0", c0)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, self.n_output)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.cnn(x)
        x = rearrange(x, 'b c w h -> b (c w h)')
        x, _ = self.lstm(x, (self.h0, self.c0))
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 
    

class MFCCFeatures(nn.Module):
    def __init__(self, 
                 sample_rate=16000, 
                 new_sample_rate=2000, 
                 lowpass_hz=400):
        super().__init__()
        self.sample_rate = sample_rate
        self.new_sample_rate = new_sample_rate
        self.lowpass_hz = lowpass_hz
        self.lowpass_filter = partial(
            aF.lowpass_biquad, 
            sample_rate=sample_rate,
            cutoff_freq=lowpass_hz)
        self.downsample = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=new_sample_rate
        )
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate = new_sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 512, 
                    "hop_length": 128, 
                    "n_mels": 64, 
                    "center": False
                    })

    def forward(self, x):       
        x = self.lowpass_filter(x)
        x = aF.preemphasis(x)
        x = self.downsample(x)
        x = self.mfcc(x)
        delta1 = aF.compute_deltas(x)
        delta2 = aF.compute_deltas(x)
        x = torch.concat([x, delta1, delta2], dim=2)
        return x
    

class MFCC_CRNN(nn.Module):
    def __init__(self, config=None, n_input=1, n_output=2):
        super().__init__()
        self.fe = MFCCFeatures()
        self.cls = CRNN(config=None, n_input=1, n_output=2)

    def forward(self, x):
        x = self.fe(x)
        x = self.cls(x)
        return x
