import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np

def acov_filter(
        data: Tensor, 
        acov: Tensor, 
        eps=1e-10) -> Tensor:
    sum_acov = sum(acov)
    if sum_acov == 0:
        sum_acov = eps
    filtered_data = F.convolve(
        data, acov, mode='same') / sum_acov
    return filtered_data

def hann_window(window_length):
    x = torch.linspace(0.5, window_length-0.5, steps=window_length)
    return torch.sin(torch.pi*x / window_length)**2 

def acov_compute(
        data: Tensor, 
        window_length=None, 
        position=None, 
        windowing_fn=hann_window):
    if not position:
        position = torch.randint(len(data) - window_length)
    windowed_data = data[position:(position + window_length)]
    max_windata = max(0, torch.max(torch.abs(windowed_data)))
    win = windowing_fn(window_length)
    windowed_data = windowed_data * win / max_windata # normalize
    spectrum = torch.fft.rfft(windowed_data, n=2*window_length)**2
    acov = torch.fft.ifftshift(torch.fft.irfft(spectrum.abs()))
    return acov

class LocalPatternFilter(nn.Module):
    def __init__(
            self, 
            n_filter: int, 
            kernel_size: int=1024) -> None:
        super().__init__()
        self.n_filter = n_filter
        self.kernel_size = kernel_size

    def _generate_filters_positions(self, length):
        positions = [torch.randint(length - self.kernel_size) for _ in range(self.n_filter)]
        return torch.tensor(np.array(positions))

    @staticmethod
    def compute_autocovariance(
        data: Tensor, 
        positions: Tensor, 
        kernel_size: int=1024) -> Tensor:
        acov = []
        for position in positions:
            windowed_data = data[..., position:(position + kernel_size)]
            max_windata = max(0, torch.max(torch.abs(windowed_data)))
            win = torch.hann_window(kernel_size)
            windowed_data = windowed_data * win / max_windata # normalize
            spectrum = torch.fft.rfft(windowed_data, n=2*kernel_size)**2
            acov += [torch.fft.ifftshift(torch.fft.irfft(spectrum.abs()))]
        return torch.stack(acov)

    def _generate_filters(self, X) -> Tensor:
        length = X.shape[-1]
        positions = self._generate_filters_positions(length)
        filters = self.compute_autocovariance(
            X, positions, kernel_size=self.kernel_size)
        return filters
    
    def forward(self, X: Tensor) -> Tensor:
        filters = self._generate_filters(X)
        X = F.conv1d(X, filters)
        return X

    