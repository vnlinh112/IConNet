import torch
from torch import nn, Tensor
from torch.nn import functional as F

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

class SimpleLocalPatternFilter(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int=1024) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    @staticmethod
    def compute_autocovariance(
            X: Tensor,
            positions,
            kernel_size) -> Tensor:
        acov = []
        for position in positions:
            windowed_data = X[..., position:(position + kernel_size)]
            max_windata = max(0, torch.max(torch.abs(windowed_data)))
            win = torch.hann_window(kernel_size)
            windowed_data = windowed_data * win / max_windata # normalize
            spectrum = torch.fft.rfft(windowed_data, n=2*kernel_size)**2
            acov += [torch.fft.ifftshift(torch.fft.irfft(spectrum.abs()))]
        return torch.stack(acov, dtype=X.dtype, device=X.device)

    def _generate_filters(self, X) -> Tensor:
        length = X.shape[-1]
        positions = torch.linspace(
            start=0, 
            end=length - self.kernel_size,
            steps=self.out_channels,
            dtype=torch.int)
        filters = self.compute_autocovariance(X, positions, self.kernel_size)
        return filters
    
    def forward(self, X: Tensor) -> Tensor:
        filters = self._generate_filters(X)
        assert filters.shape == (self.out_channels, self.in_channels, self.kernel_size)
        X = F.conv1d(X, filters)
        return X

    