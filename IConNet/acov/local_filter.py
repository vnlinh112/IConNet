from typing import Literal
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from ..nn.activation import gumbel_softmax
from .zero_crossing import zero_crossing_score

class LocalPatternFilter(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int=1023) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.position_concentration = nn.Parameter(torch.rand(self.out_channels))

    @staticmethod
    def compute_autocovariance(
            X: Tensor,
            positions,
            kernel_size) -> Tensor:
        acov = []
        for position in positions:
            end_pos = position + kernel_size
            windowed_data = X[..., position:end_pos]
            max_windata = max(0, torch.max(torch.abs(windowed_data)))
            win = torch.hann_window(kernel_size, dtype=X.dtype, device=X.device)
            windowed_data = windowed_data * win / max_windata # normalize
            spectrum = torch.fft.rfft(windowed_data, n=2*kernel_size)**2
            acov += [torch.fft.ifftshift(torch.fft.irfft(spectrum.abs()))]
        return torch.stack(acov, dim=1)

    def generate_positions(
        self,
        X: Tensor,
        mode: Literal['fixed', 
                    'learnable_condition', 
                    'zerocrossing']='fixed',
        ):
        """
        fixed: linspace
        conditioned: Dirichlet, 
                    learnable params: concentration vector (length=out_channels)
        zerocrossing: zerocrossing with gumbel softmax selection 
        """
        length = X.shape[-1]
        end = length - self.kernel_size - 1
        n_positions = self.out_channels
        if mode=='fixed':
            positions = torch.linspace(
                start=0, 
                end=end,
                steps=n_positions,
                dtype=torch.int)
        elif mode=='learnable_condition':
            pdf = torch.distributions.dirichlet.Dirichlet(
                self.position_concentration)
            positions = torch.cumsum((pdf.sample() * end).type(torch.int), dim=0)
        else: # zerocrossing
            positions = torch.topk(
                gumbel_softmax(zero_crossing_score(X), tau=1), 
                n_positions).indices
        return positions
        
    def _generate_filters(self, X) -> Tensor:
        length = X.shape[-1]
        positions = self.generate_positions(X, mode='fixed')
        filters = self.compute_autocovariance(X, positions, self.kernel_size)
        return filters
    
    def forward(self, X: Tensor) -> Tensor:
        filters = self._generate_filters(X)
        # assert filters.shape == (self.out_channels, self.in_channels, self.kernel_size)
        # X = F.conv1d(X, filters)
        return filters
