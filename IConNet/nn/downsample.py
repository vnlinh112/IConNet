import math 
import torch
from torch import nn, Tensor 
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from .pad import pad_right

def downsample(n_dim, downsample_factor) -> nn.Conv1d:
    """Downsample (i.e decimate) using strided convolution"""
    return nn.Conv1d(n_dim, n_dim, 
                     kernel_size=1, 
                     stride=downsample_factor, bias=False)

class DownsampleLayer(nn.Module):
    """Downsample (i.e decimate) using avg pool.
        TODO: Use polyphase when the `orig_sample_rate` 
        is not a multiple of the `new_sample_rate`. 
    """
    def __init__(self, downsample_factor: int):
        super().__init__()
        assert downsample_factor > 0
        self.downsample_factor = downsample_factor
        
    def _downsample_pooling(self, x):
        """Deprecated due to cannot only pad right"""
        n = x.shape[-1]
        pad = math.ceil(n/self.downsample_factor) * self.downsample_factor - n
        x = F.avg_pool1d(
            x, 
            kernel_size = self.downsample_factor, 
            stride = self.downsample_factor, 
            padding = (0,pad),  # error
            ceil_mode = False,
            count_include_pad = False)
        return x
    
    def _pad_right(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        residual = n % self.downsample_factor
        if residual > 0:
            x_residual = repeat(
                reduce(
                    x[..., -residual:], 
                    '... c n -> ... c 1', 'mean'),
                '... c 1 -> ... c s',
                s=self.downsample_factor)
            x = torch.cat([x[..., :-residual], x_residual], dim=-1)
        return x 

    def _downsample(self, x: Tensor) -> Tensor:
        return self.downsample(x, self.downsample_factor)
    
    @staticmethod
    def downsample(x: Tensor, downsample_factor: int=1) -> Tensor:
        x = pad_right(x, kernel_size=downsample_factor)
        x = reduce(
                rearrange(x, '... (n s) -> ... n s', 
                          s=downsample_factor),
                '... n s -> ... n', 'mean')
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample_factor == 1:
            return x
        x = self._downsample(x)
        return x