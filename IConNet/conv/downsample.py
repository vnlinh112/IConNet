from typing import Callable, Optional, Union
from torch import nn, Tensor 
from einops import rearrange, reduce
from .pad import PadForConv

def downsample(
        x: Tensor, 
        downsample_factor: int=1
    ) -> Tensor:
    x = PadForConv(
            kernel_size=downsample_factor,
            pad_mode='mean')(x)
    x = reduce(
            rearrange(x, '... (n s) -> ... n s', 
                        s=downsample_factor),
            '... n s -> ... n', 'mean')
    return x

class DownsampleLayer(nn.Module):
    """Downsample (i.e decimate) using avg pool."""
    def __init__(self, downsample_factor: int=1):
        super().__init__()
        assert downsample_factor > 0
        self.downsample_factor = downsample_factor
        self.pad_right = PadForConv(
            kernel_size=downsample_factor,
            pad_mode='mean')

    def _downsample(self, x: Tensor) -> Tensor:
        x = self.pad_right(x)
        x = reduce(
                rearrange(x, '... (n s) -> ... n s', 
                            s=self.downsample_factor),
                '... n s -> ... n', 'mean')
        return x
  
    def forward(self, x: Tensor) -> Tensor:
        if self.downsample_factor == 1:
            return x
        x = self._downsample(x)
        return x