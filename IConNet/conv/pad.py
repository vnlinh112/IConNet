from einops import rearrange, reduce, repeat
from typing import Literal, Optional
import torch
from torch import Tensor, nn

class PadRight(nn.Module):
    def __init__(
            self,
            pad: Optional[int]=None,
            pad_mode: Optional[Literal[
            'mean', 'zero', 'constant', 
            'replicate', 'reflect']]='mean',
            constant_value=0):

        super().__init__()
        self.pad = pad
        self.pad_mode = pad_mode
        self.constant_value = constant_value

    def _generate_pad(
        self,
        x, 
        mean_items: Optional[int]=None
    ) -> Tensor:
        if self.pad_mode == 'mean':
            if mean_items is None:
                mean_items = self.pad
            x = repeat(
                reduce(
                    x[..., -mean_items:], 
                    '... c n -> ... c 1', 'mean'),
                '... c 1 -> ... c p',
                p=self.pad)
        elif self.pad_mode == 'replicate':
            x = x[..., -self.pad:]
        elif self.pad_mode == 'reflect':
            x = torch.flip(x[..., -self.pad:], dims=(-1, ))
        else:
            x = torch.full_like(
                x[..., -self.pad:], 
                fill_value=self.constant_value)
        return x 
    
    def apply_pad(self, x) -> Tensor:
        if self.pad <= 0:
            return x
        x_pad = self._generate_pad(x, mean_items=self.pad)
        x = torch.cat([x, x_pad], dim=-1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.apply_pad(x)
    
class PadForConv(PadRight):
    def __init__(
            self,
            kernel_size: int,
            pad_mode: Optional[Literal[
            'mean', 'zero', 'constant', 
            'replicate', 'reflect']]='mean',
            constant_value=0):

        super().__init__()
        assert kernel_size > 0
        self.kernel_size = kernel_size
        self.pad = kernel_size
        self.pad_mode = pad_mode
        self.constant_value=constant_value

    
    def apply_pad(self, x) -> Tensor:
        n = x.shape[-1]
        residual = n % self.pad
        if residual <= 0:
            return x
    
        if self.pad <= 0:
            return x
        x_pad = self._generate_pad(x, mean_items=residual)
        x = torch.cat([x[..., :-residual], x_pad], dim=-1)
        return x
