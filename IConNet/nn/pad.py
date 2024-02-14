from einops import rearrange, reduce, repeat
from typing import Literal, Optional
import torch
from torch import Tensor, nn

def pad_right(
        x: Tensor, 
        kernel_size: int,
        pad_mode: Optional[Literal[
        'mean', 'zero', 'constant', 
        'replicate', 'reflect']]='mean',
        constant_value=0
    ) -> Tensor:

    n = x.shape[-1]
    residual = n % kernel_size
    pad = kernel_size
    if residual <= 0:
        return x
    
    if pad_mode == 'mean':
        mean_items = residual
        x_pad = repeat(
            reduce(
                x[..., -mean_items:], 
                '... c n -> ... c 1', 'mean'),
            '... c 1 -> ... c p',
            p=pad)
    elif pad_mode == 'replicate':
        x_pad = x[..., -pad:]
    elif pad_mode == 'reflect':
        x_pad = torch.flip(x[..., -pad:], dims=(-1, ))
    else:
        x_pad = torch.full_like(
            x[..., -pad:], 
            fill_value=constant_value)
    crop_items = residual
    x = torch.cat([x[..., :-crop_items], x_pad], dim=-1)
    return x 

class PadRight(nn.Module):
    def __init__(
            self,
            pad: int,
            pad_mode: Optional[Literal[
            'mean', 'zero', 'constant', 
            'replicate', 'reflect']]='mean',
            constant_value=0):

        super().__init__()
        self.pad = pad
        self.pad_mode = pad_mode
        self.constant_value = constant_value
    
    def forward(self, x: Tensor) -> Tensor:
        if self.pad <= 0:
            return x
        pad = self.pad
        pad_mode = self.pad_mode
        constant_value = self.constant_value
        if pad_mode == 'mean':
            x_pad = repeat(
                reduce(
                    x[..., -pad:], 
                    '... c n -> ... c 1', 'mean'),
                '... c 1 -> ... c p',
                p=pad)
        elif pad_mode == 'replicate':
            x_pad = x[..., -pad:]
        elif pad_mode == 'reflect':
            x_pad = torch.flip(x[..., -pad:], dims=(-1, ))
        else:
            x_pad = torch.full_like(
                x[..., -pad:], 
                fill_value=constant_value)
        x = torch.cat([x, x_pad], dim=-1)
        return x