import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from einops import rearrange, reduce
from .downsample import downsample
from .pad import PadForConv
from typing import Optional

def expanding_fft_conv(
        u: Tensor, v: Tensor, 
        stride: int=1,
        expanding_dim: int=1 # 1: stack, -1: concat (ensure same channel)
        ) -> Tensor:
    """
    Conv without stride then downsample.
    The input is also downsampled then stack along the `expanding_dim` axis.
    """
    y = fft_conv(u, v, stride=stride)
    u = downsample(u, stride)
    y = torch.cat([y, u], dim=expanding_dim)
    return y


def complex_matmul(
        a: Tensor,
        b: Tensor
    ) -> Tensor:
    """
    bcn,hcn->bhn or cn,hcn->hn
    """
    return torch.einsum('...cn,hcn->...hn', a, b)

def fft_conv_kernel(
        u: Tensor, 
        v: Tensor,
        stride: int=1, 
        conjugate: bool=True,
        n_fft: Optional[int]=2048) -> Tensor:
    if n_fft is None:
        n_fft = u.shape[-1]
    u_f = torch.fft.rfft(u, n=n_fft) 
    v_f = torch.fft.rfft(v, n=n_fft) 
    if conjugate:
        v_f.imag *= -1
    y_f = torch.einsum('bcn,hcn->bhn', u_f, v_f)
    y   = torch.fft.irfft(y_f, n=n_fft) 
    y   = y.abs() 
    y = downsample(y, stride)
    return y

def fft_conv(
        u: Tensor, 
        v: Tensor, 
        stride: int=1, 
        conjugate: bool=True,
        n_fft: int=2048) -> Tensor:
    """
    Conv without stride then downsample.
    """
    L   = u.shape[-1]
    u   = PadForConv(
            kernel_size=n_fft,
            pad_mode='reflect').apply_pad(u)
    u   = rearrange(
            fft_conv_kernel(
                rearrange(
                    u,
                    'b c (n n_fft) -> (n_fft b) c n', 
                    n_fft=n_fft), 
                v, 
                stride=stride, 
                conjugate=conjugate, 
                n_fft=n_fft),
            '(n_fft b) h n -> b h (n n_fft)', 
            n_fft=n_fft)
    L = int(math.ceil(L/stride))
    return u[..., :L]