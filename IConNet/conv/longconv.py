from typing import Callable, Optional, Union, Literal
import torch
from torch import nn, Tensor 
from torch.nn import functional as F
from einops import rearrange, reduce
from .pad import PadForConv, PadRight
import math
from .downsample import DownsampleLayer
from ..nn.normalizer import CustomNormLayer
from ..nn.activation import NLReLU
from ..firconv import signal

class LongConv1d(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            kernel_size: int, 
            stride: int=1, 
            groups: int=1,
            conv_mode: Literal[
                'conv', 'fftconv']='conv',
            n_fft: Optional[int]=None,
            norm_type: Literal[
                'BatchNorm',
                'LayerNorm',
                'InstanceNorm',
                'GroupNorm',
                'LocalResponseNorm'
            ]='LocalResponseNorm'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        
        if groups > 1 and conv_mode == 'fftconv':
            raise ValueError(
                f'LongConv1d with fftconv mode does not support groups.') 
        
        self.groups = groups
        self.conv_mode = conv_mode

        if conv_mode == 'conv':
            self.pad_layer = PadRight(
                    pad=self.kernel_size - 1,
                    pad_mode='reflect')
        else:
            self.n_fft = signal.get_nfft(kernel_size, n_fft)
            self.pad_layer = PadForConv(
                    kernel_size=self.n_fft,
                    pad_mode='reflect')
        
        self.stride_layer = DownsampleLayer(
            downsample_factor=stride)
        
        self.norm_type = norm_type
        self.norm_layer = CustomNormLayer(
            name=norm_type,
            num_channels=out_channels)
        self.act = NLReLU()

    def post_conv(self, X: Tensor) -> Tensor:
        X   = self.stride_layer(X)
        X   = self.act(X)
        X   = self.norm_layer(X)
        return X

    def _apply_conv(
            self, X: Tensor, filters: Tensor) -> Tensor:
        n = X.shape[-1]
        X = self.pad_layer(X)
        X = F.conv1d(X, filters, groups=self.groups)
        X = self.stride_layer(X)
        m = int(math.ceil(n/self.stride))
        X = self.post_conv(X)
        return X[..., :m]
    
    def _apply_fftconv(
            self, X: Tensor, filters: Tensor) -> Tensor:
        n = X.shape[-1]
        X = self.pad_layer(X)
        X = self._fft_conv(X, filters)
        m = int(math.ceil(n/self.stride))
        return X[..., :m]
    
    def _fft_conv_kernel(
            self,
            u: Tensor, 
            v: Tensor,
            conjugate: bool=True) -> Tensor:
        """
        Conv without stride then downsample.
        """
        u_f = torch.fft.rfft(u, n=self.n_fft) 
        v_f = torch.fft.rfft(v, n=self.n_fft) 
        if conjugate:
            v_f.imag *= -1
        y_f = torch.einsum('bcn,hcn->bhn', u_f, v_f)
        y   = torch.fft.irfft(y_f, n=self.n_fft).abs() 
        y   = self.post_conv(y)
        return y
    
    def _fft_conv_kernel_wrapper(
            self,
            u: Tensor, 
            v: Tensor) -> Tensor:
        return self._fft_conv_kernel(u, v)

    def _fft_conv(self, u: Tensor, v: Tensor) -> Tensor: 
        t = u.shape[-1] // self.n_fft
        u   = rearrange(
                self._fft_conv_kernel_wrapper(
                    rearrange(u, 'b c (n t) -> (t b) c n', t=t), 
                    v),
                '(t b) h n -> b h (n t)', t=t)
        return u
    
    def downsample(self, X: Tensor) -> Tensor:
        return self.stride_layer(X)

    def forward(
            self, X: Tensor, filters: Tensor) -> Tensor:
        assert filters.shape[0] == self.out_channels
        assert filters.shape[1] == self.in_channels
        if self.conv_mode == 'conv':
            return self._apply_conv(X, filters)
        return self._apply_fftconv(X, filters)


class ResidualConv1d(LongConv1d):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int = 1, 
            groups: int = 1, 
            conv_mode: Literal['fftconv'] = 'fftconv', 
            n_fft: Optional[int] = None, 
            norm_type: Literal[
                'BatchNorm', 'LayerNorm', 
                'InstanceNorm', 'GroupNorm', 
                'LocalResponseNorm'] = 
                'LocalResponseNorm',
            residual_connection_type: 
                Optional[Literal[
                'stack', 'concat', 'add', 
                'contract']]='contract'):
        
        super().__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            groups, 
            conv_mode, 
            n_fft, 
            norm_type=norm_type)
        
        
        self.residual_connection_type = residual_connection_type
        if residual_connection_type == 'stack':
            residual_fn = lambda x, y: torch.cat([x, y], dim=1)
            self.out_channels = out_channels + in_channels
            self.norm_layer = CustomNormLayer(
                name=norm_type,
                num_channels=self.out_channels
            )
        elif residual_connection_type == 'add':
            residual_fn = lambda x, y: x + y
        elif residual_connection_type == 'contract':
            residual_fn = lambda x, y: torch.einsum('bcn,bhn -> bhn', x, y)
        elif residual_connection_type == 'concat' and in_channels==out_channels:
            residual_fn = lambda x, y: torch.cat([x, y], dim=-1)
        else: # concat but different in_channels and out_channels
            self.residual_connection_type = None
            residual_fn = lambda x, y: y
        self.apply_residual_connection = residual_fn
    
    def _fft_conv_kernel_wrapper(
            self,
            u: Tensor, 
            v: Tensor) -> Tensor:
        residual = u.shape[1] - self.in_channels
        y = self._fft_conv_kernel(u[:, residual:, :], v)
        r = self.stride_layer(u)
        return self.apply_residual_connection(r, y)
    
    def forward(
            self, X: Tensor, filters: Tensor) -> Tensor:
        assert filters.shape[1] == self.in_channels
        return self._apply_fftconv(X, filters)
