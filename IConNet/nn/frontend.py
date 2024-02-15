from typing import Literal, Optional, Union, Callable, Iterable
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from einops import rearrange, reduce, repeat
import numpy as np
from collections import OrderedDict

from .activation import NLReLU
from .FirConv import FirConvLayer
from .normalizer import CustomNormLayer
from .downsample import DownsampleLayer
from .pad import PadRight

class FeBlocks(nn.Module):
    """
    Output: (B Cx Nx)
    Cx, Nx: depend on the `residual_connection_type` and `pooling`
    residual_connection_type:
        * `stack`: stack along channel dim (requires same length across channels)
        * `concat`: concat (requires same channel dimension across blocks)
        * `add`: add (similar to ResNet, requires same shape input)
        * `contract`: contract('bch,buv->buv')
        * `gated_*`: apply activation before performing add or contract
        * None, or `none`: return the last input
    """
    def __init__(self, 
            n_block: int=2,
            n_input_channel: int=3,
            n_channel: Iterable[int]=(64, 64),
            kernel_size: Iterable[int]=(5, 13),
            stride: Iterable[int]=(1, 1),
            groups: Iterable[int]=(1, 1),
            window_k: Iterable[int]=(2, 3),
            residual_connection_type: 
                Optional[Literal[
                'stack', 'concat', 'add', 
                'contract']]='contract',
            conv_type: Literal['firwin', 'sinc']='firwin',
            conv_mode: Literal['conv', 'fftconv']='conv',
            n_fft: Optional[int]=2048,
            norm_type: Literal[
                'BatchNorm',
                'LayerNorm',
                'InstanceNorm',
                'GroupNorm',
                'LocalResponseNorm'
            ]='LocalResponseNorm',
            pooling: Optional[Literal[
                'max', 'mean',
                'sum', 'min']]=None
        ):
        super().__init__()

        self.n_block = n_block
        self.n_input_channel = n_input_channel
        self.n_channel = n_channel 
        self.kernel_size = kernel_size
        self.stride = stride 
        self.groups = groups
        self.window_k = window_k
        
        self.residual_connection_type = residual_connection_type
        self.conv_type = conv_type
        self.conv_mode = conv_mode
        self.n_fft = n_fft
        self.norm_type = norm_type
        self.pooling = pooling

        if self.residual_connection_type=='stack':
            norm_channels = n_input_channel + n_channel[0]
        else: 
            norm_channels = n_channel[0]

        blocks = [nn.Sequential(OrderedDict({
                "pad": PadRight(
                    pad=kernel_size[0] - 1,
                    pad_mode='reflect'),
                "layer": FirConvLayer(
                    in_channels = n_input_channel, 
                    out_channels = n_channel[0], 
                    kernel_size=kernel_size[0], 
                    stride=1, 
                    groups=groups[0],
                    window_k=window_k[0],
                    conv_type = conv_type,
                    conv_mode = conv_mode,
                    n_fft = n_fft),
                "downsample": DownsampleLayer(
                    downsample_factor=stride[0]),
                "norm": CustomNormLayer(
                    name=norm_type,
                    num_channels=norm_channels)
            }))]
        
        n_channel_i_2 = n_input_channel
        for i in range(1, n_block):
            if self.residual_connection_type == 'stack':
                in_channels = n_channel_i_2 + n_channel[i-1]
                n_channel_i_2 = n_channel[i-1]
                norm_channels = n_channel[i-1] + n_channel[i]
            else:
                in_channels = n_channel[i-1]
                norm_channels = n_channel[i]
            blocks += [nn.Sequential(OrderedDict({
                    "pad": PadRight(
                        pad=kernel_size[i] - 1,
                        pad_mode='reflect'),
                    "layer": FirConvLayer(
                        in_channels = in_channels, 
                        out_channels = n_channel[i], 
                        kernel_size=kernel_size[i], 
                        stride=1, 
                        groups=groups[i],
                        window_k=window_k[i],
                        conv_type = conv_type,
                        conv_mode = conv_mode,
                        n_fft = n_fft),
                    "downsample": DownsampleLayer(
                        downsample_factor=stride[i]),
                    "norm": CustomNormLayer(
                        name=norm_type,
                        num_channels=norm_channels)
                }))]
        
        self.blocks = nn.ModuleList(blocks)
        self._set_output_channel()
        self.act = NLReLU()
        

    def _set_output_channel(self):
        if self.residual_connection_type == 'stack': # B 2C N
            self.n_output_channel = sum(self.n_channel) + self.n_input_channel
        elif self.residual_connection_type == 'concat': # B C 2N
            assert len(np.unique(self.n_channel)) == 1
            self.n_output_channel = self.n_channel[0] 
        elif self.residual_connection_type == 'add': # B C N
            assert len(np.unique(self.n_channel)) == 1
            self.n_output_channel = self.n_channel[-1] 
        elif self.residual_connection_type == 'contract': # B C N
            self.n_output_channel = self.n_channel[-1] 
        if self.pooling is not None:
            self.n_output_channel = 1

    def _apply_residual_connection(self, x1: Tensor, x2: Tensor):
        if self.residual_connection_type == 'stack': 
            assert x1.shape[-1] == x2.shape[-1]
            x2 = torch.cat([x1, x2], dim=1) # B 2C N
        elif self.residual_connection_type == 'concat':
            assert x1.shape[1] == x2.shape[1]
            x2 = torch.cat([x1, x2], dim=-1) # B C 2N
        elif self.residual_connection_type == 'add':
            assert x1.shape == x2.shape
            x2 = x1 + x2
        elif self.residual_connection_type == 'contract':
            x2 = torch.einsum('bkm,bhn -> bhn', x1, x2)
        return x2
    
    def _apply_pooling(self, x: Tensor):
        x = reduce(x, '... c n -> ... 1 c', self.pooling).contiguous()
        return x

    def forward(self, x):
        for i in range(0, self.n_block):
            if self.conv_mode == 'fftconv':
                x1 = x
            else:
                x1 = self.blocks[i].pad(x)
            x1 = self.blocks[i].layer(x1)
            x1 = self.blocks[i].downsample(x1)
            x = self.blocks[i].downsample(x)
            x = self._apply_residual_connection(x, self.act(x1))
            x = self.blocks[i].norm(x)
        if self.pooling:
            self._apply_pooling(x)
        return x 
    