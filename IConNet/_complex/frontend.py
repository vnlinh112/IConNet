from typing import Literal, Optional, Union, Callable
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from einops import rearrange, reduce, repeat
import numpy as np
from collections import OrderedDict

from ..nn.activation import nl_relu, NLReLU
from ..firconv.firconv import FirConvLayer
from ..nn.normalizer import CustomNormLayer
from .downsample2 import Downsample2 as ComplexFirconv
from .firconv import FirConvLayer as Firconv2
from functools import partial

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
            n_channel: Union[
                tuple[int], list[int]]=(64, 64),
            kernel_size: Union[
                tuple[int], list[int]]=(5, 13),
            stride: Union[
                tuple[int], list[int]]=(1, 1),
            window_k: Union[
                tuple[int], list[int]]=(2, 3),
            residual_connection_type: 
                Optional[Literal[
                'stack', 'concat', 'add', 
                'contract', 'gated_add', 
                'gated_contract']]='gated_contract',
            residual_gate_fn: Callable=nl_relu,
            filter_type: Literal['firwin', 'sinc']='firwin',
            norm_layer_name: Literal[
                'BatchNorm',
                'LayerNorm',
                'InstanceNorm',
                'GroupNorm',
                'LocalResponseNorm'
            ]='LocalResponseNorm',
            pooling: Optional[Literal[
                'max', 'mean',
                'sum', 'min']]=None,
            is_complex: bool=True
        ):
        super().__init__()

        self.n_block = n_block
        self.n_input_channel = n_input_channel
        self.n_channel = n_channel 
        self.kernel_size = kernel_size
        self.stride = stride 
        self.window_k = window_k
        self.residual_connection_type = residual_connection_type
        self.residual_gate_fn = residual_gate_fn
        self.filter_type = filter_type
        self.norm_layer_name = norm_layer_name
        self.pooling = pooling
        self.blocks = self._create_blocks(is_complex=is_complex)
        self._set_output_channel()
        self.act = NLReLU()

    def _create_blocks(self, is_complex=True):
        if is_complex: # firwin fftconv
            Layer = ComplexFirconv
        else:
            Layer = partial(Firconv2, layer_mode=self.filter_type)
            # Layer = partial(FirConvLayer, filter_type=self.filter_type)
        
        blocks = [nn.Sequential(OrderedDict({
                "layer": Layer(
                    self.n_input_channel, self.n_channel[0], 
                    kernel_size=self.kernel_size[0], 
                    stride=self.stride[0], 
                    window_k=self.window_k[0]),
                "norm": CustomNormLayer(
                    name=self.norm_layer_name,
                    num_channels=self.n_channel[0])
            }))]
        blocks += [nn.Sequential(OrderedDict({
                "layer": Layer(
                    self.n_channel[i-1], self.n_channel[i], 
                    kernel_size=self.kernel_size[i], 
                    stride=self.stride[i], 
                    window_k=self.window_k[i]),
                "norm": CustomNormLayer(
                    name=self.norm_layer_name,
                    num_channels=self.n_channel[i])
            })) for i in range(1, self.n_block)]
        
        return nn.ModuleList(blocks)

    def _set_output_channel(self):
        if self.residual_connection_type == 'stack': # B 2C N
            self.n_output_channel = sum(self.n_channel) 
        elif self.residual_connection_type == 'concat': # B C 2N
            assert len(np.unique(self.n_channel)) == 1
            self.n_output_channel = self.n_channel[0] 
        elif self.residual_connection_type == 'add': # B C N
            assert len(np.unique(self.n_channel)) == 1
            self.n_output_channel = self.n_channel[-1] 
        elif self.residual_connection_type == 'contract': # B C N
            self.n_output_channel = self.n_channel[-1] 
        elif self.residual_connection_type == 'gated_add': # B C N
            assert len(np.unique(self.n_channel)) == 1
            self.n_output_channel = self.n_channel[-1] 
        elif self.residual_connection_type == 'gated_contract': # B C N
            self.n_output_channel = self.n_channel[-1] 
        if self.pooling is not None:
            self.n_output_channel = 1

    def _apply_residual_connection(self, x1: Tensor, x2: Tensor):
        if self.residual_connection_type == 'stack': # TODO: downsample then stack
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
        elif self.residual_connection_type == 'gated_add':
            assert x1.shape == x2.shape
            x1 = self.residual_gate_fn(x1)
            x2 = x1 + x2
        elif self.residual_connection_type == 'gated_contract':
            x1 = self.residual_gate_fn(x1)
            x2 = torch.einsum('bkm,bhn -> bhn', x1, x2)
        return x2
    
    def _apply_pooling(self, x: Tensor):
        x = reduce(x, '... c n -> ... 1 c', self.pooling)
        return x

    def forward(self, x):
        b = x.shape[0]
        x = self.blocks[0](x)
        for i in range(1, self.n_block):
            x1 = self.blocks[i](self.act(x))
            x = self._apply_residual_connection(x, x1)
        x = self.act(x)
        if self.pooling:
            self._apply_pooling(x)
        return x 