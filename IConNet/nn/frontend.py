from typing import Literal, Optional, Union, Callable, Iterable
import torch
from torch import Tensor, nn
from einops import rearrange, reduce
from ..firconv import FirConvLayer
from ..conv import ResidualConv1d

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
            n_input_channel: int=1,
            n_channel: Iterable[int]=(64, 64),
            kernel_size: Optional[Iterable[int]]=(5, 13),
            stride: Optional[Iterable[int]]=(1, 1),
            window_k: Iterable[int]=(2, 3),
            residual_connection_type: 
                Optional[Literal[
                'stack', 'concat', 'add', 
                'contract']]='concat',
            filter_type: Literal['firwin', 'sinc']='firwin',
            learnable_bands: bool=True,
            learnable_windows: bool=True,
            shared_window: bool=False,
            filter_init: Optional[Literal[
                'lognorm', 'mel', 'random', 
                'none']]='mel', 
            mel_resolution: int=4,
            window_func: Optional[Literal['hann', 'hanning', 
                                'hamming', 'rectangle', 'none']]=None,
            conv_mode: Literal['conv', 'fftconv']='fftconv',
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
        self.window_k = window_k
        self.residual_connection_type = residual_connection_type
        self.filter_type = filter_type
        self.conv_mode = conv_mode
        self.norm_type = norm_type
        self.pooling = pooling

        self.learnable_bands = learnable_bands
        self.learnable_windows = learnable_windows
        self.shared_window = shared_window
        self.window_func = window_func
        self.mel_resolution = mel_resolution

        self.in_channels = [n_input_channel] + self.n_channel[:-1]
        self.blocks = nn.ModuleList([FrontEndBlock(
                in_channels = self.in_channels[i], 
                out_channels = self.n_channel[i], 
                kernel_size = self.kernel_size[i], 
                window_k = self.window_k[i], 
                stride = self.stride[i], 
                conv_mode = self.conv_mode,
                norm_type = self.norm_type,
                residual_connection_type = self.residual_connection_type,
                filter_type = self.filter_type,
                learnable_bands = self.learnable_bands,
                learnable_windows = self.learnable_windows,
                shared_window = self.shared_window,
                window_func = self.window_func,
                mel_resolution = self.mel_resolution,
            ) for i in range(self.n_block)])
        
        self.n_output_channel = self.get_output_features()

        if self.pooling is not None:
            self.n_output_channel = 1
    
    def get_all_output_features(self) -> list[int]:
        outputs = [self.blocks[i].get_output_features()
                   for i in range(self.n_block)]
        return outputs 
    
    def get_output_features(self) -> int:
        if self.residual_connection_type == 'stack':
            return self.n_input_channel + sum(self.n_channel)
        return self.blocks[-1].get_output_features()

    def _apply_pooling(self, x: Tensor):
        if self.pooling:
            x = reduce(x, '... c n -> ... 1 c', self.pooling)
        return x
    
    def _apply_blocks(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x):
        x = self._apply_blocks(x)
        if self.pooling is not None:
            x = self._apply_pooling(x)
        return x 

class FrontEndBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            kernel_size: Optional[int]=511,
            stride: Optional[int]=1,
            groups: Optional[int]=1,
            window_k: Optional[int]=2,
            residual_connection_type: 
                Optional[Literal[
                'stack', 'concat', 'add', 
                'contract']]='concat',
            filter_type: Literal['firwin', 'sinc']='firwin',
            learnable_bands: bool=True,
            learnable_windows: bool=True,
            shared_window: bool=False,
            filter_init: Optional[Literal[
                'lognorm', 'mel', 'random', 
                'none']]='mel', 
            mel_resolution: int=4,
            window_func: Optional[Literal['hann', 'hanning', 
                                'hamming', 'rectangle', 'none']]=None,
            conv_mode: Literal['conv', 'fftconv']='fftconv',
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
        self.groups = groups 
        self.window_k = window_k
        self.residual_connection_type = residual_connection_type
        self.filter_type = filter_type
        self.conv_mode = conv_mode
        self.norm_type = norm_type

        self.learnable_bands = learnable_bands
        self.learnable_windows = learnable_windows
        self.shared_window = shared_window
        self.window_func = window_func
        self.mel_resolution = mel_resolution

        if groups > 1 and residual_connection_type=='stack':
            raise ValueError(f'FIRConv does not support groups \
                             when residual_connection_type==`stack`.\
                             Set groups=1 instead.')

        conv = ResidualConv1d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            groups = groups,
            conv_mode = self.conv_mode,
            norm_type = self.norm_type,
            residual_connection_type = self.residual_connection_type
        )
        layer = FirConvLayer(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            window_k = window_k,
            filter_type = filter_type,
            conv_layer = conv,
            learnable_bands = self.learnable_bands,
            learnable_windows = self.learnable_windows,
            shared_window = self.shared_window,
            window_func = self.window_func,
            mel_resolution = self.mel_resolution)
        
        self.block = nn.ModuleDict({
            "layer": layer,
            "downsample": conv,
        })

    def get_output_features(self) -> int:
        return self.block.downsample.out_channels

    def forward(self, X: Tensor) -> Tensor:
        return self.block.layer(X)