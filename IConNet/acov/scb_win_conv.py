from typing import Literal, Optional, Callable
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from einops import rearrange, reduce, repeat
import numpy as np
import math
from ..firconv.signal import (
    get_last_window_time_point,
    DEFAULT_SAMPLE_RATE)
from functools import partial

class SCBWinConv(nn.Module):
    """Convolution layer with learnable window function & SCB weights.

    Parameters
    ----------
    in_channels :   Number of input channels.
    out_channels :  Number of filters.
    kernel_size :   Filter length. This does not affect the number of learnable parameters.
    stride :        `int`, also downsampling factor. out_length = ceil(in_length / stride)

    filter_type :   `constant`: embedding weights from pretrained SCB
    filter_type :   file path of tensor SCB pretrained embedding weights (out_channels, kernel_size)
    learnable_windows: If `True` (default), the window(s) will be parametrized 
                        with `window_k` parameters using the general consine window function [3].
    shared_window:  Using one window filter across all channels. Default is `False`.
                    This will be ignored when `learnable_windows`=`learnable_windows`.
    window_func:    'hamming', 'hanning', 'hann', 'rectangle' or 'none', optional.
                    Window function. Default is `None`. Only be used when `learnable_windows`=`False`.
    window_k:       `int`, optional
                    Number of learnable params for the window. Default is 2.
    sample_rate :   `int`. Sample rate for setting `min_low_hz` and `min_band_hz`. 
                    Default is 16000 Hz.

    """
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int=1, 
            groups=1,
            learnable_windows: bool=True,
            shared_window: bool=False,
            window_k: int=2,
            filter_type: Literal['constant']='constant',
            filter_init: str='', 
            window_func: Optional[Literal['hann', 'hanning', 
                                'hamming', 'rectangle', 'none']]=None,
            sample_rate=DEFAULT_SAMPLE_RATE,
            dtype=torch.float32, eps=1e-12):

        super().__init__()
        assert in_channels >=1 and out_channels >= 1 
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size >= 1
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0: 
            kernel_size -= 1

        self.kernel_size = kernel_size
        self.stride = stride
        self.filter_type = filter_type
        self.groups = groups
        
        self.conv_layer = partial(
            F.conv1d,
            stride=self.stride, 
            groups=self.groups,
            padding='same'
        )

        self.dtype = dtype
        self.eps = eps
        self.learnable_windows = learnable_windows
        self.shared_window = shared_window
        self.sample_rate = sample_rate
        
        self.filter_init = filter_init
        self.window_func = window_func
        self.window_k = window_k

        self._init_windows()
        self._init_filters(filter_init)
    
    def _init_windows(self):
        if self.learnable_windows:
            self._init_learnable_windows()
        else:
            self._init_windows_func()
    
    def _init_windows_func(self):
        if self.window_func == 'hamming':
            window = torch.hamming_window(self.kernel_size)
        elif self.window_func.isin(['hann', 'hanning']):
            window = torch.hanning_window(self.kernel_size)
        else: # 'none', rectangle
            window = torch.ones((self.kernel_size))
        self.register_buffer('windows', window) 

    def _init_learnable_windows(self):
        assert self.window_k > 0
        window_params = torch.rand(self.window_k, dtype=self.dtype)
        if not self.shared_window:
            window_params = repeat(
                window_params, 'p -> h c p', 
                h=self.out_channels, c=self.in_channels).contiguous()
        self.window_params = nn.Parameter(window_params)
        self.last_window_time_point = get_last_window_time_point(self.kernel_size)
        
    def _generate_learnable_windows(self):
        """generate general cosine window from win_params"""
        assert self.learnable_windows == True
        self.window_params_idx = torch.arange(
            self.window_k, dtype=self.dtype, 
            device=self.device, requires_grad=True)
        self.window_time_mesh = torch.einsum(
            'p,k->pk', 
            self.window_params_idx, 
            torch.linspace(
                start = 0, 
                end = self.last_window_time_point, 
                steps = self.kernel_size,
                dtype=self.dtype, device=self.device, 
                requires_grad=True))
        self.windows = reduce(
            torch.einsum(
                '...p,p,pk->...pk',
                self.window_params,
                torch.cos(self.window_params_idx * torch.pi), 
                torch.cos(self.window_time_mesh)), 
            '... p k -> ... k', 'sum').contiguous()

    def _init_filters(self, embedding_weight_path):
        filters = torch.load(embedding_weight_path)
        assert filters.shape == (self.out_channels, self.kernel_size)
        filters = repeat(
            filters, 'h k -> h c k', 
            c=self.in_channels).contiguous()
        self.register_buffer('filters', filters)
    

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, in_channels, n_samples)
            Batch of waveforms.

        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of SCB filters activations.
        """
        self.device = waveforms.device
        if self.learnable_windows:
            self._generate_learnable_windows()

        waveforms = self.conv_layer(waveforms, self.filters * self.windows)
        return waveforms

