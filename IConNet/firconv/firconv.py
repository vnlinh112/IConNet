from typing import Literal, Optional, Callable
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from einops import rearrange, reduce, repeat
import numpy as np
import math
from .signal import (
    firwin_mesh, to_mel, to_hz, 
    get_last_window_time_point, get_nfft, 
    DEFAULT_SAMPLE_RATE)
from ..nn.activation import logical_switch
from functools import partial

class FirConvLayer(nn.Module):
    """Interface for the FIR-kernel-based convolution layer with learnable window function & transition bandwidth.

    Parameters
    ----------
    in_channels :   Number of input channels.
    out_channels :  Number of filters.
    kernel_size :   Filter length. This does not affect the number of learnable parameters.
    stride :        `int`, also downsampling factor. out_length = ceil(in_length / stride)

    filter_type :    `firwin` (default) or `sinc`.
        - `firwin`: FIR-filter-based convolution layer using windowing method. [1]
        - `sinc`:   Multi-channel Sinc-based convolution. [2]
    conv_mode:      `conv` or `fftconv` (defautl).
        - `conv`:   using conv1d function
        - `fftconv`: using FFT conv (multiplication in the freq domain) with complex numbers
    n_fft:          int, default: 2048. Used for `fftconv`. 

    learnable_bands: If `True` (default), each kernel will be parametrized by the 
                    `lowcut_band` and `bandwidth` parameters.
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

    References:           
    ----------
    [1] scipy.signal.firwin2
    [2] SincNet
    [3] scipy.signal.windows.general_cosine 

    """
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int=1, 
            groups=1,
            conv_layer: Optional[Callable]=None,
            learnable_bands: bool=True,
            learnable_windows: bool=True,
            shared_window: bool=False,
            window_k: int=2,
            filter_type: Literal['firwin', 'sinc']='firwin',
            filter_init: Optional[Literal[
                'lognorm', 'mel', 'random', 
                'none']]='mel', 
            mel_resolution: int=4,
            window_func: Optional[Literal['hann', 'hanning', 
                                'hamming', 'rectangle', 'none']]=None,
     
            min_low_hz=0, min_band_hz=0, 
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
        self.n_fft = get_nfft(kernel_size)
        self.stride = stride
        self.filter_type = filter_type
        self.groups = groups
        
        if conv_layer is None:
            self.conv_layer = partial(
                F.conv1d,
                stride=self.stride, 
                groups=self.groups
            )
        else:
            self.conv_layer = conv_layer

        self.dtype = dtype
        self.eps = eps
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.learnable_bands = learnable_bands
        self.learnable_windows = learnable_windows
        self.shared_window = shared_window
        self.sample_rate = sample_rate
        
        self.filter_init = filter_init
        self.set_mel_resolution(mel_resolution)
        self.window_func = window_func
        self.window_k = window_k

        self._init_bands()
        self._init_windows()

        if self.filter_type == 'sinc':
            self._sinc_init_filters()
        else: # 'firwin'
            self._firwin_init_filters()  
        
    def set_mel_resolution(
            self, mel_resolution, 
            min_mel_bins: int=8):
        max_mel_resolution = int(math.ceil(self.out_channels/min_mel_bins))
        self.mel_resolution = min(mel_resolution, max_mel_resolution)

    def _init_bands(self):
        bands = torch.empty(self.out_channels, self.in_channels)
        if self.filter_init == 'lognorm':
            lowcut_bands = self._lognorm_init(bands, scale=.25)
            bandwidths = self._lognorm_init(bands, scale=.1)
        elif self.filter_init == 'mel' or self.filter_init == 'multi_resolution_mel':
            lowcut_bands, bandwidths = self._mel_init(
                n_filter = self.out_channels, 
                n_repeat = self.in_channels,
                mel_resolution = self.mel_resolution)
        else:
            lowcut_bands = torch.rand_like(bands)
            bandwidths = torch.rand_like(bands)

        if self.learnable_bands:
            self.lowcut_bands = nn.Parameter(lowcut_bands)
            self.bandwidths = nn.Parameter(bandwidths)
        else:
            self.register_buffer('lowcut_bands', lowcut_bands)
            self.register_buffer('bandwidths', bandwidths)

    def _lognorm_init(self, x: Tensor, mean=.1, std=.4, scale=.25):
        return x.log_normal_(mean=mean, std=std)*scale

    def _mel_init(self, n_filter, n_repeat, mel_resolution):
        low_bands, bandwidths = np.array([]), np.array([])
        _n_filter = n_filter // mel_resolution
        for i in range(mel_resolution):
            downsample_factor = 2**i
            start_low_hz = self.min_low_hz / downsample_factor
            delta_hz = self.min_band_hz / downsample_factor
            new_sr = self.sample_rate / downsample_factor
            end_low_hz = new_sr / 2 - (start_low_hz + delta_hz)
            mel = np.linspace(to_mel(start_low_hz), 
                            to_mel(end_low_hz), _n_filter + 1)
            _low_bands = to_hz(mel) / new_sr
            _bandwidths = np.diff(_low_bands)
            low_bands = np.concatenate([low_bands, _low_bands[:_n_filter]])
            bandwidths = np.concatenate([bandwidths, _bandwidths[:_n_filter]])
        low_bands = repeat(
            torch.tensor(low_bands, dtype=self.dtype), 
            'h -> h c', 
            c=n_repeat).contiguous()
        bandwidths = repeat(
            torch.tensor(bandwidths, dtype=self.dtype), 
            'h -> h c', 
            c=n_repeat).contiguous()
        return low_bands, bandwidths

    
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

    def _sinc_init_filters(self):
        n = self.kernel_size / 2
        filter_time = repeat(
            torch.arange(-n, n), 
            'k -> c k', 
            c=self.in_channels).contiguous() #* self.sample_rate/2 
        self.register_buffer('filter_time', filter_time)

    def _sinc_generate_filters(self):
        """Generate FIR filters in time domain using the sinc method.
        """
        highcut_bands = self.lowcut_bands.abs() + self.bandwidths.abs()
        high_time = torch.einsum(
            'hc,ck->hck', 
            highcut_bands, self.filter_time)
        self.filters = torch.einsum(
            'hc,hck->hck', 
            2 * highcut_bands, 
            torch.sinc(2 * high_time))
        low_time = torch.einsum(
            'hc,ck->hck', 
            self.lowcut_bands.abs(), self.filter_time)
        self.filters = self.filters - torch.einsum(
            'hc,hck->hck', 
            2 * self.lowcut_bands.abs(), 
            torch.sinc(2 * low_time))
        filters_max = reduce(
            self.filters, 'b c l -> b c ()', 'max')
        self.filters = self.windows * self.filters #/ filters_max
    
    
    def _firwin_init_filters(self):
        mesh_freq, shift_freq = firwin_mesh(self.kernel_size)
        self.register_buffer("mesh_freq", mesh_freq)
        self.register_buffer("shift_freq", shift_freq)
        self.mesh1 = torch.empty_like(self.mesh_freq, requires_grad=True)
        self.mesh2 = torch.empty_like(self.mesh_freq, requires_grad=True)
        self.filters = torch.rand(
            self.out_channels, self.in_channels, self.kernel_size, 
            dtype=self.dtype, requires_grad=True)
    
    def _firwin_generate_filters(self):
        """First, construct the filter in the frequency domain 
            then transformed back to the time domain using iFFT.
        """
        # interpolate the desired filter coefficients in freq domain into the freq mesh
        # example: mesh [0. .25 .5 .75 1.], low1=.1 low2=.6 => [0. 1. 1. 0. 0.]
        m = self.mesh_freq.shape[-1]
        self.mesh1 = repeat(
            self.lowcut_bands, 
            'h c -> h c m', m=m).contiguous().to(self.device)
        self.mesh1 = self.mesh_freq - self.mesh1.abs()
        self.mesh2 = self.mesh1 - self.bandwidths.abs()[..., None]
        self.mesh1 = logical_switch(self.mesh1)
        self.mesh2 = 1 - logical_switch(self.mesh2)
        
        # bring the firwin to time domain & multiply with the window 
        # hcm,hcm,m -> hcm
        self.filters = torch.fft.irfft(
            self.mesh1 * self.mesh2 * self.shift_freq,
            n=self.n_fft)[..., :self.kernel_size]
        # (hck,hck -> hck) if shared_window=False, otherwise (hck,k -> hck)
        self.filters = self.filters.abs() * self.windows 
        self.filters = torch.fft.ifftshift(
            self.filters).type(self.dtype)

    def _generate_filter(self):
        if self.filter_type == 'sinc':
            self._sinc_generate_filters()
        else:
            self._firwin_generate_filters()

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.

        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.

        """
        self.device = waveforms.device
        if self.learnable_windows:
            self._generate_learnable_windows()

        self._generate_filter()
        waveforms = self.conv_layer(waveforms, self.filters)
        return waveforms

