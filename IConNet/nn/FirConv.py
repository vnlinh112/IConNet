from typing import Literal, Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange, reduce, repeat
import numpy as np
from .signal import firwin_mesh, to_mel, to_hz, get_last_window_time_point, DEFAULT_SAMPLE_RATE
from .activation import logical_switch

class FirConvLayer(nn.Module):
    """Interface for the FIR-kernel-based convolution layer with learnable window function & transition bandwidth.

    Parameters
    ----------
    in_channels :   Number of input channels.
    out_channels :  Number of filters.
    kernel_size :   Filter length. This does not affect the number of learnable parameters.
    stride :        `int`, also downsampling factor. out_length = ceil(in_length / stride)

    layer_mode :    `firwin` (default) or `sinc`.
        - `firwin`: FIR-filter-based convolution layer using windowing method. [1]
        - `sinc`:   Multi-channel Sinc-based convolution. [2]

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
            kernel_size:int, 
            stride: int=1, 
            layer_mode: Literal['firwin', 'sinc']='firwin',

            learnable_bands: bool=True,
            learnable_windows: bool=True,
            shared_window: bool=False,
            window_k: int=2,
            filter_init: Optional[Literal['lognorm', 'mel', 
                                'multi_resolution_mel', 'random', 
                                'none']]='multi_resolution_mel', 
            multi_resolution_count: int=4,
            window_func: Optional[Literal['hann', 'hanning', 
                                'hamming', 'rectangle', 'none']]=None,

            padding=0, dilation=1, 
            bias=False, groups=1,
            min_low_hz=0, min_band_hz=0, 
            
            sample_rate=DEFAULT_SAMPLE_RATE,
            
            dtype=torch.float32, eps=1e-12):

        super().__init__()
        self.in_channels = in_channels
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0: 
            kernel_size += 1
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.layer_mode = layer_mode
        self.padding = padding # should only right pad (0, pad)

        if bias:
            raise ValueError(f'FIRConv does not support bias.')
        if groups > 1:
            raise ValueError(f'FIRConv does not support groups.')
        
        self.bias = bias
        self.groups = groups
        self.dilation = dilation

        self.dtype = dtype
        self.eps = eps
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.learnable_bands = learnable_bands
        self.learnable_windows = learnable_windows
        self.shared_window = shared_window
        self.sample_rate = sample_rate
        
        self.filter_init = filter_init
        if filter_init == 'mel':
            self.multi_resolution_count = 1
        else:
            self.multi_resolution_count = multi_resolution_count

        self.window_func = window_func
        self.window_k = window_k

        self._init_bands()
        self._init_windows()

        self._init_mode()

    def _init_mode(self):
        if self.layer_mode == 'sinc':
            n = self.kernel_size / 2
            filter_time = repeat(
                torch.arange(-n, n), 
                'k -> c k', 
                c=self.in_channels) #* self.fs/2
            self.register_buffer('filter_time', filter_time)
        else: # 'firwin'
            self._firwin_init_filters()

    def _init_bands(self):
        bands = torch.empty(self.out_channels, self.in_channels)
        if self.filter_init == 'lognorm':
            lowcut_bands = self._lognorm_init(bands, scale=.25)
            bandwidths = self._lognorm_init(bands, scale=.1)
        elif self.filter_init == 'mel' or self.filter_init == 'multi_resolution_mel':
            lowcut_bands, bandwidths = self._mel_init(
                n_filter = self.out_channels, 
                n_repeat = self.in_channels)
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

    def _mel_init(self, n_filter, n_repeat):
        low_bands, bandwidths = np.array([]), np.array([])
        _n_filter = n_filter // self.multi_resolution_count
        for i in range(self.multi_resolution_count):
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
        self.filters = self.windows * self.filters / filters_max
    
    
    def _firwin_init_filters(self):
        mesh_freq, shift_freq = firwin_mesh(self.kernel_size)
        self.register_buffer("mesh_freq", mesh_freq)
        self.register_buffer("shift_freq", shift_freq)
        self.mesh1 = torch.empty_like(self.mesh_freq, requires_grad=True)
        self.mesh2 = torch.empty_like(self.mesh_freq, requires_grad=True)
        self.fir_filters = torch.rand(
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
        self.fir_filters = torch.fft.irfft(
            self.mesh1 * self.mesh2 * self.shift_freq, 
            n=self.kernel_size)
        # (hck,hck -> hck) if shared_window=False, otherwise (hck,k -> hck)
        self.fir_filters = self.fir_filters * self.windows 
        self.fir_filters = torch.fft.ifftshift(
            self.fir_filters).type(self.dtype)

    def _generate_filter(self):
        if self.layer_mode == 'sinc':
            self._sinc_generate_filters()
        else:
            self._firwin_generate_filters()

    def _apply_filters(self, X):
        # stride is downsampling factor 
        # TODO: replace stride with strided conv
        if self.layer_mode == 'firwin':
            in_length = X.shape[-1]
            p = self.stride - in_length % self.stride
            X = F.pad(X, (0,p)).to(self.device)
        X = F.conv1d(X, self.fir_filters, 
                     stride=self.stride, 
                     dilation=self.dilation)
        return X

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
        waveforms = self._apply_filters(waveforms)
        return waveforms


class GeneralCosineWindow(nn.Module):
    """
    Use for window parametrization.
    Caution: There might be an issue with torch.einsum when training 
        with multiple GPU (https://github.com/pytorch/pytorch/issues/82308)
    """
    def __init__(self, 
            kernel_size, 
            window_params=[0.5,0.5], 
            dtype=torch.float32):
        super().__init__()
        assert window_params is not None 
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.window_params = torch.tensor(self.window_params, dtype=dtype)
        window_params_dim = len(self.window_params.shape)
        # window_params: (p) or (hcp)
        assert window_params_dim == 1 or window_params_dim == 3 
        self.windows_k = self.window_params.shape[-1]
        self.shared_window = window_params_dim == 1
        
        i = torch.arange(self.window_k, dtype=self.dtype)
        last_time_point = get_last_window_time_point(self.kernel_size)
        A_init = torch.einsum(
                    'p,k->pk',
                    self.i,
                    torch.linspace(0, last_time_point, self.kernel_size))
        self.register_buffer('i', i)
        self.register_buffer('A_init', A_init)

    def _generate_windows(self):
        A = reduce(
                torch.einsum(
                    '...p,p,pk->...pk',
                    self.window_params,
                    torch.cos(self.i * torch.pi), 
                    torch.cos(self.A_init)), 
                '... p k -> ... k', 'sum').contiguous()
        return A

    def right_verse(self, A):
        A = self._generate_windows()
        return A
            
    def forward(self, A):
        A = self._generate_windows()
        return A