from typing import Literal, Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from einops import rearrange, reduce, repeat
import opt_einsum as oe
import numpy as np
from .complex_fftconv import fft_conv_complex2 as fft_conv
from ..firconv.signal import firwin_mesh, to_mel, to_hz

DEFAULT_SAMPLE_RATE = 16000

def lognorm_init(x, mean=.1, std=.4, scale=.25):
    return x.log_normal_(mean=mean, std=std)*scale

def mel_init(n_filter, n_repeat, 
            min_low_hz=0, min_band_hz=0, 
            sample_rate=16000, downsample_count=4):
    low_bands, bandwidths = np.array([]), np.array([])
    _n_filter = n_filter // downsample_count
    for i in range(downsample_count):
        downsample_factor = 2**i
        start_low_hz = min_low_hz / downsample_factor
        delta_hz = min_band_hz / downsample_factor
        new_sr = sample_rate / downsample_factor
        end_low_hz = new_sr / 2 - (start_low_hz + delta_hz)
        mel = np.linspace(to_mel(start_low_hz), 
                          to_mel(end_low_hz), _n_filter + 1)
        _low_bands = to_hz(mel) / new_sr
        _bandwidths = np.diff(_low_bands)
        low_bands = np.concatenate([low_bands, _low_bands[:_n_filter]])
        bandwidths = np.concatenate([bandwidths, _bandwidths[:_n_filter]])
    low_bands = repeat(torch.tensor(low_bands, dtype=torch.float), 'h -> h c', c=n_repeat)
    bandwidths = repeat(torch.tensor(bandwidths, dtype=torch.float), 'h -> h c', c=n_repeat)
    return low_bands, bandwidths

class Downsample2(nn.Module):
    """Convolution-based downsample layer with learnable window function & transition bandwidth

    Parameters
    ----------
    in_channels : Number of input channels.
    out_channels :  Number of filters.
    kernel_size : Filter length. This does not affect the number of learnable parameters or output length.
    stride : `int`, also downsampling factor. out_length = ceil(in_length / stride)
    fs : `int`. Sample rate. Default is 2.
    window_func: 'learnable', 'hamming', 'hanning', 'hann' or 'none', optional
        Window function. Default is 'learnable'.
    window_k: `int`, optional
        Number of learnable params for the window. Default is 3.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_cosine.html

    """
    def __init__(
            self, in_channels, out_channels, kernel_size, 
            fs=2, stride=1, padding=0, 
            min_low_hz=0, min_band_hz=0, sample_rate=16000,
            conv_mode: Optional[Literal['conv', 'fftconv']]='fftconv', 
            filter_init: Optional[Literal['lognorm', 'mel']]='mel', 
            window_func: Literal['learnable']='learnable', 
            window_k: int=2, eps=1e-12):

        super().__init__()
        self.in_channels = in_channels
        if kernel_size % 2 == 0: # Forcing the filters to be odd (i.e, perfectly symmetrics)
            kernel_size += 1
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.fs = fs
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.sample_rate = sample_rate
        self.eps = eps
        self.conv_mode = conv_mode
        
        bands = torch.empty(out_channels, in_channels)
        if filter_init == 'lognorm':
            lowcut_bands = lognorm_init(bands, scale=.25)
            bandwidths = lognorm_init(bands, scale=.1)
        elif filter_init == 'mel':
            lowcut_bands, bandwidths = mel_init(
                out_channels, in_channels, min_low_hz, min_band_hz, sample_rate
            )
        else:
            lowcut_bands = torch.rand_like(bands)
            bandwidths = torch.rand_like(bands)

        self.lowcut_bands = nn.Parameter(lowcut_bands)
        self.bandwidths = nn.Parameter(bandwidths)

        self.fir_filters = torch.rand(out_channels, in_channels, kernel_size, 
                                      dtype=torch.float, requires_grad=True)
        mesh_freq, shift_freq = firwin_mesh(kernel_size, fs)
        self.register_buffer("mesh_freq", mesh_freq)
        self.register_buffer("shift_freq", shift_freq)
        self.mesh1 = torch.empty_like(self.mesh_freq, requires_grad=True)
        self.mesh2 = torch.empty_like(self.mesh_freq, requires_grad=True)
        self.x_freq = torch.empty_like(self.mesh_freq, requires_grad=True)
        self.firwin_freq = torch.empty_like(self.mesh_freq, requires_grad=True)
        self.firwin_time = torch.empty_like(self.fir_filters, requires_grad=True)

        # window
        self.window_func = window_func
        # different for each channel
        assert window_k > 0
        self.window_k = window_k
        window_params = repeat(torch.rand(window_k, dtype=torch.float), 'p -> h c p', h=out_channels, c=in_channels)
        self.window_params = nn.Parameter(window_params)

    def forward(self, waveforms, trainable=True):
        device = waveforms.device

        # generate general cosine window from win_params
        k = torch.linspace(0, 2*math.pi, self.kernel_size, 
                           requires_grad=trainable, device=device)
        i = torch.arange(self.window_k, dtype=torch.float, device=device)[..., None]
        self.fir_filters = reduce(self.window_params[..., None] * (-1)**i * torch.cos(i * k), 
                                  'h c p k -> h c k', 'sum').to(device)

        # interpolate the desired filter coefficients in freq domain into the freq mesh
        # example: mesh [0. .25 .5 .75 1.], low1=.1 low2=.6 => [0. 1. 1. 0. 0.]
        m = self.mesh_freq.shape[-1]
        self.mesh1 = repeat(self.lowcut_bands, 'h c -> h c m', m=m)
        self.mesh1 = self.mesh_freq - self.mesh1.abs()
        self.mesh2 = self.mesh1 - self.bandwidths.abs()[..., None]
        self.mesh1 = torch.tanh(100*self.mesh1/(self.mesh1.abs()+self.eps))
        self.mesh2 = 1 - torch.tanh(100*self.mesh2/(self.mesh2.abs()+self.eps))

        self.x_freq = self.mesh1 * self.mesh2 
        self.firwin_freq = oe.contract('hcm,m->hcm', self.x_freq, self.shift_freq) 
        
        # bring the firwin to time domain & multiply with the time-domain window 
        self.firwin_time = torch.fft.irfft(self.firwin_freq, n=self.kernel_size)
        self.fir_filters = oe.contract('hck,hck->hck', self.fir_filters, self.firwin_time)
        self.fir_filters = torch.fft.ifftshift(self.fir_filters).float()

        # stride is downsampling factor 
        in_length = waveforms.shape[-1]
        out_length = math.ceil(in_length / self.stride)
        p = self.stride - in_length % self.stride
        X = F.pad(waveforms, (0,p))

        if self.conv_mode == 'conv':
            X = F.conv1d(X, self.fir_filters, stride=self.stride)
        else:
            X = fft_conv(X, self.fir_filters, stride=self.stride)[..., :out_length]
        return X
    