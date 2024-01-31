from typing import Literal, Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from einops import rearrange, reduce, repeat
import opt_einsum as oe
import numpy as np

import torch.nn.utils.parametrize as parametrize
from .signal import firwin_mesh
from .fftconv import fft_conv_complex2 as fft_conv

class FIRConv(nn.Module):
    """Multi-channel Sinc-based convolution with learnable window function

    Parameters
    ----------
    in_channels : `int`
        Number of input channels.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length. This does not affect the number of learnable parameters.
    sample_rate : `int`, optional
        Sample rate. Default is 16000.
    window_func: 'learnable', 'hamming', 'hanning', 'hann' or 'none', optional
        Window function. Default is 'learnable'.
    window_k: `int`, optional
        Number of learnable params for the window. Default is 3.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_cosine.html

    Usage
    -----
    See `torch.nn.Conv1d`

    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158

    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    @staticmethod
    def get_general_cosine_window(n, params):
        # https://github.com/pytorch/pytorch/blob/main/torch/signal/windows/windows.py#L643
        constant = 2*math.pi / (n-1)
        device = params.device
        k = torch.linspace(0, (n-1)*constant, n, device=device)
        a_i = torch.tensor([(-1) ** i * w for i, w in enumerate(params)], dtype=torch.float, device=device)
        i = torch.arange(a_i.shape[0], device=device)
        return torch.tensor((a_i.unsqueeze(-1) * torch.cos(i.unsqueeze(-1) * k)).sum(0), dtype=torch.float, device=device)

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False, groups=1,
                 sample_rate=16000, min_low_hz=50, min_band_hz=2,
                 window_func: Literal['learnable', 'hamming', 
                                      'hanning', 'hann', 'none']='learnable', 
                 window_k: Literal[2,3,4]=3):

        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size += 1

        self.out_channels = out_channels
        self.window_func = window_func
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError(f'SincConv does not support bias.')
        if groups > 1:
            raise ValueError(f'SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = min_low_hz
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel) / self.sample_rate

        # filter lower frequency (out_channels, in_channels)
        low_hz_ = torch.Tensor(hz[:-1]).repeat(in_channels, 1).view(
                                                  out_channels,in_channels,1)
        self.low_hz_ = nn.Parameter(low_hz_)

        # filter frequency band (out_channels, in_channels)
        band_hz_ = torch.Tensor(torch.diff(hz)).repeat(in_channels, 1).view(
                                                    out_channels,in_channels,1)
        self.band_hz_ = nn.Parameter(band_hz_)

        # window
        if self.window_func == 'hamming':
            self.window_ = torch.hamming_window(self.kernel_size)
        elif self.window_func == 'hanning' or self.window_func == 'hann':
            self.window_ = torch.hanning_window(self.kernel_size)
        elif self.window_func == 'learnable': # same for all channel
            self.window_k = window_k
            self.window_params = nn.Parameter(torch.rand(window_k)) 
            self.window_ = self.get_general_cosine_window(self.kernel_size, 
                                                        self.window_params)
        else: # 'none'
            self.window_ = torch.ones((self.kernel_size))

        # (in_channels, kernel_size)
        n = self.kernel_size / 2
        self.n_ = torch.arange(-n, n).repeat(in_channels,1) / self.sample_rate

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

        self.n_ = self.n_.to(waveforms.device)
        if self.window_func == 'learnable':
            self.window_ = self.get_general_cosine_window(self.kernel_size, self.window_params)
        self.window_ = self.window_.to(waveforms.device)

        # TODO: reimplement this
        low = self.min_low_hz / self.sample_rate + torch.abs(self.low_hz_)
        high = low + self.min_band_hz /self.sample_rate + torch.abs(self.band_hz_)

        # (out, inp, 1) * (1, inp, ker) => (out, inp, ker)
        f_times_t = low * self.n_ * self.sample_rate
        # (out, inp, inp) * (out, inp, ker) => (out, inp, ker)
        low_pass1 = 2 * low * torch.sinc(2 * f_times_t)
        
        f_times_t = high * self.n_ * self.sample_rate
        low_pass2 = 2 * high * torch.sinc(2 * f_times_t)

        band_pass = low_pass2 - low_pass1
        max_, _ = torch.max(band_pass, dim=2, keepdim=True)
        band_pass = band_pass / max_
        
        # (out, inp, ker) * (ker) => (out, inp, ker)
        filters = (band_pass * self.window_) 
        self.filters = filters

        return F.conv1d(waveforms, filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)
    

class Downsample(nn.Module):
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
    def __init__(self, in_channels, out_channels, kernel_size, 
                 fs=2, stride=1, padding=0, 
                 window_func: Optional[Literal['learnable', 'hamming', 
                                      'hanning', 'hann']]='learnable', 
                 window_k: int=2):

        super().__init__()
        self.in_channels = in_channels
        if kernel_size % 2 == 0: # Forcing the filters to be odd (i.e, perfectly symmetrics)
            kernel_size += 1
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.fs = fs
        self.mesh_freq, self.shift_freq = firwin_mesh(kernel_size, fs)  
        self.lowcut_bands = nn.Parameter(torch.rand(out_channels, in_channels))
        self.bandwidths = nn.Parameter(torch.rand(out_channels, in_channels))
        self.fir_filters = torch.rand(out_channels, in_channels, kernel_size, 
                                      requires_grad=True)
        
        # window
        self.window_func = window_func
        if self.window_func == 'hamming':
            self.window = torch.hamming_window(self.kernel_size)
        elif self.window_func == 'hanning' or self.window_func == 'hann':
            self.window = torch.hanning_window(self.kernel_size)
        elif self.window_func == 'learnable': # same for all channel
            assert window_k > 0
            self.window_k = window_k
            self.window_params = nn.Parameter(torch.rand(window_k))
            self.window = torch.rand(self.kernel_size, requires_grad=True)
        else: # 'none'
            self.window = torch.ones((self.kernel_size))

    def forward(self, waveforms, trainable=True):
        device = waveforms.device
        
        self.window_params = self.window_params.to(device)
        self.lowcut_bands = self.lowcut_bands.to(device)
        self.bandwidths = self.bandwidths.to(device)
        self.mesh_freq = self.mesh_freq.to(device)
        self.shift_freq = self.shift_freq.to(device)

        # generate general cosine window from win_params
        k = torch.linspace(0, 2*math.pi, self.kernel_size, 
                           requires_grad=trainable, device=device)
        i = torch.arange(self.window_k, dtype=torch.float, device=device)[..., None]
        self.window = (self.window_params[..., None] * (-1)**i * torch.cos(i * k)).sum(0).to(device)
        # TODO: need to fftshift the window 

        # interpolate the desired filter coefficients in freq domain into the freq mesh
        # example: mesh [0. .25 .5 .75 1.], low1=.1 low2=.6 => [0. 1. 1. 0. 0.]
        self.fir_filters = repeat(self.window, 'k -> h c k', 
                                  h=self.out_channels, c=self.in_channels)
        m = self.mesh_freq.shape[-1]
        self.mesh1 = repeat(self.lowcut_bands, 'h c -> h c m', m=m)
        self.mesh1 = self.mesh_freq - self.mesh1.abs()
        self.mesh2 = self.mesh1 - self.bandwidths.abs()[..., None]

        self.mesh1 = torch.clamp(torch.exp(self.mesh1), min=0., max=1.) # torch.where(mesh1 >= 0., 1., 0.)
        self.mesh2 = torch.clamp(torch.exp(-self.mesh2), min=0., max=1.) # torch.where(mesh2 <= 0., 1., 0.)
        self.x_freq = self.mesh1 * self.mesh2 #  torch.logical_and(mesh1, mesh2).float()
        self.firwin_freq = oe.contract('hcm,m->hcm', self.x_freq, self.shift_freq) 
        # TODO: understand the effect of shift
        
        # bring the firwin to time domain & multiply with the time-domain window 
        self.firwin_time = torch.fft.irfft(self.firwin_freq, n=self.kernel_size) #[..., :self.kernel_size] 
        self.firwin_time = torch.fft.ifftshift(self.firwin_time) # TODO: see if we need fftshift
        self.fir_filters = oe.contract('hck,hck->hck', self.fir_filters, self.firwin_time)

        # stride is downsampling factor 
        in_length = waveforms.shape[-1]
        out_length = math.ceil(in_length / self.stride)
        p = self.stride - in_length % self.stride
        X = F.pad(waveforms, (0,p))
        X = fft_conv(X, self.fir_filters, stride=self.stride)[..., :out_length]
        return X