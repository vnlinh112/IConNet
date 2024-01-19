from typing import Literal
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from einops import rearrange, reduce
import opt_einsum as oe
import numpy as np

class WSincConv(nn.Module):
    """Multi-channel Sinc-based convolution with learnable window function

    Parameters
    ----------
    in_channels : `int`
        Number of input channels.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Default is 16000.
    window_func: 'learnable', 'hamming', 'hanning', 'hann' or 'none', optional
        Window function. Default is 'learnable'.
    window_k: `int`, optional
        Number of params for the learnable window. Default is 3.
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
    