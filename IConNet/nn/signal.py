import torch
import torch.nn.functional as F
from torch import Tensor
import math
from einops import rearrange, reduce
import numpy as np
from ..fftconv import fft_conv_complex as fft_conv

DEFAULT_SAMPLE_RATE = 16000

def nextpow2(x):
    """
    Example: nextpow2(6) = 3 since 2^3 = 8 is the nearest to 6
    """
    if x <= 0:
        return 1
    return int(math.ceil(math.log2(x)))

def absolute_normalization(arr):
    if type(arr) == Tensor:
        return torch.abs(arr / torch.abs(arr).max())
    return np.abs(arr / np.abs(arr).max())

def magnitude_to_db(x, eps=1e-10):
    if type(x) == Tensor:
        return 20 * torch.log10(torch.maximum(x, eps))
    return 20 * np.log10(np.maximum(x, eps))

def to_mel(hz):
    if type(hz) == Tensor:
        return 2595 * torch.log10(1 + hz / 700)
    return 2595 * np.log10(1 + hz / 700)

def to_hz(mel):
    if type(mel) == Tensor:
        return 700 * (torch.pow(10, mel / 2595) - 1)
    return 700 * (np.power(10, mel / 2595) - 1)

def get_last_window_time_point(M, symmeric: bool=True):
        """Handle both odd & even kernel size"""
        constant = 2 * math.pi / ((M - 1) if symmeric else M)
        end = (M - 1) * constant
        return end

def general_cosine_window(n=32, arr=[.5,.5], out_type=Tensor):
    """
    Ref: torch/signal/windows/windows.py#L643
    """
    k = np.linspace(0, 2*np.pi, n)
    a_i = np.array([(-1) ** i * w for i, w in enumerate(arr)])
    i = np.arange(a_i.shape[0])
    win = (a_i[:,None] * np.cos(i[:,None] * k)).sum(0)
    if type(arr) == Tensor or out_type == Tensor:
        win = torch.tensor(win)
    return win

def firwin_mesh(kernel_size, fs=2, out_type=Tensor):
    """
    Returns:
        mesh_freq: (out_channels, in_channels, mesh_length) or (H C M).
            A uniform mesh in the frequency-domain with length M > kernel_size.
        shift_freq: (H C M). To adjust the phases of the coefficients so that the first
            coefficient of the inverse FFT are the desired filter coefficients.
    """
    nyq = fs/2
    nfreqs = 1 + 2 ** nextpow2(kernel_size)
    mesh_freq = np.linspace(0.0, nyq, nfreqs)
    shift_freq = np.exp(-(kernel_size - 1) / 2. * 1.j * np.pi * mesh_freq / nyq)
    if out_type == Tensor:
        mesh_freq, shift_freq = torch.tensor(mesh_freq), torch.tensor(shift_freq)
    return mesh_freq, shift_freq

def firwin(window_length, band_max, band_min=0,
           window_params=[0.5,0.5], fs=2) -> Tensor:
    """
    FIR filter design using the window method.
    Ref: scipy.signal.firwin2
    """
    x, shift = firwin_mesh(window_length, fs)

    # Linearly interpolate the desired response on a uniform mesh `x`.
    # Similar to np.interp(x, [0, band_min, band_max, 1], [0, 1, 1, 0])
    # Note: torch.where, torch.logical_* ... are not supported by autograd
    fx = torch.where((x <= band_max) & (x >= band_min), 1., 0.) 
    fx2 = fx * shift
    firwin_time = torch.fft.irfft(fx2)
    window = general_cosine_window(window_length, window_params)
    out = firwin_time[:window_length] * window
    out = torch.fft.ifftshift(out)
    return out

def downsample_by_n(x, filter, n, band_offset=0, band_cutoff=1) -> Tensor:
    p = n - x.shape[-1] % n
    padding = (0,p)
    x = F.pad(x, padding)
    x = fft_conv(x, filter, stride=n, 
                 band_offset=band_offset, band_cutoff=band_cutoff)
    return x
    
def downsample_wave(x, orig_freq=44500, new_freq=4450, rolloff=0.99,
                   filter_width=4000, orig_freq_offset=0, 
                   orig_freq_min=0, orig_freq_max=None) -> Tensor:
    if orig_freq <= new_freq:
        return x
    
    assert new_freq >= 1
    assert orig_freq % new_freq == 0

    band_offset = orig_freq_offset / orig_freq
    band_min = orig_freq_min / orig_freq
    downsample_factor = orig_freq // new_freq
    band_max = 1 / downsample_factor
    
    if orig_freq_max:
        band_cutoff = orig_freq_max / orig_freq
        band_max = min(band_max, band_cutoff)
    else:
        band_cutoff = 1
    
    orig_freq = orig_freq // downsample_factor
    new_freq = 1
    
    downsample_filter = bandpass_filter(
        band_max=band_max,
        filter_width=filter_width,
        transition_bandwidth=1-rolloff,
        band_min=band_min
    )
    downsample_filter = rearrange(downsample_filter, 'n -> 1 1 n')
    x = downsample_by_n(x, downsample_filter, downsample_factor, 
                        band_offset, band_cutoff)
    return x


def bandpass_filter(band_max=0.5, filter_width=256, 
                   transition_bandwidth=0.03, band_min=0) -> Tensor:
    """
    Calculate the highest frequency we need to preserve 
    and the lowest frequency we allow to pass through.
    Note that frequency is on a scale from 0 to 1 
    where 0 is 0 and 1 is Nyquist frequency of
    the signal BEFORE downsampling.
    """
    band_max = band_max * (1 + transition_bandwidth)
    band_min = band_min * (1 - transition_bandwidth)
    filter = firwin(filter_width, band_max=band_max, band_min=band_min)
    return filter


def get_window_freq_response(window, n_cycles=10):
    length = len(window)
    A = np.fft.fft(window, length*n_cycles)
    response = absolute_normalization(A)
    response = magnitude_to_db(response) 
    return response

    