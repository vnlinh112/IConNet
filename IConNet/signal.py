import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from einops import rearrange, reduce
import numpy as np
import matplotlib.pyplot as plt
from .fftconv import fft_conv_simple as fft_conv

def nextpow2(x):
    """
    Example: nextpow2(6) = 3 since 2^3 = 8 is the nearest to 6
    """
    if x <= 0:
        return 1
    return int(math.ceil(math.log2(x)))

def absolute_normalization(arr):
    return np.abs(arr / np.abs(arr).max())

def magnitude_to_db(x, eps=1e-10):
    return 20 * np.log10(np.maximum(x, eps))

def general_cosine_window(n, arr):
    """
    Ref: https://github.com/pytorch/pytorch/blob/main/torch/signal/windows/windows.py#L643
    """
    k = torch.linspace(0, 2*torch.pi, n)
    a_i = torch.tensor([(-1) ** i * w for i, w in enumerate(arr)])
    i = torch.arange(a_i.shape[0])
    win = (a_i[:,None] * torch.cos(i[:,None] * k)).sum(0)
    return win

def firwin(window_length, band_max, band_min=0,
           window_params=[0.5,0.5], fs=2) -> Tensor:
    """
    FIR filter design using the window method. (Ref: scipy.signal.firwin2)
    """
    nyq = fs/2
    
    nfreqs = 1 + 2 ** nextpow2(window_length)

    # Linearly interpolate the desired response on a uniform mesh `x`.
    x = torch.linspace(0.0, nyq, nfreqs)
    # Similar to np.interp(x, [0, band_min, band_max, 1], [0, 1, 1, 0])
    fx = torch.where((x <= band_max) & (x >= band_min), 1., 0.) 

    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    shift = torch.exp(-(window_length - 1) / 2. * 1.j * torch.pi * x / nyq)
    fx2 = fx * shift

    # Use irfft to compute the inverse FFT.
    out_full = torch.fft.irfft(fx2)
    window = general_cosine_window(window_length, window_params)
    out = out_full[:window_length] * window
    return out

def downsample_by_n(x, filter, n, band_offset=0) -> Tensor:
    p = n - x.shape[-1] % n
    padding = (0,p)
    x = F.pad(x, padding)
    x = fft_conv(x, filter, stride=n, band_offset=band_offset)
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
        band_max = min(band_max, orig_freq_max / orig_freq)
    
    orig_freq = orig_freq // downsample_factor
    new_freq = 1
    
    downsample_filter = bandpass_filter(
        band_max=band_max,
        filter_width=filter_width,
        transition_bandwidth=1-rolloff,
        band_min=band_min
    )
    downsample_filter = rearrange(downsample_filter, 'n -> 1 1 n')
    x = downsample_by_n(x, downsample_filter, downsample_factor, band_offset)
    return x


def bandpass_filter(band_max=0.5, filter_width=256, 
                   transition_bandwidth=0.03, band_min=0) -> Tensor:
    """
    Calculate the highest frequency we need to preserve and the lowest frequency we allow
    to pass through.
    Note that frequency is on a scale from 0 to 1 where 0 is 0 and 1 is Nyquist frequency of
    the signal BEFORE downsampling.
    """
    band_max = band_max * (1 + transition_bandwidth)
    band_min = band_min * (1 - transition_bandwidth)
    filter = firwin(filter_width, band_max=band_max, band_min=band_min)
    return filter


def visualize_waveform(filename="", audio_dir="./", 
        y="", sr=16000, title="", zoom_xlim=[0.05,0.1]):
    if filename:
        import soundfile as sf
        y, sr = sf.read(audio_dir + filename)
    if not title:
        title = filename
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(15, 3))
    ax.set(title='Full waveform: ' + title)
    ax.plot(y)
    
    ax2.set(title='Sample view: ' + title, xlim=np.multiply(zoom_xlim, sr))
    ax2.plot(y, marker='.')

def visualize_window(window, window_name="", f_xlim=None, f_ylim=None, f_xhighlight=-90.2, sr=16000):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
    axes[0].plot(window)
    axes[0].set_title(f"{window_name} window")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Sample")

    length = len(window)
    n_cycles = sr//length
    A = np.fft.fft(window, length*n_cycles)/ (length/2) 
    freq = np.fft.fftfreq(len(A), d=n_cycles/length * 2) 
    response = absolute_normalization(A)
    response = magnitude_to_db(response) 
    if not f_xlim:
        f_xlim = [0,0.5]
    if not f_ylim:
        f_ylim = [-140, 10]
    axes[1].set_xlim(f_xlim)
    axes[1].set_ylim(f_ylim)
    axes[1].plot(freq, response)
    axes[1].set_title("Frequency response of the window")
    axes[1].set_ylabel("Normalized magnitude [dB]")
    axes[1].set_xlabel("Normalized frequency [cycles per sample]")
    axes[1].axhline(f_xhighlight, color='red')
    