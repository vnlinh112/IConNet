import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from einops import rearrange, reduce
import numpy as np
import matplotlib.pyplot as plt
from fftconv import fft_conv_s4

nextpow2 = lambda x: int(math.ceil(math.log2(x))) # Example: nextpow2(6) = 3 since 2^3=8 is the nearest to 6

absolute_normalization = lamda x: np.abs(x / np.abs(x).max())

magnitude_to_db = lamda x: 20 * np.log10(np.maximum(x, 1e-10))

def general_cosine_window(n, arr):
    """
    Ref: https://github.com/pytorch/pytorch/blob/main/torch/signal/windows/windows.py#L643
    """
    k = torch.linspace(0, 2*torch.pi, n)
    a_i = torch.tensor([(-1) ** i * w for i, w in enumerate(arr)])
    i = torch.arange(a_i.shape[0])
    win = (a_i[:,None] * torch.cos(i[:,None] * k)).sum(0)
    return win

def firwin(window_length, freq_cutoff, window_params=[0.5,0.5], fs=2) -> Tensor:
    """
    FIR filter design using the window method.
    Ref: scipy.signal.firwin2
    """
    nyq = fs/2
    # assert numtaps % 2 == 1
    
    nfreqs = 1 + 2 ** nextpow2(window_length)

    # Linearly interpolate the desired response on a uniform mesh `x`.
    x = torch.linspace(0.0, nyq, nfreqs)
    fx = torch.where(x < freq_cutoff, 1., 0.) # np.interp(x, [0, fcutoff, 1], [1, 1, 0])

    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    shift = torch.exp(-(window_length - 1) / 2. * 1.j * torch.pi * x / nyq)
    fx2 = fx + shift

    # Use irfft to compute the inverse FFT.
    out_full = torch.fft.irfft(fx2)
    window = general_cosine_window(window_length, window_params)
    out = out_full[:window_length] * window
    return out

def downsample_by_n(x, filterKernel, n) -> Tensor:
    p = n - x.shape[-1] % n
    padding = (0,p)
    x = F.pad(x, padding)
    x = fft_conv_s4(x, filterKernel, stride=n)
    return x
    
def resample_wave(x, orig_freq=44500, new_freq=4450, rolloff=0.99,
                   lowpass_filter_width=4000) -> Tensor:
    downsample_factor = orig_freq // new_freq
    orig_freq = orig_freq // downsample_factor
    new_freq = 1
    
    downsample_filter = lowpass_filter(
        band_center=1 / downsample_factor,
        kernelLength=lowpass_filter_width,
        transitionBandwidth=1-rolloff,
    )
    downsample_filter = rearrange(downsample_filter, 'n -> 1 1 n')
    x = downsample_by_n(
        x, downsample_filter, downsample_factor
    )
    return x


def lowpass_filter(band_center=0.5, kernelLength=256, transitionBandwidth=0.03) -> Tensor:
    """
    Calculate the highest frequency we need to preserve and the lowest frequency we allow
    to pass through.
    Note that frequency is on a scale from 0 to 1 where 0 is 0 and 1 is Nyquist frequency of
    the signal BEFORE downsampling.
    """
    stopbandMin = band_center * (1 + transitionBandwidth)
    filterKernel = firwin(kernelLength, stopbandMin)
    return filterKernel


def visualize_waveform(filename="", audio_dir="./", 
        y="", sr=16000, title="", zoom_xlim=[0.05,0.1]):
    import soundfile as sf
    print(title)
    if filename:
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
    response = magnitude_to_db(response, 1e-10) 
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
    