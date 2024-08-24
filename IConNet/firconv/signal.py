from typing import Optional
import torch
from torch import Tensor
import math
import numpy as np

DEFAULT_SAMPLE_RATE = 16000

def nextpow2(x):
    """
    Example: nextpow2(6) = 3 since 2^3 = 8 is the nearest to 6
    """
    if x <= 0:
        return 1
    return int(math.ceil(math.log2(x)))

def get_nfft(
        kernel_size: int, 
        n_fft: Optional[int]=None,
        requires_gt_kernel: bool=False) -> int:
    """
    Arguments:
        `kernel_size`: int, convolution layer's kernel_size
        `n_fft`: optional, for condition validation
        `requires_gt_kernel`: ensure `n_fft` is greater than the `kernel_size`,
                use when generating firwin mesh.

    Returns: 
        `n_fft` as a power of 2 for optimal cuFFT computation.
    """
    assert kernel_size >= 1
    if requires_gt_kernel:
        kernel_size += 1
    if n_fft is None:
        n_fft = 2**nextpow2(kernel_size)
    else: # validating
        assert n_fft > kernel_size
        assert n_fft == 2**nextpow2(n_fft)
    return n_fft

def absolute_normalization(arr):
    if type(arr) == Tensor:
        return torch.abs(arr / torch.abs(arr).max())
    return np.abs(arr / np.abs(arr).max())

def magnitude_to_db(x, eps=1e-10):
    if type(x) == Tensor:
        return 20 * torch.log10(torch.maximum(x, torch.tensor(eps)))
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
        symmeric = M % 2 == 1
        constant = 2 * math.pi / ((M - 1) if symmeric else M)
        end = (M - 1) * constant
        return end

def get_window_freq_response(window, n_cycles=10):
    length = len(window)
    A = np.fft.fft(window, length*n_cycles)
    response = absolute_normalization(A)
    response = magnitude_to_db(response) 
    return response

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

def firwin_mesh(
        kernel_size: int, 
        n_fft: Optional[int]=None, 
        fs: int=2, out_type=Tensor):
    """
    kernel_size: ideally (2**a - 1) so that nfreqs is 2**a
    Returns:
        mesh_freq: (out_channels, in_channels, mesh_length) or (H C M).
            A uniform mesh in the frequency-domain with length M > kernel_size.
        shift_freq: (H C M). To adjust the phases of the coefficients so that the first
            coefficient of the inverse FFT are the desired filter coefficients.

    """
    nyq = fs/2
    if n_fft is None:
        n_fft = get_nfft(kernel_size, requires_gt_kernel=True)
    else:
        assert n_fft > kernel_size
        assert n_fft == 2**nextpow2(n_fft)
    
    mesh_freq = np.linspace(0.0, nyq, n_fft)
    shift_freq = np.exp(-(kernel_size - 1) / 2. * 1.j * np.pi * mesh_freq / nyq)
    if out_type == Tensor:
        mesh_freq, shift_freq = torch.tensor(mesh_freq), torch.tensor(shift_freq)
    return mesh_freq, shift_freq

def firwin(window_length, band_max, band_min=0,
           window_params=[0.5,0.5], fs=2) -> Tensor:
    """
    FIR filter design using the window method.
    Ref: scipy.signal.firwin2

    Linearly interpolate the desired response on a uniform mesh `x`.
    Similar to `np.interp(x, [0, band_min, band_max, 1], [0, 1, 1, 0])`.
    Note: `torch.where` and `torch.logical_*` ... are not supported by autograd. 
    """
    x, shift = firwin_mesh(window_length, fs)
    fx = torch.where((x <= band_max) & (x >= band_min), 1., 0.) 
    fx2 = fx * shift
    firwin_time = torch.fft.irfft(fx2)
    window = general_cosine_window(window_length, window_params)
    out = firwin_time[:window_length] * window
    out = torch.fft.ifftshift(out)
    return out
