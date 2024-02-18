import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce
import numpy as np
from .complex_fftconv import fft_conv_complex as fft_conv
from ..firconv.signal import firwin

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



    