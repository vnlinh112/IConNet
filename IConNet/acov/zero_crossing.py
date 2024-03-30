import soundfile as sf
import torch
from torch import Tensor, nn
from einops import rearrange, repeat

def zero_crossings(y: Tensor) -> Tensor:
    """Find the zero-crossings of a signal ``y``: indices ``i`` such that
    ``sign(y[i]) != sign(y[j])`` along the last axis`."""
    y = rearrange(y, '... t n_fft -> t ... n_fft')
    t = y.shape[0]
    z = []
    for i in range(t):
        result = torch.signbit(y[i][...,:-1]) != torch.signbit(y[i][...,1:]) 
        z += [torch.concat([result[...,:1], result])]
    z = rearrange(torch.stack(z, dim=0), 't ... n_fft -> ... t n_fft')
    return z

def zero_crossing_rate(
    y: Tensor,
    n_fft: int = 1024,
    stride: int = 256
) -> Tensor:
    """Compute the zero-crossing rate of an audio time series.

    Returns
    -------
    zcr : np.ndarray [shape=(..., 1, t)]
        ``zcr[..., 0, i]`` is the fraction of zero crossings in frame ``i``
    """
    y = rearrange(y, '... (t n_fft) -> ... t n_fft', n_fft=n_fft)
    crossings = zero_crossings(y)
    zcrate = rearrange(
        nn.functional.avg_pool1d(
            rearrange(crossings.type(torch.float), 
                      '... t n_fft -> ... 1 (t n_fft)'), 
            kernel_size=n_fft, stride=stride),
        '... 1 t -> ... t')
    return zcrate
    
def samples_like(
    X: Tensor,
    sr: float = 16000,
    stride: int = 256,
    n_fft: int = 1024,
    return_time = False
) -> Tensor:
    """Return an array of time values to match the time axis (last axis)."""
    frames = torch.arange(X.shape[-1]) 
    offset = int(n_fft // 2)
    time = (frames * stride + offset) 
    if return_time:
        time = time / float(sr)
    if len(X.shape) == 3:
        b, c, n = X.shape
        time = repeat(time, 't -> b c t', b=b, c=c).contiguous()
    return time


def test(audio_file, quantiles = [0.6,0.8], n_fft=1024):
    y, sr = sf.read(audio_file)
    t = int(len(y) / n_fft)
    zcr = zero_crossing_rate(torch.tensor(y[:int(t*n_fft)]))
    samples = samples_like(zcr)
    if quantiles is None:
        return samples
    q = torch.quantile(
        zcr, q=torch.tensor(quantiles), dim=-1, keepdim=True)
    return samples[(zcr > q[0]) & (zcr < q[1])]
