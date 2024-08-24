import soundfile as sf
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from einops import rearrange, repeat, reduce
from typing import Optional, Literal, Union

def zero_crossings(y: Tensor, dtype=None) -> Tensor:
    """Find the zero-crossings of a signal ``y``: indices ``i`` such that
    ``sign(y[i]) != sign(y[j])`` along the last axis`."""
    y = rearrange(y, '... t n_fft -> t ... n_fft')
    t = y.shape[0]
    z = []
    for i in range(t):
        result = torch.signbit(y[i][...,:-1]) != torch.signbit(y[i][...,1:]) 
        z += [torch.concat([result[...,:1], result], dim=-1)]
    z = rearrange(torch.stack(z, dim=0), 't ... n_fft -> ... t n_fft')
    if dtype:
        z = z.type(dtype)
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
        F.avg_pool1d(
            rearrange(crossings.type(torch.float), 
                      '... t n_fft -> ... 1 (t n_fft)'), 
            kernel_size=n_fft, stride=stride),
        '... 1 t -> ... t')
    return zcrate

def zero_crossing_score(    
    y: Tensor,
    n_fft: int = 1024,
    stride: int = 256
) -> Tensor:
    """Compute the zero-crossing score of an audio time series.

    Returns
    -------
    zcr : np.ndarray [shape=(..., 1, t)]
        ``zcr[..., 0, i]`` is the zero crossing score in frame ``i`` 
        with score in range [0,1]
    """
    y = rearrange(y, '... (t n_fft) -> ... t n_fft', n_fft=n_fft)
    crossings = rearrange(zero_crossings(y, dtype=torch.float), 
                      '... t n_fft -> ... (t n_fft)')
    zcrate = rearrange(
        F.avg_pool1d(
            crossings, kernel_size=n_fft, stride=stride),
        '... 1 t -> ... t')
    nonzero_mean = rearrange(
        F.avg_pool1d(
            torch.where(crossings>0, crossings, 0.0), 
            kernel_size=n_fft, stride=stride),
        '... 1 t -> ... t')
    score = F.sigmoid(1 - 8*zcrate + 2*nonzero_mean)
    return score

def zero_crossing_score_chunks(
        y: Tensor, mask_ratio=0.2, score_offset=0.5) -> Tensor:
    """
    If no symmetry constraint: set `mask_ratio=0` and `score_offset=0`
    """
    y /= repeat(reduce(y, 'b n -> b' , 'max'), 'b -> b n', n=1)
    crossings = zero_crossings(y, dtype=torch.float)
    zcrate = crossings.mean(dim=-1)
    length = y.shape[-1]
    mid_q = length//4   # length//16 * 5
    crossings_mid = crossings[:, mid_q:-mid_q]
    nonzero_mean = torch.clamp(crossings, min=0).mean(dim=-1)
    nonzero_mean_mid = torch.clamp(crossings_mid, min=0).mean(dim=-1)
    kernel_size = (length + 1) // 8
    y_pos_avg = F.avg_pool1d(
        torch.clamp(y, min=0), 
        kernel_size=kernel_size, stride=kernel_size)
    emb_data_cond = torch.diff(y_pos_avg) > 0
    cond1 = emb_data_cond[:, 1:3].sum(dim=-1) == 2
    cond2 = emb_data_cond[:, -2:].sum(dim=-1) == 0
    cond3 = y_pos_avg[:, 3] - y_pos_avg[:, -1] > 0
    cond4 = y_pos_avg[:, 4] - y_pos_avg[:, 0] > 0
    emb_data_mask = cond1 * cond2 * cond3 * cond4
    score0 = F.sigmoid(1 - 8*zcrate + nonzero_mean + nonzero_mean_mid)
    score1 = mask_ratio * emb_data_mask
    score = score0 + score1*(1-score0) - score_offset
    return score

def zero_crossing_score_chunks_v2(
        y: Tensor, mask_ratio=0.2, score_offset=0.5) -> Tensor:
    """
    If no symmetry constraint: set `mask_ratio=0` and `score_offset=0`
    """
    y /= repeat(reduce(y, 'b n -> b' , 'max'), 'b -> b n', n=1)
    crossings = zero_crossings(y, dtype=torch.float)
    zcrate = crossings.mean(dim=-1)
    length = y.shape[-1]
    mid_q = length//4   # length//16 * 5
    nonzero_mean = torch.clamp(y, min=0).mean(dim=-1)
    nonzero_mean_mid = torch.clamp(y[:, mid_q:-mid_q], min=0).mean(dim=-1)
    kernel_size = (length + 1) // 8
    y_pos_avg = F.avg_pool1d(
        torch.clamp(y, min=0), 
        kernel_size=kernel_size, stride=kernel_size)
    emb_data_cond = torch.diff(y_pos_avg) > 0
    cond1 = emb_data_cond[:, 1:3].sum(dim=-1) == 2
    cond2 = emb_data_cond[:, -2:].sum(dim=-1) == 0
    cond3 = y_pos_avg[:, 3] - y_pos_avg[:, -1] > 0
    cond4 = y_pos_avg[:, 4] - y_pos_avg[:, 0] > 0
    emb_data_mask = cond1 * cond2 * cond3 * cond4
    score0 = F.sigmoid(1 - 8*zcrate + nonzero_mean + nonzero_mean_mid)
    score1 = mask_ratio * emb_data_mask
    score = score0 + score1*(1-score0) - score_offset
    return score

def zero_crossing_score_chunks_v3(
        y: Tensor, mask_ratio=0.2, score_offset=0.5) -> Tensor:
    """
    If no symmetry constraint: set `mask_ratio=0` and `score_offset=0`
    """
    y /= repeat(reduce(y, 'b n -> b' , 'max'), 'b -> b n', n=1)
    crossings = zero_crossings(y, dtype=torch.float)
    zcrate = crossings.mean(dim=-1)
    length = y.shape[-1]
    y_pos = torch.clamp(y, min=0)
    nonzero_mean = y_pos.mean(dim=-1)
    kernel_size = (length + 1) // 8
    y_pos_avg = F.avg_pool1d(
        y_pos, kernel_size=kernel_size, stride=kernel_size)
    emb_data_cond = torch.diff(y_pos_avg) > 0
    cond1 = emb_data_cond[:, 1:3].sum(dim=-1) == 2
    cond2 = emb_data_cond[:, -2:].sum(dim=-1) == 0
    cond3 = y_pos_avg[:, 3] - y_pos_avg[:, -1] > 0
    cond4 = y_pos_avg[:, 4] - y_pos_avg[:, 0] > 0
    emb_data_mask = cond1 * cond2 * cond3 * cond4
    score_zc = 1 - 8*zcrate + nonzero_mean/5
    score0 = F.sigmoid(score_zc)
    score = score0*emb_data_mask
    return score

    
def samples_like(
    X: Tensor,
    sr: float = 16000,
    stride: int = 256,
    n_fft: int = 1024,
    return_time = False,
    offset: Optional[Union[Literal['auto'], int]] = 'auto',
    max_sample: Optional[int]=None,
    keepdim: bool=True
) -> Tensor:
    """Return an array of time values to match the time axis (last axis)."""
    frames = torch.arange(X.shape[-1], device=X.device) 
    
    if offset is None:
        offset = 0
    elif type(offset)!=int:
        offset = stride
    time = frames * stride + offset
    if max_sample:
        time = torch.where(time > max_sample, max_sample, time)
    if return_time:
        time = time / float(sr)
    if keepdim:
        if len(X.shape) == 3:
            b, c, n = X.shape
            time = repeat(time, 't -> b c t', b=b, c=c).contiguous()
        if len(X.shape) == 2:
            b, n = X.shape
            time = repeat(time, 't -> b t', b=b).contiguous()
    return time

# selected_idx = torch.topk(
#     gumbel_softmax(zero_crossing_score(x), tau=1), 32).indices

def signal_distortion_ratio(
    preds: Tensor, target: Tensor, beta=10) -> Tensor:
    """`Scale-invariant signal-to-distortion ratio`_ (SI-SDR).
    Measuring how good a source sound.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``

    Returns:
        Float tensor with shape ``(...,)`` of SDR values per sample

    -----
    Ref: torchmetrics.functional.audio.sdr
    """
    assert preds.ndim == target.ndim
    assert preds.shape[-1] == target.shape[-1]
    eps = torch.finfo(preds.dtype).eps
    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
    target_scaled = alpha * target
    noise = target_scaled - preds
    val = (torch.sum(target_scaled**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return beta * torch.log10(val)

def signal_loss(preds: Tensor, target: Tensor, beta=0.1) -> Tensor:
    sdr = signal_distortion_ratio(preds, target, beta=-beta)
    return torch.clamp(sdr, min=0)

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
