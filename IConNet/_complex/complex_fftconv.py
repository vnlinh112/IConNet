from typing import Iterable, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
import math
from einops import rearrange

def fft_conv_simple(u: Tensor, v: Tensor, stride: int, band_offset=0.0) -> Tensor:
    L  = u.shape[-1]
    u_f = torch.fft.fft(u, n=L) 
    v_f = torch.fft.fft(v, n=L) 
   
    y_f = torch.einsum('bhl,chl->bchl', u_f, v_f) 

    # TODO: polyphase, stride for each channel?!
    n_fft = y_f.shape[-1]
    if stride is not None and stride > 1:
        down_sample_factor = stride
        p = down_sample_factor - n_fft % down_sample_factor
        y_f = F.pad(y_f, (0, p))
        n_fft_offset = math.floor(band_offset * n_fft)
        n_fft = math.ceil(n_fft/down_sample_factor + n_fft_offset)
        L = math.ceil(L/down_sample_factor)
        y_f = y_f[..., n_fft_offset:n_fft]
    
    y   = torch.fft.irfft(y_f)[..., :L] # (B C H L)
    y   = rearrange(y, 'b c h l -> b h (l c)') # TODO: fix this?!
    return y


def fft_conv_complex(
        u: Tensor, v: Tensor, 
        stride: int=1, groups: int=1,
        band_offset=0.0, band_cutoff=1.0, stretch=1,
        dtype=torch.float32) -> Tensor:
    """
    Recommmended to downsample after conv rather than using strided conv.
    """
    L   = u.shape[-1]
    u_f = torch.fft.fft(u, n=L) # (B H L)
    v_f = torch.fft.fft(v, n=L) # (C H L)
   
    y_f = complex_matmul(u_f, v_f)

    # TODO: polyphase, stride for each channel?!
    n_fft = y_f.shape[-1]
    if stride is not None and stride > 1:
        down_sample_factor = stride
        p = down_sample_factor - n_fft % down_sample_factor
        y_f = F.pad(y_f, (0, p))
        n_fft_offset = max(0, math.ceil(band_offset * n_fft) - 1)
        n_fft_cutoff = math.ceil(band_cutoff * n_fft)
        n_fft2 = math.ceil(n_fft/(down_sample_factor-1))
        n_fft_cut = min(n_fft2 + n_fft_offset, n_fft_cutoff)
        y_f = y_f[..., n_fft_offset:n_fft_cut]
        if y_f.shape[-1] < n_fft2:
            p = n_fft2 - y_f.shape[-1]
            y_f = F.pad(y_f, (0,p))
        L = math.ceil(L/down_sample_factor) - int(L%down_sample_factor>0)    
    
    y   = torch.fft.ifft(y_f,n=L*stretch)[..., :int(L*stretch)] # (B C L)
    y   = y.abs() #.type(dtype)
    return y


def complex_matmul(
        a: Tensor, b: Tensor, 
        groups: int = 1, dtype=torch.complex64) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).

    a = a.type(dtype)
    b = b.type(dtype)

    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.complex(real, imag)
    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int], str] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int], str) If int, Number of zero samples to pad then
            input on the last dimension. If str, "same" supported to pad input for size preservation.
        padding_mode: (str) Padding mode to use from {constant, reflection, replication}.
                      reflection not available for 3d.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
        dilation: (Union[int, Iterable[int]) Dilation rate for the kernel.
        groups: (int) Number of groups for the convolution.

    Returns:
        (Tensor) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    if isinstance(padding, str):
        if padding == "same":
            if stride != 1 or dilation != 1:
                raise ValueError("stride must be 1 for padding='same'.")
            padding_ = [(k - 1) / 2 for k in kernel.shape[2:]]
        else:
            raise ValueError(f"Padding mode {padding} not supported.")
    else:
        padding_ = to_ntuple(padding, n=n)

    # internal dilation offsets
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # Pad the input signal & kernel tensors (round to support even sized convolutions)
    signal_padding = [r(p) for p in padding_[::-1] for r in (math.floor, math.ceil)]
    signal = F.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    if signal.size(-1) % 2 != 0:
        signal = F.pad(signal, [0, 1])

    kernel_padding = [
        pad
        for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.size(i) - kernel.size(i)]
    ]
    padded_kernel = F.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    signal_fr = torch.fft.rfftn(signal.float(), dim=tuple(range(2, signal.ndim)))
    kernel_fr = torch.fft.rfftn(padded_kernel.float(), dim=tuple(range(2, signal.ndim)))

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    # (b, c, nqyst) with nqys = length/2 (unit: samples per cycle)
    if stride_ is not None:
        down_sample_factor = stride_[-1]
        n_fft = output_fr.shape[-1]
        p = down_sample_factor - n_fft % down_sample_factor
        output_fr = F.pad(output_fr, (0, p))
        output_fr = output_fr[..., : n_fft//down_sample_factor]
    
    output = torch.fft.irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # # Remove extra padded values
    # crop_slices = [slice(None), slice(None)] + [
    #     slice(0, (signal_size[i] - kernel.size(i) + 1), stride_[i - 2])
    #     for i in range(2, signal.ndim)
    # ]
    # output = output[crop_slices].contiguous()

    return output
