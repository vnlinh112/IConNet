import torch
from torch import nn
from torch.nn import functional as F
from .signal import *
from einops import rearrange, reduce
import opt_einsum as oe

class PositiveWeight(nn.Module):
    """
    # Example:
    m = Model()
    parametrize.register_parametrization(m, "window_params", PositiveWeight())

    # Combine different parametrization methods:
    import geotorch
    geotorch.sphere(m, "window_params")
    parametrize.register_parametrization(m, "window_params", PositiveWeight())
    """
    def forward(self, X):
        X = X.abs()
        return X

class Clamp(nn.Module):
    def forward(self, X, val_min=0.0, val_max=1.0):
        X = X.clamp(min=val_min, max=val_max)
        return X
    
class ClampSin(nn.Module):
    def forward(self, X):
        return X.sin()

class ClampCos(nn.Module):
    def forward(self, X):
        return X.cos()
    
class ClampTanh(nn.Module):
    def forward(self, X):
        return X.tanh()

class SoftmaxWeight(nn.Module):
    def forward(self, X):
        X = F.softmax(X)
        return X

class PreluWeight(nn.Module):
    def forward(self, X, a=0.2):
        a = torch.tensor(a)
        X = F.prelu(X, a)
        return X
    
class GeneralCosineWindow(nn.Module):
    def __init__(self, window_length):
        super().__init__()
        self.window_length = window_length

    def forward(self, X, params=[0.5,0.5]):
        n = X.shape[-1]
        k = torch.linspace(0, 2*torch.pi, n)
        a_i = torch.tensor([(-1) ** i * w for i, w in enumerate(params)], 
                           dtype=torch.float)[..., None]
        i = torch.arange(a_i.shape[0])[..., None]
        X = torch.tensor((a_i * torch.cos(i * k)).sum(0))        
        return X
    
class FIRWindow(nn.Module):
    def forward(self, W, band_max=1, band_min=0, fs=2):
        """
        FIR filter design using the window method. (Ref: scipy.signal.firwin2)
        W: GeneralCosineWindow with length N in time domain
        turn W into FIR window
        """
        window_length = W.shape[-1]
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
        
        out_full = torch.fft.irfft(fx2)
        W = W * out_full[:window_length]
        return W
    

class FIRWinFilters(nn.Module):
    """FIR filter design using the window method. (Ref: scipy.signal.firwin2)
    Forward steps:
        First, linearly interpolate the desired response on a uniform mesh `x`.
        Then adjust the phases of the coefficients so that the first `ntaps` of the
        inverse FFT are the desired filter coefficients.
    """

    @staticmethod
    def generate_firwin_mesh(window_length, fs=2):
        """Frequency-domain mesh"""
        nyq = fs/2
        nfreqs = 1 + 2 ** nextpow2(window_length)
        mesh_freq = torch.linspace(0.0, nyq, nfreqs) # (out_channels, in_channels, mesh_length) or (H C M)
        shift_freq = torch.exp(-(window_length - 1) / 2. * 1.j * torch.pi * mesh_freq / nyq)
        return mesh_freq, shift_freq
    
    def __init__(self, window_length, fs=2):
        super().__init__()
        self.window_length = window_length
        self.fs = fs
        mesh_freq, shift_freq = self.generate_firwin_mesh(window_length, fs)
        self.register_buffer("mesh_freq", mesh_freq)
        self.register_buffer("shift_freq", shift_freq)
    
    def forward(self, W, bandwidths, lowcut_bands):
        """
        Args:
            W: (out_channels, in_channels, kernel_size) or (H C K). Time-domain windows.
            bandwidths, lowcut_bands:  (out_channels, in_channels) or (H C)
        Returns:
            W: FIR filters
        """        
        mesh1 = self.mesh_freq - lowcut_bands[..., None]
        mesh2 = mesh1 - bandwidths[..., None]
        x_freq = torch.logical_and(mesh1 >=0, mesh2 <= 0).float() # (H C M)
        firwin_freq = oe.contract('hcm,m->hcm', x_freq, self.shift_freq) 
        firwin_time = torch.fft.irfft(firwin_freq)[..., :self.window_length] # (H C K)
        W = firwin_time * W
        return W
    
    # def right_inverse(self, W, bandwidths, lowcut_bands):
    #     return W

    
class FIRFilter(nn.Module):
    """FIR filter design using the sinc method. (Ref: Sincnet)
    """
    def __init__(self, window_length, in_channels, fs=2):
        super().__init__()
        self.window_length = window_length
        self.in_channels = in_channels
        self.fs = fs
        nyq = fs/2 
        n = window_length / 2
        mesh_freq = torch.arange(-n, n).repeat(in_channels,1) * nyq
        self.register_buffer("mesh_freq", mesh_freq)

    # TODO: high, low: PositiveWeight
    def forward(self, X, lowcut_bands, bandwidths):
        """
        Args:
            lowcut_bands, bandwidths: (out_channels, in_channels, 1) or (H C 1)
        Returns:
            X: filters (out_channels, in_channels, kernel_size) or (H C K)
        """
        f_times_t = oe.contract('hc,ck->hck', lowcut_bands, self.mesh_freq)
        low_pass1 = oe.contract('hcc,hck->hck', 
                                2 * lowcut_bands, torch.sinc(2 * f_times_t))
        highcut_bands = lowcut_bands + bandwidths
        f_times_t = oe.contract('hc,ck->hck', highcut_bands, self.mesh_freq)
        low_pass2 = oe.contract('hcc,hck->hck', 
                                2 * highcut_bands, torch.sinc(2 * f_times_t))
        band_pass = low_pass2 - low_pass1
        band_pass = band_pass / reduce(band_pass, 'b c l -> b c ()', 'max')
        X2 = oe.contract('hck,hck->hck', band_pass, X) # output filters
        return X2
    