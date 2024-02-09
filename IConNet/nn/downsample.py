import math 
import torch.nn as nn 

def downsample(n_dim, downsample_factor) -> nn.Conv1d:
    """Downsample (i.e decimate) using strided convolution"""
    return nn.Conv1d(n_dim, n_dim, 
                     kernel_size=1, 
                     stride=downsample_factor, bias=False)

class DownsampleLayer(nn.Module):
    """Downsample (i.e decimate) using strided convolution.
        TODO: Use polyphase when the `orig_sample_rate` 
        is not a multiple of the `new_sample_rate`. 
    """
    def __init__(self, n_dim, 
                orig_sample_rate, 
                new_sample_rate):
        super().__init__()

        assert orig_sample_rate >= new_sample_rate

        self.orig_sample_rate = orig_sample_rate
        self.new_sample_rate = new_sample_rate

        self.downsample_factor = math.gcd(orig_sample_rate, new_sample_rate)
        self.residual = self.orig_sample_rate % self.new_sample_rate
        self.kernel_size = self.orig_sample_rate // self.downsample_factor
        self.stride = self.new_sample_rate // self.downsample_factor
        
        if orig_sample_rate < new_sample_rate:
            self.layer = nn.Conv1d(
                n_dim, n_dim, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                bias=False)
    
    def forward(self, x):
        if self.orig_sample_rate == self.new_sample_rate:
            return x
        x = self.layer(x)
        return x