import torch
from torch import nn
from einops import reduce
from .signal import get_last_window_time_point

class GeneralCosineWindow(nn.Module):
    """
    Use for window parametrization.
    """
    def __init__(self, 
            kernel_size, 
            window_params=[0.5,0.5], 
            dtype=torch.float32):
        super().__init__()
        assert window_params is not None 
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.window_params = torch.tensor(self.window_params, dtype=dtype)
        window_params_dim = len(self.window_params.shape)
        # window_params: (p) or (hcp)
        assert window_params_dim == 1 or window_params_dim == 3 
        self.windows_k = self.window_params.shape[-1]
        self.shared_window = window_params_dim == 1
        
        i = torch.arange(self.window_k, dtype=self.dtype)
        last_time_point = get_last_window_time_point(self.kernel_size)
        A_init = torch.einsum(
                    'p,k->pk',
                    self.i,
                    torch.linspace(0, last_time_point, self.kernel_size))
        self.register_buffer('i', i)
        self.register_buffer('A_init', A_init)

    def _generate_windows(self):
        A = reduce(
                torch.einsum(
                    '...p,p,pk->...pk',
                    self.window_params,
                    torch.cos(self.i * torch.pi), 
                    torch.cos(self.A_init)), 
                '... p k -> ... k', 'sum').contiguous()
        return A

    def right_verse(self, A):
        A = self._generate_windows()
        return A
            
    def forward(self, A):
        A = self._generate_windows()
        return A