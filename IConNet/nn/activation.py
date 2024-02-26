import torch
import torch.nn.functional as F
from torch import Tensor, nn

class NLReLU(nn.Module):
    def __init__(self, beta: float=1.):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return nl_relu(x, self.beta)
        
def nl_relu(x: Tensor, beta: float=1.):
    return torch.log(1 + beta * F.relu(x)) 

def logical_switch(x: Tensor, beta: float=10., eps=1e-12):
    """
    A differentiable version of the following function: 
        torch.where(x  >= 0, 1., 0.)
    Example:
    x = torch.tensor([  -11.,  -2., -0.4,   0.,  0.5,    1.,  9.])
    y = torch.tensor([    0.,   0.,  0.,    0.,  1,      1.,  1.])
    assert torch.allclose(logical_switch(x), y) # true
    """
    return torch.tanh(beta*x/(x.abs()+eps))

class LogicalSwitch(nn.Module):
    def forward(self, X):
        X = logical_switch(X)
        return X

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