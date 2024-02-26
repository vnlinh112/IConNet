from typing import Literal, Optional, Union, Callable
from torch import Tensor, nn

class CustomNormLayer(nn.Module):
    def __init__(
            self, 
            name: Literal[
                'BatchNorm',
                'LayerNorm',
                'InstanceNorm',
                'GroupNorm',
                'LocalResponseNorm'
            ],
            num_channels: int,
            n_local_size: int=2,
            num_groups: int=2):
        super().__init__()
        self.name = name
        self.num_channels = num_channels
        self.size = n_local_size
        self.num_groups = num_groups

        if self.num_channels == 0: # for downsample module only
            self.layer = None
        else:
            if name == 'BatchNorm':
                self.layer = nn.BatchNorm1d(
                    num_features=num_channels)
            elif name == 'InstanceNorm':
                self.layer = nn.InstanceNorm1d(
                    num_features=num_channels)
            elif name == 'GroupNorm':
                self.layer = nn.GroupNorm(
                    num_groups=num_groups, 
                    num_channels=num_channels)
            elif name == 'LocalResponseNorm':
                self.layer = nn.LocalResponseNorm(
                    size=n_local_size)   
            elif name == 'LayerNorm':
                self.layer = nn.GroupNorm(
                    num_groups=1, 
                    num_channels=num_channels)
            else:
                self.layer = None
    
    def forward(self, x: Tensor):
        if self.layer is not None:
            x = self.layer(x)
        return x