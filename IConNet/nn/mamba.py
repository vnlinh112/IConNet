from typing import Literal
import torch.nn as nn
from einops import rearrange, reduce
from mamba_ssm import Mamba

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.15) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout))
    
    def forward(self, x):
        return self.ffn(x)

class MambaBlock(nn.Module):
    """
    Arguments:
        d_model:    Model dimension d_model
        d_state:    SSM state expansion factor
        d_conv:     Local convolution width
        expand:     Block expansion factor
        
    num_parameters = 3 * expand * d_model^2
    forward:        B N C -> B N C
    """
    def __init__(
            self, 
            d_model, 
            d_conv=4, 
            d_state=2, 
            expand=2) -> None:
        super().__init__()
        self.sa_head = Mamba( 
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand)
        self.ffn = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    