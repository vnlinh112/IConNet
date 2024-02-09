from typing import Literal, Optional, Union
from einops import rearrange, reduce
import torch.nn as nn
from collections import OrderedDict

class Classifier(nn.Module):
    """
    N: n_input (feature)
    M: n_output
    (B 1 N -> B 1 M) or (B N -> B M)
    
    Output: logits
    """
    def __init__(
            self, 
            n_input: int, 
            n_output: int,
            n_block: int,
            n_hidden_dim: Optional[Union[
                tuple[int], list[int]]] = None
            ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_block = n_block
        
        if n_hidden_dim is None:
            n_hidden_dim = [n_input // 2 for i in range(n_block)]
        self.n_hidden_dim = n_hidden_dim

        blocks = [nn.Sequential(OrderedDict({
                "norm": nn.LayerNorm(self.n_input),
                "layer": nn.Linear(self.n_input, n_hidden_dim[0])
            }))]
        blocks += [nn.Sequential(OrderedDict({
                "norm":nn.LayerNorm(n_hidden_dim[i-1]),
                "layer": nn.Linear(n_hidden_dim[i-1], n_hidden_dim[i])
            })) for i in range(1, n_block)] 
        self.blocks = nn.ModuleList(blocks)
        self.act = nn.LeakyReLU()
        self.output_layer = nn.Linear(
            n_hidden_dim[-1], 
            self.n_output)
    
    def forward(self, x):
        assert len(x.shape) == 2 or x.shape[1] == 1
        for block in self.blocks:
            x = block(x)
        x = self.act(x)
        x = self.output_layer(x)
        return x