import torch 
import torch.nn as nn
from einops import rearrange, reduce
from typing import Optional, Literal
from .mamba import MambaBlock

class Seq2SeqBlocks(nn.Module):
    """
    C: n_input_channel
    H: n_output_channel
    N: input sequence length
    (B C N) => (B H N)
    Output: logits
    """
    def __init__(
            self, 
            n_block: int, 
            n_input_channel: int, 
            n_output_channel: int):
        super().__init__()
        self.blocks = nn.LSTM(
                input_size=n_input_channel, 
                hidden_size=n_output_channel, 
                num_layers=n_block, 
                batch_first=True)
    
    def forward(self, x):
        x, _ = self.blocks(
            rearrange(x, 'b c n -> b n c'))
        x = rearrange(x, 'b n h -> b h n')
        return x
    

class Seq2OneBlocks(nn.Module):
    """
    C: n_input_channel
    H: n_output_channel
    N: input sequence length
    (B C N) => (B H N)
    Output: logits
    """
    def __init__(
            self, 
            n_block: int, 
            n_input_channel: int, 
            n_output_channel: int,
            bidirectional: bool = True,
            pooling: Literal[
                'max', 'min',
                'mean', 'sum'] = 'mean',
            use_context: bool = True):
        super().__init__()
        self.blocks = nn.LSTM(
                input_size=n_input_channel, 
                hidden_size=n_output_channel, 
                num_layers=n_block, 
                batch_first=True,
                bidirectional=bidirectional)
        self.n_block = n_block
        self.D = 1 + int(bidirectional)
        self.Cxt = 1 + int(use_context) * 2
        self.bidirectional = bidirectional
        self.pooling = pooling 
        self.use_context = use_context
        self.n_out_feature = n_output_channel * self.D * self.Cxt * n_block
    
    def forward(self, x):
        x, context = self.blocks(
            rearrange(x, 'b c n -> b n c'))
        x = reduce(x, 'b n (d h) -> b (d h)', self.pooling, d=self.D)
        if not self.use_context:
            return x
        hidden_state, cell_state = context
        hidden_state = rearrange(
            hidden_state, '(d l) b h -> b (l d h)', d=self.D, l=self.n_block)
        cell_state = rearrange(
            cell_state, '(d l) b h -> b (l d h)', d=self.D, l=self.n_block
        )
        return torch.cat([x, hidden_state, cell_state], dim=-1)
    
class MambaSeq2OneBlocks(nn.Module):
    """B C N -> B C """
    def __init__(self,
            n_block: int, 
            n_input_channel: int, 
            n_output_channel: int,
            pooling: Literal[
                'max', 'min',
                'mean', 'sum'] = 'mean',
            kernel_size: int=7,
            state_expansion_factor: int=2,
            block_expansion_factor: int=2,
            ):
        self.n_block = n_block
        self.n_input_channel = n_input_channel
        self.n_output_channel = n_output_channel
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.block_expansion_factor = block_expansion_factor
        self.state_expansion_factor = state_expansion_factor
        
        self.input_projection = nn.Linear(
            n_input_channel, n_output_channel)

        self.blocks = nn.Sequential(*[MambaBlock( 
            d_model=self.n_output_channel,
            d_state=self.state_expansion_factor,  
            d_conv=self.kernel_size,   
            expand=self.block_expansion_factor,   
        ) for _ in range(self.n_block)])

    def forward(self, x):
        x = self.input_projection(
            rearrange(x, 'b c n -> b n c'))
        return reduce(
            self.blocks(x),
            'b n c -> b c', self.pooling)