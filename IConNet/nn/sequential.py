import torch 
import torch.nn as nn
from einops import rearrange, reduce
from typing import Optional, Literal

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
    

class Seq2MBlocks(nn.Module):
    """
    C: n_input_channel
    H: n_output_channel
    N: input sequence length
    M: output sequence length
    forward: B C N -> B (M H) 
    """
    def __init__(
            self, 
            n_block: int, 
            n_input_channel: int, 
            n_output_channel: int,
            out_seq_length: int,
            bidirectional: bool = True,
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
        # only use cell state (long term memory), not hidden state
        self.Cxt = 1 + int(use_context) * 1 
        self.bidirectional = bidirectional
        self.use_context = use_context
        self.out_seq_length = out_seq_length
        self.n_out_feature = n_output_channel * self.D * out_seq_length
    
    def forward(self, x):
        x, context = self.blocks(
            rearrange(x, 'b c n -> b n c'))
        m = self.out_seq_length
        m_l = m - self.n_block
        x = rearrange(x[:, -m_l:, :], 
                      'b ml (d h) -> b ml (d h)', 
                      d=self.D) 

        if not self.use_context:
            return x
        _, cell_state = context
        cell_state = rearrange(cell_state, 
            '(d l) b h -> b l (d h)', 
            d=self.D, l=self.n_block
        )
        x = rearrange(
            torch.cat([x, cell_state], dim=1),
            'b m dh -> b (m dh)', m=m)
        return x
    