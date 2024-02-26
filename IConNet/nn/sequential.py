import torch 
import torch.nn as nn
from einops import rearrange, reduce
from typing import Optional, Literal
from ..conv.pad import PadForConv

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
        # only use cell state (long term memory), not hidden state
        self.Cxt = 1 + int(use_context) * 1
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
        _, cell_state = context
        cell_state = rearrange(
            cell_state, '(d l) b h -> b (l d h)', d=self.D, l=self.n_block
        )
        return torch.cat([x, cell_state], dim=-1)
    

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
            use_context: bool = True,
            mix_pooling: bool = False):
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
        self.Cxt = int(use_context) 
        self.bidirectional = bidirectional
        self.use_context = use_context
        self.out_seq_length = out_seq_length
        self.pooling_fn_list = ['max', 'min', 'mean', torch.var]
        self.n_chunk = self.out_seq_length - self.n_block * self.Cxt 
        self.pad_layer = PadForConv(
                    kernel_size=self.n_chunk,
                    pad_mode='mean')
        p = len(self.pooling_fn_list)
        h = n_output_channel * self.D
        m = self.n_chunk * p + self.Cxt
        self.n_out_feature = h * m
        
    
    def forward(self, x):
        x, context = self.blocks(
            rearrange(x, 'b c n -> b n c'))
        x = self.pad_layer(
            rearrange(x, 'b n dh -> b dh n'))
        y = []
        for pooling_fn in self.pooling_fn_list:
            y += [reduce(x, 'b dh (m t) -> b dh m', pooling_fn, 
                        m=self.n_chunk)]
        y = rearrange(torch.cat(y, dim=-1),
                      'b dh pm -> b pm dh')
        # print(y.shape)
        if not self.use_context:
            return rearrange(y, 'b m dh -> b (dh m)')
        _, c = context
        c = rearrange(c, 
            '(d l) b h -> b l (d h)', 
            d=self.D, l=self.n_block
        )
        # print(c.shape)
        y = torch.cat([y, c], dim=1)
        # print(y.shape)
        y = rearrange(
            y,
            'b m dh -> b (dh m)')
        return y
    