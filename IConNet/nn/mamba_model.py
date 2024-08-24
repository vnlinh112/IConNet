from typing import Literal
from .frontend import FeBlocks
from .classifier import Classifier
import torch.nn as nn
from einops import rearrange, reduce
from .mamba import MambaBlock

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
        super().__init__()
        
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

class M17(nn.Module):
    """A classification model using FIRConv + Mamba
    """

    def __init__(
            self, 
            config, 
            n_input=None, 
            n_output=None):
        
        super().__init__()
        self.config = config
        if n_input is None:
            n_input = config.n_input
        if n_output is None:
            n_output = config.n_output
        self.n_input = n_input 
        self.n_output = n_output
        self.fe_blocks = FeBlocks(
            n_input_channel = n_input, 
            n_block = config.fe.n_block,
            n_channel = config.fe.n_channel, 
            kernel_size = config.fe.kernel_size, 
            stride = config.fe.stride, 
            window_k = config.fe.window_k,
            residual_connection_type = config.fe.residual_connection_type,
            filter_type = config.fe.filter_type,
            conv_mode=config.fe.conv_mode,
            norm_type=config.fe.norm_type,
            pooling = None) 
        self.fe_n_feature = self.fe_blocks.n_output_channel
        self.seq_blocks = MambaSeq2OneBlocks(
            n_block=config.seq.n_block,
            n_input_channel=self.fe_n_feature, 
            n_output_channel=config.seq.n_channel,
            state_expansion_factor=config.seq.d_state,  
            kernel_size=config.seq.d_conv,    
            block_expansion_factor=config.seq.expand,    
            pooling=self.pooling
        ) 
        self.n_feature = self.seq_blocks.n_output_channel
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim
        )

    def forward(self, x):
        x = self.fe_blocks(x)
        x = self.seq_blocks(x)
        x = self.cls_head(x)
        return x