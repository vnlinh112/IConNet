from .frontend import FeBlocks
from .sequential import MambaSeq2OneBlocks
from .classifier import Classifier
import torch.nn as nn

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