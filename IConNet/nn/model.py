from .frontend import FeBlocks
from .sequential import Seq2SeqBlocks, Seq2OneBlocks
from .classifier import Classifier
from typing import Literal, Optional, Union
from einops import rearrange, reduce
import torch.nn as nn
from .activation import NLReLU
from ..utils import config as cfg

class M10(nn.Module):
    """
    A classification model using FIRConv => pooling => FFN
    """

    def __init__(
            self, 
            config: cfg.ModelConfig, 
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
            pooling = None) # if pooling here, n_feature=1
        self.pooling = cfg.get_optional_config_value(self.config.fe.pooling)
        self.n_feature = self.fe_blocks.n_output_channel
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim
        )

    def forward(self, x):
        x = self.fe_blocks(x)
        if self.pooling is not None:
            x = reduce(x, 'b c n -> b c', self.pooling)
        else: 
            x = rearrange(x, 'b c n -> b (c n)')
        x = self.cls_head(x)
        return x 

class M11(nn.Module):
    """
    A classification model using FIRConv + LSTM
    """

    def __init__(
            self, 
            config: cfg.ModelConfig, 
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
            pooling = None) # if pooling here, n_feature=1
        # self.pooling = cfg.get_optional_config_value(self.config.fe.pooling)
        self.n_feature = self.fe_blocks.n_output_channel
        self.seq_blocks = Seq2SeqBlocks(
            n_block=1,
            n_input_channel=self.n_feature,
            n_output_channel=64
        )
        self.cls_head = Classifier(
            n_input = 64,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim
        )

    def forward(self, x):
        x = self.fe_blocks(x)
        # if self.pooling is not None:
            # x = reduce(x, 'b c n -> b 1 c', self.pooling)
        # else: 
        x = self.seq_blocks(x)
        x = self.cls_head(reduce(x, 'b h n -> b h', 'max'))
        return x 


class M12(nn.Module):
    """
    A classification model using FIRConv + LSTM with hidden state
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
            conv_mode=config.fe.conv_mode,
            norm_type=config.fe.norm_type,
            pooling = None) # if pooling here, n_feature=1
        self.fe_n_feature = self.fe_blocks.n_output_channel
        self.seq_blocks = Seq2OneBlocks(
            n_block = config.seq.n_block,
            n_input_channel = self.fe_n_feature,
            n_output_channel = config.seq.n_channel,
            pooling = config.seq.pooling,
            use_context = config.seq.use_context,
            bidirectional = config.seq.bidirectional 
        )
        self.n_feature = self.seq_blocks.n_out_feature
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
    

class M13(nn.Module):
    """
    A classification model using FIRConv with fftconv => pooling => FFN
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
            pooling = None) # if pooling here, n_feature=1
        self.pooling = cfg.get_optional_config_value(self.config.fe.pooling)
        self.n_feature = self.fe_blocks.n_output_channel
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim,
            norm_type=config.cls.norm_type
        )

    def forward(self, x):
        x = self.fe_blocks(x)
        if self.pooling is not None:
            x = reduce(x, 'b c n -> b c', self.pooling)
        else: 
            x = rearrange(x, 'b c n -> b (c n)')
        x = self.cls_head(x)
        return x 