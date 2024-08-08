from .frontend import FeBlocks
from .sequential import (
    Seq2SeqBlocks, Seq2OneBlocks, Seq2MBlocks)
from .classifier import Classifier, FeedForward
from typing import Literal, Optional, Union
from collections import OrderedDict
from einops import rearrange, reduce
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
from ..utils.config import get_optional_config_value
import torch

class M10(nn.Module):
    """
    A classification model using FIRConv => pooling => FFN
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
            pooling = None) # if pooling here, n_feature=1
        self.pooling = get_optional_config_value(self.config.fe.pooling)
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
            pooling = None) 
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
        x = self.seq_blocks(x)
        x = self.cls_head(reduce(x, 'b h n -> b h', 'max'))
        return x 
    
    def predict(self, X):
        logits = self.forward(X)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs


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
        if 'mel_resolution' in config.fe.keys():
            self.fe_blocks = FeBlocks(
                n_input_channel = n_input, 
                n_block = config.fe.n_block,
                n_channel = config.fe.n_channel, 
                kernel_size = config.fe.kernel_size, 
                stride = config.fe.stride, 
                window_k = config.fe.window_k,
                mel_resolution = config.fe.mel_resolution,
                residual_connection_type = config.fe.residual_connection_type,
                filter_type = config.fe.filter_type,
                conv_mode=config.fe.conv_mode,
                norm_type=config.fe.norm_type,
                pooling = None) # if pooling here, n_feature=1
        else:
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
        self.fe_n_feature = self.fe_blocks.n_output_channel
        self.seq_blocks = Seq2OneBlocks(
            n_block = config.seq.n_block,
            n_input_channel = self.fe_n_feature,
            n_output_channel = config.seq.n_channel,
            pooling = get_optional_config_value(config.seq.pooling),
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
            mel_resolution = config.fe.mel_resolution,
            residual_connection_type = config.fe.residual_connection_type,
            filter_type = config.fe.filter_type,
            conv_mode=config.fe.conv_mode,
            norm_type=config.fe.norm_type,
            pooling = None) # if pooling here, n_feature=1
        self.pooling = get_optional_config_value(self.config.fe.pooling)
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
    

class M13sinc(nn.Module):
    """A classification model using FirConv with Sinc layers
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
            learnable_bands = config.fe.learnable_bands,
            learnable_windows = config.fe.learnable_windows,
            shared_window = config.fe.shared_window,
            window_func = config.fe.window_func,
            mel_resolution = config.fe.mel_resolution,
            conv_mode=config.fe.conv_mode,
            norm_type=config.fe.norm_type,
            pooling = None) # if pooling here, n_feature=1
        self.pooling = get_optional_config_value(self.config.fe.pooling)
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
    

class M18(nn.Module):
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
            mel_resolution = config.fe.mel_resolution,
            residual_connection_type = config.fe.residual_connection_type,
            filter_type = config.fe.filter_type,
            conv_mode=config.fe.conv_mode,
            norm_type=config.fe.norm_type,
            pooling = None) 
        self.fe_n_feature = self.fe_blocks.n_output_channel
        self.seq_blocks = Seq2MBlocks(
            n_block = config.seq.n_block,
            n_input_channel = self.fe_n_feature,
            n_output_channel = config.seq.n_channel,
            use_context = config.seq.use_context,
            bidirectional = config.seq.bidirectional,
            out_seq_length = config.seq.out_seq_length 
        )
        self.n_feature = self.seq_blocks.n_out_feature
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim,
            dropout = config.cls.dropout
        )

    def forward(self, x):
        x = self.fe_blocks(x)
        x = self.seq_blocks(x)
        x = self.cls_head(x)
        return x 
    
    def extract_embedding(self, x):
        x = self.fe_blocks(x)
        x = self.seq_blocks(x)
        logits = self.cls_head.blocks[0](x)
        return logits
    
class M18mfcc(nn.Module):
    """
    A classification model using MFCC + FFN + LSTM with hidden state
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

        
        self.fe_n_feature = self.n_input
        self.seq_blocks = Seq2MBlocks(
            n_block = config.seq.n_block,
            n_input_channel = self.fe_n_feature,
            n_output_channel = config.seq.n_channel,
            use_context = config.seq.use_context,
            bidirectional = config.seq.bidirectional,
            out_seq_length = config.seq.out_seq_length 
        )
        self.n_feature = self.seq_blocks.n_out_feature
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim,
            dropout = config.cls.dropout
        )

    def forward(self, x):
        x = self.seq_blocks(x)
        x = self.cls_head(x)
        return x 
    
    def extract_embedding(self, x):
        x = self.seq_blocks(x)
        logits = self.cls_head.blocks[0](x)
        return logits
    
class M13mfcc(nn.Module):
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
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate = config.mfcc.sample_rate,
            n_mfcc=config.mfcc.n_mfcc,
            melkwargs={
                "n_fft": config.mfcc.n_fft, 
                "hop_length": config.mfcc.hop_length, 
                "n_mels": config.mfcc.n_mels,
                "center": False
                })
        self.pooling = get_optional_config_value(self.config.pooling)
        self.n_feature = config.mfcc.n_mfcc
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim,
            norm_type=config.cls.norm_type
        )

    def forward(self, x):
        x = self.mfcc(x)
        if self.pooling is not None:
            x = reduce(x, 'b 1 c n -> b c', self.pooling)
        else: 
            x = rearrange(x, 'b c n -> b (c n)')
        x = self.cls_head(x)
        return x 
    
class M13mel(nn.Module):
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
        self.mfcc = torchaudio.transforms.MelSpectrogram(
            sample_rate = config.mfcc.sample_rate,
            n_fft = config.mfcc.n_fft,
            hop_length = config.mfcc.hop_length,
            n_mels=config.mfcc.n_mels,
            center=False)
        self.pooling = get_optional_config_value(self.config.pooling)
        self.n_feature = config.mfcc.n_mels
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim,
            norm_type=config.cls.norm_type
        )

    def forward(self, x):
        x = self.mfcc(x)
        if self.pooling is not None:
            x = reduce(x, 'b 1 c n -> b c', self.pooling)
        else: 
            x = rearrange(x, 'b c n -> b (c n)')
        x = self.cls_head(x)
        return x 
    

class M19(nn.Module):
    """A classification model using Sinc layers and 1DConv without residual connection
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
            filter_type = config.fe.filter_type,
            learnable_bands = config.fe.learnable_bands,
            learnable_windows = config.fe.learnable_windows,
            shared_window = config.fe.shared_window,
            window_func = config.fe.window_func,
            conv_mode=config.fe.conv_mode,
            norm_type=config.fe.norm_type,
            residual_connection_type=None,
            pooling = None) # if pooling here, n_feature=1
        self.pooling = get_optional_config_value(self.config.fe.pooling)
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
    

class M20(nn.Module):
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
            filter_type = config.fe.filter_type,
            learnable_bands = config.fe.learnable_bands,
            learnable_windows = config.fe.learnable_windows,
            shared_window = config.fe.shared_window,
            window_func = config.fe.window_func,
            conv_mode=config.fe.conv_mode,
            norm_type=config.fe.norm_type,
            residual_connection_type=None,
            pooling = None) # if pooling here, n_feature=1
        self.fe_n_feature = self.fe_blocks.n_output_channel
        self.seq_blocks = Seq2MBlocks(
            n_block = config.seq.n_block,
            n_input_channel = self.fe_n_feature,
            n_output_channel = config.seq.n_channel,
            use_context = config.seq.use_context,
            bidirectional = config.seq.bidirectional,
            out_seq_length = config.seq.out_seq_length 
        )
        self.n_feature = self.seq_blocks.n_out_feature
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim,
            dropout = config.cls.dropout
        )

    def forward(self, x):
        x = self.fe_blocks(x)
        x = self.seq_blocks(x)
        x = self.cls_head(x)
        return x 
    
    def extract_embedding(self, x):
        x = self.fe_blocks(x)
        x = self.seq_blocks(x)
        logits = self.cls_head.blocks[0](x)
        return logits
    

class M21(nn.Module):
    """A classification model using Sinc layers and 1DConv without residual connection
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
            filter_type = config.fe.filter_type,
            learnable_bands = config.fe.learnable_bands,
            learnable_windows = config.fe.learnable_windows,
            shared_window = config.fe.shared_window,
            window_func = config.fe.window_func,
            conv_mode=config.fe.conv_mode,
            norm_type=config.fe.norm_type,
            residual_connection_type= None,
            pooling = None) # if pooling here, n_feature=1
        self.pooling = get_optional_config_value(self.config.fe.pooling)
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
    

class M22(nn.Module):
    """A classification model using 2 Sinc layers and 1DConv 
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
        self.fe1 = FeBlocks(
            n_input_channel = n_input, 
            n_block = config.fe1.n_block,
            n_channel = config.fe1.n_channel, 
            kernel_size = config.fe1.kernel_size, 
            stride = config.fe1.stride, 
            window_k = config.fe1.window_k,
            filter_type = config.fe1.filter_type,
            learnable_bands = config.fe1.learnable_bands,
            learnable_windows = config.fe1.learnable_windows,
            shared_window = config.fe1.shared_window,
            window_func = config.fe1.window_func,
            conv_mode=config.fe1.conv_mode,
            norm_type=config.fe1.norm_type,
            residual_connection_type= None,
            pooling = None) # if pooling here, n_feature=1
        self.fe2 = FeBlocks(
            n_input_channel = n_input, 
            n_block = config.fe2.n_block,
            n_channel = config.fe2.n_channel, 
            kernel_size = config.fe2.kernel_size, 
            stride = config.fe2.stride, 
            window_k = config.fe2.window_k,
            filter_type = config.fe2.filter_type,
            learnable_bands = config.fe2.learnable_bands,
            learnable_windows = config.fe2.learnable_windows,
            shared_window = config.fe2.shared_window,
            window_func = config.fe2.window_func,
            conv_mode=config.fe2.conv_mode,
            norm_type=config.fe2.norm_type,
            residual_connection_type= None,
            pooling = None) # if pooling here, n_feature=1
        self.pooling = get_optional_config_value(self.config.fe2.pooling)
        self.n_feature = self.fe1.n_output_channel + self.fe2.n_output_channel
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim,
            norm_type=config.cls.norm_type
        )

    def forward(self, x):
        x = self.fe1(x)
        x2 = self.fe2(rearrange(x, 'b c n -> b 1 (n c)'))
        x = reduce(x, 'b c n -> b c', self.pooling)
        x2 = reduce(x2, 'b c n -> b c', self.pooling)
        x = torch.concat([x,x2], dim=-1)
        x = self.cls_head(x)
        return x 
    
class M23(nn.Module):
    """A classification model using 2 Sinc layers with 1DConv + LSTM
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
        self.fe1 = FeBlocks(
            n_input_channel = n_input, 
            n_block = config.fe1.n_block,
            n_channel = config.fe1.n_channel, 
            kernel_size = config.fe1.kernel_size, 
            stride = config.fe1.stride, 
            window_k = config.fe1.window_k,
            filter_type = config.fe1.filter_type,
            learnable_bands = config.fe1.learnable_bands,
            learnable_windows = config.fe1.learnable_windows,
            shared_window = config.fe1.shared_window,
            window_func = config.fe1.window_func,
            conv_mode=config.fe1.conv_mode,
            norm_type=config.fe1.norm_type,
            residual_connection_type= None,
            pooling = None) # if pooling here, n_feature=1
        self.fe2 = FeBlocks(
            n_input_channel = 1, 
            n_block = config.fe2.n_block,
            n_channel = config.fe2.n_channel, 
            kernel_size = config.fe2.kernel_size, 
            stride = config.fe2.stride, 
            window_k = config.fe2.window_k,
            filter_type = config.fe2.filter_type,
            learnable_bands = config.fe2.learnable_bands,
            learnable_windows = config.fe2.learnable_windows,
            shared_window = config.fe2.shared_window,
            window_func = config.fe2.window_func,
            conv_mode=config.fe2.conv_mode,
            norm_type=config.fe2.norm_type,
            residual_connection_type= None,
            pooling = None) # if pooling here, n_feature=1
        self.pooling = get_optional_config_value(self.config.fe2.pooling)
        self.fe_n_feature = 2
        self.seq_blocks = Seq2MBlocks(
            n_block = config.seq.n_block,
            n_input_channel = self.fe_n_feature,
            n_output_channel = config.seq.n_channel,
            use_context = config.seq.use_context,
            bidirectional = config.seq.bidirectional,
            out_seq_length = config.seq.out_seq_length 
        )
        self.n_feature = self.seq_blocks.n_out_feature
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim,
            dropout = config.cls.dropout
        )

    def forward(self, x):
        x1 = reduce(self.fe1(x), 'b c n -> b 1 c', 
                self.config.fe1.pooling)
        x  = rearrange(x, 'b c n -> b 1 (n c)')
        x  = torch.concat([x, x1], dim=-1)
        x  = reduce(self.fe2(x), 'b c n -> b 1 c', 
                self.config.fe2.pooling)
        x  = torch.concat([x1, x], dim=1)
        x  = self.seq_blocks(x)
        x  = self.cls_head(x)
        return x 


class M24(nn.Module):
    """A classification model using 2 Sinc layers with 1DConv + LSTM
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
        self.fe1 = FeBlocks(
            n_input_channel = n_input, 
            n_block = config.fe1.n_block,
            n_channel = config.fe1.n_channel, 
            kernel_size = config.fe1.kernel_size, 
            stride = config.fe1.stride, 
            window_k = config.fe1.window_k,
            filter_type = config.fe1.filter_type,
            learnable_bands = config.fe1.learnable_bands,
            learnable_windows = config.fe1.learnable_windows,
            shared_window = config.fe1.shared_window,
            window_func = config.fe1.window_func,
            conv_mode=config.fe1.conv_mode,
            norm_type=config.fe1.norm_type,
            residual_connection_type= None,
            pooling = None) # if pooling here, n_feature=1
        self.fe2 = FeBlocks(
            n_input_channel = 1, 
            n_block = config.fe2.n_block,
            n_channel = config.fe2.n_channel, 
            kernel_size = config.fe2.kernel_size, 
            stride = config.fe2.stride, 
            window_k = config.fe2.window_k,
            filter_type = config.fe2.filter_type,
            learnable_bands = config.fe2.learnable_bands,
            learnable_windows = config.fe2.learnable_windows,
            shared_window = config.fe2.shared_window,
            window_func = config.fe2.window_func,
            conv_mode=config.fe2.conv_mode,
            norm_type=config.fe2.norm_type,
            residual_connection_type= None,
            pooling = None) # if pooling here, n_feature=1
        self.pooling = get_optional_config_value(self.config.fe2.pooling)
        self.fe_n_feature = 2
        self.seq_blocks = Seq2OneBlocks(
            n_block = config.seq.n_block,
            n_input_channel = self.fe_n_feature,
            n_output_channel = config.seq.n_channel,
            use_context = config.seq.use_context,
            bidirectional = config.seq.bidirectional,
            pooling=self.config.seq.pooling 
        )
        self.n_feature = self.seq_blocks.n_out_feature
        self.cls_head = Classifier(
            n_input = self.n_feature,
            n_output = n_output,
            n_block = config.cls.n_block, 
            n_hidden_dim = config.cls.n_hidden_dim,
            dropout = config.cls.dropout
        )

    def forward(self, x):
        x1 = reduce(self.fe1(x), 'b c n -> b 1 c', 
                self.config.fe1.pooling)
        x  = rearrange(x, 'b c n -> b 1 (n c)')
        x  = torch.concat([x, x1], dim=-1)
        x  = reduce(self.fe2(x), 'b c n -> b 1 c', 
                self.config.fe2.pooling)
        x  = torch.concat([x1, x], dim=1)
        x  = self.seq_blocks(x)
        x  = self.cls_head(x)
        return x 