from typing import Literal, Optional, Union
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from .local_filter import SpeechSegmentSelector

from einops import reduce, rearrange, repeat
from .loss import AudioOverlapInformationLoss

import collections
from .audio_vqvae import VqLoss, VectorQuantizer, AudioVqVae

VqClsLoss = collections.namedtuple(
    "VqClsLoss", ["perplexity", "loss_vq", "loss_cls"])

AudioVQMixClsLoss = collections.namedtuple(
    "AudioVQMixClsLoss", [
        "wave_perplexity", "wave_loss_vq", 
        "mfcc_perplexity", "mfcc_loss_vq", "loss_cls"])

class AudioVQ(nn.Module):
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            commitment_cost: float=0.25,
            distance_type: Literal['euclidean', 'dot']='euclidean'
        ):
        super().__init__()
        
        self.embedding_dim = embedding_dim 
        self.distance_type = distance_type
        
        self.embedding = VectorQuantizer(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost,
            distance_type=distance_type
        )

        self.num_embeddings = num_embeddings 

    def forward(self, X: Tensor) -> tuple[Tensor, VqClsLoss]:
        _, encodings, vq_loss = self.embedding(X)
        zero_loss = torch.tensor(0., device=X.device)
        loss = VqClsLoss(*vq_loss, loss_cls=zero_loss)
        return encodings, loss


class AudioVQMix(AudioVQ):
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            commitment_cost: float=0.25,
            distance_type: Literal['euclidean', 'dot']='euclidean'
        ):
        super().__init__(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost,
            distance_type=distance_type
        )
        
        self.mfcc_embedding = VectorQuantizer(
            num_embeddings=num_embeddings//4, 
            embedding_dim=38, 
            commitment_cost=commitment_cost,
            distance_type=distance_type
        )

        self.num_embeddings = self.embedding.num_embeddings + self.mfcc_embedding.num_embeddings

    def transform_mfcc(self, X: Tensor) -> Tensor:
        X_mfcc = torch.fft.rfft(self.wave_embedding.transform_freq(X)).real[..., 2:40]
        X_mfcc = self.mfcc_embedding.norm_fn(X_mfcc)
        return X_mfcc

    def forward(self, X: Tensor) -> tuple[Tensor, AudioVQMixClsLoss]:
        _, wave_encoding_indices, wave_loss = self.embedding(X)
        X_mfcc = self.transform_mfcc(X)
        _, mfcc_encoding_indices, mfcc_loss = self.mfcc_embedding(X_mfcc)
        loss = AudioVQMixClsLoss(
            *wave_loss, *mfcc_loss, None
        )
        mfcc_encoding_indices = mfcc_encoding_indices + self.embedding.num_embeddings
        encodings = rearrange(
            torch.stack([wave_encoding_indices, mfcc_encoding_indices], dim=1),
            'b c n -> b (n c)')
        return encodings, loss


class AudioVQEncoder(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            num_embeddings: int, 
            embedding_dim: int, 
            kernel_size: int=1023,
            stride: int=128,
            sample_rate: int=16000,
            num_tokens_per_second: int=64,
            max_num_tokens: int=768, 
            commitment_cost: float=0.25,
            distance_type: Literal['euclidean', 'dot']='euclidean',
            codebook_type: Literal['wave', 'wave_mfcc']='wave',
            positional_encoding_type: Optional[Literal[
                'sample', 'frame', 'learnable']]='sample',
            loss_type: Literal['overlap', 'minami', 
                            'maxami', 'mami']='maxami',
        ):
        super().__init__()

        self.in_channels = in_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.kernel_size = embedding_dim
        self.stride = stride
        self.sample_rate = sample_rate
        self.max_num_tokens = max_num_tokens
        self.distance_type = distance_type
        self.codebook_type = codebook_type
        self.positional_encoding_type = positional_encoding_type
        self.num_tokens_per_second = num_tokens_per_second

        self.tokenizer = SpeechSegmentSelector(
            in_channels=in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            max_num_tokens=max_num_tokens
        )
        if codebook_type=='wave_mfcc':
            self.vq = AudioVQMix(
                num_embeddings=num_embeddings, 
                embedding_dim=embedding_dim, 
                distance_type=distance_type,
                commitment_cost=commitment_cost
            )
        else:
            self.vq = AudioVQ(
                num_embeddings=num_embeddings, 
                embedding_dim=embedding_dim, 
                distance_type=distance_type,
                commitment_cost=commitment_cost
            )
        self.codebook_size = self.vq.num_embeddings

        self.embedding = nn.Embedding(
            num_embeddings=self.codebook_size, 
            embedding_dim=self.embedding_dim,
            padding_idx=0)
        self.embedding.weight.data.uniform_(
            -1/self.num_embeddings, 1/self.num_embeddings)
        
        i = torch.arange(self.embedding_dim//2)
        t = 1/10000**(2*i / self.embedding_dim)[None, :]
        self.register_buffer("positional_encoding_mesh", t)

        if positional_encoding_type == 'learnable':
            self.positional_embedding = nn.Embedding(
                num_embeddings=max_num_tokens, 
                embedding_dim=kernel_size)
            
        self.positional_stride = self.sample_rate // self.num_tokens_per_second
        self.max_audio_length = self.max_num_tokens * self.positional_stride

    def generate_positional_encoding(self, num_tokens):
        """
        Return:
            pe: (seq_len, d_model) sinusoid position encoding matrix [Vaswani et al,.]
            .. pe[pos_id, 2i] =  sin(pos_id/10000^(2i / d_model)) 
            .. pe[pos_id, 2i+1] =  cos(pos_id/10000^(2i / d_model)) 
        """
        if self.positional_encoding_type is None:
            return 0
        t = self.positional_encoding_mesh
        if self.positional_encoding_type == 'learnable':
            pos = torch.arange(num_tokens, device=t.device)
            pos = torch.clamp(pos * self.positional_stride, 
                              max=self.max_num_tokens-1)
            pe = self.positional_embedding(pos)
        else:
            pos = torch.arange(num_tokens, device=t.device)[:, None]
            if self.positional_encoding_type=='sample':
                pos = pos*self.stride
            pe = rearrange(
                    torch.stack([torch.sin(pos*t), torch.cos(pos*t)], dim=1), 
                    'p c d -> p (d c)')
        return pe
    
    def encode(self, X: Tensor) -> tuple[Tensor, Union[VqClsLoss, AudioVQMixClsLoss]]:
        tokens = self.tokenizer(X)
        encodings, loss = self.vq(tokens)
        encodings = self.embedding(encodings)
        encodings = encodings + self.generate_positional_encoding(encodings.shape[1])
        return encodings, loss
    
    def forward(self, X: Tensor) -> Tensor:
        encodings, _ = self.encode(X)
        return encodings
    