from typing import Literal, Optional, Union, Callable
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from einops import reduce, rearrange, repeat
from ..nn.classifier import FeedForward, FeedForwardAddNorm
from ..nn.mamba_model import MambaSeq2OneBlocks, MambaBlock
from .audio_vqmix import AudioVQEncoder, AudioVQMixClsLoss

from .model import SCB, SCB12

class SCB10(SCB):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_embeddings: int, 
        embedding_dim: int, 
        kernel_size: int=1023,
        stride: int=128,
        max_num_tokens: int=2048, 
        cls_dim: int=500,
        sample_rate: int=16000,
        commitment_cost: float=0.25,
        distance_type: Literal['euclidean', 'dot']='euclidean'
    ):
        super().__init__(
            in_channels=in_channels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            cls_dim=cls_dim,
            num_classes=num_classes,
            sample_rate=sample_rate,
            distance_type=distance_type,
            kernel_size=kernel_size,
            stride=stride,
            commitment_cost=commitment_cost
        )

        self.encoder = AudioVQEncoder(
            in_channels=in_channels,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            kernel_size=kernel_size,
            stride=stride,
            max_num_tokens=max_num_tokens,
            commitment_cost=commitment_cost,
            distance_type=distance_type
        )  

        self.seq_encoder = MambaSeq2OneBlocks(
            n_block=2, 
            n_input_channel=embedding_dim, 
            n_output_channel=embedding_dim,
            pooling='mean',
            kernel_size=4, # only 2 or 4
            state_expansion_factor=2,
            block_expansion_factor=2,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.cls_dim),
            nn.LayerNorm(self.cls_dim),
            FeedForward(self.cls_dim),
            nn.Linear(self.cls_dim, num_classes)
        )

    def train_embedding_cls(
            self, X: Tensor, Y: Optional[Tensor]=None
        ) -> tuple[Tensor, AudioVQMixClsLoss]:
        
        tokens, loss = self.encoder.encode(X)
        Z = self.seq_encoder(rearrange(tokens, 'b n c -> b c n'))
        logits = self.classifier(Z)

        if Y is not None:
            loss_cls = F.cross_entropy(logits.squeeze(), Y)
            loss = loss._replace(loss_cls=loss_cls)
        return logits, loss
    
    def train_ssl(self, X: Tensor):
        pass
    
    def forward(self, X: Tensor) -> Tensor:
        logits, _ = self.train_embedding_cls(X)
        return logits
    

class SCB12Mamba(SCB12):
    def __init__(
        self,
        in_channels: int,    
        num_embeddings: int, 
        embedding_dim: int, 
        commitment_cost: float,
        num_classes: int, 
        stride: int=8,
        cls_dim: int=500,
        sample_rate: int=16000,
        sample_mode: Literal['fixed', 
                            'zero_crossing', 
                            'envelope']='envelope',
        distance_type: Literal['euclidean', 'dot']='euclidean',
        loss_type: Literal['overlap', 'minami', 
                            'maxami', 'mami']='overlap',
        codebook_pretrained_path: Optional[str]=None,
        freeze_codebook: bool=False,
        num_mamba_block: int=1,
        num_tokens_per_second: int=64,
        max_num_tokens: int=768, 
    ):
        super().__init__(
            in_channels=in_channels, 
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            cls_dim=cls_dim, 
            num_classes=num_classes, 
            sample_rate=sample_rate, 
            stride=stride, 
            sample_mode=sample_mode, 
            distance_type=distance_type, 
            loss_type=loss_type,
            commitment_cost=commitment_cost,
            codebook_pretrained_path=codebook_pretrained_path,
            freeze_codebook=freeze_codebook,
            num_mamba_block=num_mamba_block,
            num_tokens_per_second=num_tokens_per_second,
            max_num_tokens=max_num_tokens 
        )

        self.seq_blocks = nn.Sequential(*[MambaBlock(
            d_model=cls_dim, 
            d_conv=4, 
            d_state=2, 
            expand=4
        ) for _ in range(num_mamba_block)])

    def seq_forward(self, X: Tensor) -> Tensor:
        return self.seq_blocks(X)