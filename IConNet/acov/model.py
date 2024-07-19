from typing import Literal
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from ..conv.pad import PadForConv
from .local_filter import LocalPatternFilter, LocalPatterns, FilterEnhance
from .audio_vqvae import AudioVqVae, VqVaeLoss
from .loss import AudioOverlapInformationLoss, AudioMutualInfoMask
from einops import reduce, rearrange, repeat
from ..nn.mamba import FeedForward

class SCB8(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,         
        num_embeddings: int, 
        embedding_dim: int, 
        commitment_cost: float,
        num_classes: int, 
        num_tokens_per_second: int=8, # recommended: 4, 8 or 16
        downsampling: int=8, # recommended: 5 or 8
        cls_dim: int=500,
        sample_rate: int=16000,
        sample_mode: Literal['fixed', 
                            'zero_crossing', 
                            'envelope']='envelope',
        distance_type: Literal['euclidean', 'dot']='euclidean'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.cls_dim = cls_dim
        self.num_classes = num_classes
        self.num_tokens_per_second = num_tokens_per_second
        self.sample_rate = sample_rate
        self.downsampling = downsampling
        self.sample_mode = sample_mode
        self.distance_type = distance_type

        self.vqvae = AudioVqVae(
            in_channels=in_channels,
            out_channels=out_channels,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost,
            sample_rate=sample_rate,
            downsampling = downsampling,
            sample_mode=sample_mode,
            distance_type=distance_type
        )  
        self.linear_projection = nn.Linear(
            self.num_embeddings, self.cls_dim)
        self.ffn = FeedForward(self.cls_dim)
        self.ln1 = nn.LayerNorm(self.cls_dim)
        self.ln2 = nn.LayerNorm(self.cls_dim)
        self.cls_head = nn.Linear(
            self.cls_dim, num_classes)
        
        self.project = self.vqvae.project
        self.total_downsampling = self.downsampling * self.vqvae.stride
        
    @property
    def embedding_filters(self) -> Tensor:
        return self.vqvae.embedding_filters
    
    def classify(self, X: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.project(X)
        # TODO: mask before mean or ?
        Z = self.linear_projection(latent)
        Z = reduce(self.ffn(self.ln1(Z)),
                     'b l h -> b h', 'mean')
        logits = self.cls_head(self.ln2(Z))
        return logits, latent

    def train_embedding(self, X: Tensor) -> tuple[Tensor, VqVaeLoss]:
        return self.vqvae(X)

    def train_embedding_cls(self, X: Tensor) -> tuple[Tensor, VqVaeLoss]:
        _, loss = self.train_embedding(X)
        logits, Z = self.classify(X)
        return logits, loss
    
    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.forward(X)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs

    def forward(self, X: Tensor) -> Tensor:
        logits, _ = self.classify(X)
        return logits
