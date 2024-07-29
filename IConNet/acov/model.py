from typing import Literal, Optional
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from .audio_vqvae import AudioVqVae, VqVaeLoss
from einops import reduce, rearrange, repeat
from ..nn.mamba import FeedForward
from ..nn.mamba_model import MambaSeq2OneBlocks
from .audio_vqmix import AudioVQEncoder, AudioVQMixClsLoss
from ..nn.classifier import Classifier


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


class SCB10(nn.Module):
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
        super().__init__()
        self.in_channels = in_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.cls_dim = cls_dim
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.distance_type = distance_type
        self.kernel_size = kernel_size
        self.stride = stride

        self.audio_encoder = AudioVQEncoder(
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

        # self.cls_head = Classifier(
        #     n_input = embedding_dim,
        #     n_output = num_classes,
        #     n_block = 1, 
        #     n_hidden_dim = (cls_dim,),
        #     dropout = 0.1
        # )

        self.linear_projection = nn.Linear(
            self.embedding_dim, self.cls_dim)
        self.ffn = FeedForward(self.cls_dim)
        self.ln1 = nn.LayerNorm(self.cls_dim)
        # self.ln2 = nn.LayerNorm(self.cls_dim)
        self.cls_head = nn.Linear(
            self.cls_dim, num_classes)

    def train_cls(
            self, X: Tensor, Y: Optional[Tensor]=None
        ) -> tuple[Tensor, AudioVQMixClsLoss]:
        
        tokens, loss = self.audio_encoder.encode(X)
        Z = self.seq_encoder(rearrange(tokens, 'b n c -> b c n'))
        # logits = self.cls_head(Z)
        Z = self.linear_projection(Z)
        Z = self.ffn(self.ln1(Z))
        logits = self.cls_head(Z)

        if Y is not None:
            loss_cls = F.cross_entropy(logits.squeeze(), Y)
            loss = loss._replace(loss_cls=loss_cls)
        return logits, loss
    
    def train_ssl(self, X: Tensor):
        pass
    
    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.forward(X)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs

    def forward(self, X: Tensor) -> Tensor:
        logits, _ = self.train_cls(X)
        return logits