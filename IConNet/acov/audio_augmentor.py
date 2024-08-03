from typing import Literal, Optional
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from ..nn.activation import gumbel_softmax, nl_relu, NLReLU
from ..conv.pad import PadForConv
from .zero_crossing import zero_crossings, zero_crossing_score, samples_like
from einops import reduce, rearrange, repeat
from .loss import AudioMutualInfoMask, AudioMutualInfo, AudioOverlapInformationLoss
from .local_filter import LocalPatterns, SpeechSegmentSelector
from .audio_vqvae import AudioVqVae, VectorQuantizer, VqLoss, VqVaeLoss
from ..nn.classifier import FeedForward, FeedForwardAddNorm
import collections

ContrastiveLoss = collections.namedtuple(
    "ContrastiveLoss", ["intra", "inter"])

class AudioAugmentor(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            kernel_size: int,
            stride: int,
            num_embeddings: int, 
            embedding_dim: int, 
            commitment_cost: float=0.1,
            max_num_tokens: int=2048,
            distance_type: Literal['euclidean', 'dot']='euclidean',
            loss_type: Literal['overlap', 'minami', 
                            'maxami', 'mami']='overlap',
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size 
        self.stride = stride
        self.max_num_tokens = max_num_tokens
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.distance_type = distance_type
        self.out_channels = num_embeddings
        self.loss_type = loss_type

        self.tokenizer = SpeechSegmentSelector(
            in_channels=in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            max_num_tokens=max_num_tokens
        )

        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost,
            distance_type=distance_type
        )  

        self.audio_criterion = AudioOverlapInformationLoss(
            kernel_size=512, 
            stride=250,
            downsampling=8,
            loss_type=loss_type, 
            reduction='mean'
        )

        self.encoder = FeedForwardAddNorm(self.num_embeddings)
        zero_loss = torch.tensor(0., dtype=torch.float64)
        self.register_buffer('zero_loss', zero_loss)
    
    @property
    def embedding_filters(self) -> Tensor:
        return self.vq.embedding.weight

    def project(self, X: Tensor) -> Tensor:
        filters = repeat(
            self.embedding_filters,
            'h n -> h c n', c=self.in_channels)
        X_filtered = F.conv1d(X, filters, padding='same')
        return X_filtered
    
    def compute_recon_loss(
            self, x: Tensor, filters: Tensor) -> Tensor:
        X = x[None, ...]
        X_filtered = F.conv1d(X, filters, padding='same')  # b h n
        X_filtered = self.tokenizer.norm_fn(X_filtered)
        recon_loss = self.audio_criterion(X_filtered, X)
        return recon_loss

    def tokenize(self, X: Tensor) -> Tensor:
        X = self.tokenizer.norm_fn(X) # b h c n
        filters = self.tokenizer._extract_all_filters(X)
        return filters
    
    def train_embedding_ssl(self, X: Tensor) -> VqVaeLoss:
        tokens = self.tokenize(X)
        tokens, _, vq_loss = self.vq(tokens)
        recon_loss = torch.tensor(0., dtype=torch.float64, device=X.device, requires_grad=True)
        for i, x in enumerate(X):
            recon_loss = recon_loss + self.compute_recon_loss(x, tokens[i])
        recon_loss = recon_loss / X.shape[0]
        return VqVaeLoss(*vq_loss, recon_loss)

    def train_projector_ssl(self, X: Tensor) -> VqVaeLoss:
        """Not recommended: codebook collapse"""
        X_filtered = self.project(X)
        X_filtered = self.tokenizer.norm_fn(X_filtered)
        recon_loss = self.audio_criterion(X_filtered, X)
        return VqVaeLoss(self.zero_loss, self.zero_loss, loss_recon=recon_loss)
    
    def compute_contrastive_loss(self, X: Tensor, X_negative: Optional[Tensor]=None):
        target = torch.ones(X.shape[0], device=X.device)
        B = X.shape[0]
        loss = torch.tensor(0., dtype=torch.float64, device=X.device, requires_grad=True)
        if X_negative is None: # intra-class loss
            for b in range(B):
                loss = loss + F.cosine_embedding_loss(
                    X, X.roll(shifts=b, dims=0), target=target)
        else:            
            for b in range(B):
                loss = loss + F.cosine_embedding_loss(
                    X, X_negative.roll(shifts=b, dims=0), target=-target)
        return loss / B
    
    def encode(self, X: Tensor) -> Tensor:
        X = nl_relu(self.project(X))
        X = reduce(X, 'b n h -> b h', 'mean')
        X = self.encoder(X)
        return X
                
    def train_encoder_contrastive(
            self, X1: Tensor, X2: Tensor, 
            lambda_intra: float=0.1, lambda_inter: float=1.) -> VqVaeLoss:
        Z1 = self.encode(X1)
        Z2 = self.encode(X2)
        if lambda_intra > 0:
            loss_intra = self.compute_contrastive_loss(Z1) + self.compute_contrastive_loss(Z2)
        else:
            loss_intra = self.zero_loss
        loss_inter = self.compute_contrastive_loss(Z1, Z2)
        loss = lambda_inter * loss_inter + lambda_intra * loss_intra
        return VqVaeLoss(self.zero_loss, self.zero_loss, loss_recon=loss)
        
    def forward(self, X: Tensor) -> Tensor: 
        X_augmented = self.project(X)
        return X_augmented