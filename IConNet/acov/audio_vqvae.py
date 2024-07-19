from typing import Literal
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from ..nn.activation import nl_relu
from ..conv.pad import PadForConv
from .local_filter import LocalPatternFilter, LocalPatterns, FilterEnhance
from .loss import AudioOverlapInformationLoss, AudioMutualInfoMask
from einops import reduce, rearrange, repeat

import collections

VqVaeClsLoss = collections.namedtuple(
    "VqVaeClsLoss", ["perplexity", "loss_vq", "loss_recon", "loss_cls"])

VqVaeLoss = collections.namedtuple(
    "VqVaeLoss", ["perplexity", "loss_vq", "loss_recon"])

VqLoss = collections.namedtuple(
    "VqLoss", ["perplexity", "loss_vq"])

class VectorQuantizer(nn.Module):
    """
    code_fft = nl_relu(fft(win*wave)**2) / max
    code_wave = ifftshift(ifft(exp(code_fft)-1)) / max
    
    TODO: L_ortho
    """
    def __init__(
        self, 
        num_embeddings, 
        embedding_dim, 
        commitment_cost,
        distance_type: Literal['euclidean', 'dot']='euclidean'
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost
        self.distance_type = distance_type
        window = torch.hann_window(self.embedding_dim)
        self.register_buffer("window", window)

    def transform_freq(self, embedding: Tensor) -> Tensor:
        eps = torch.finfo(embedding.dtype).eps
        freq = nl_relu(torch.fft.rfft(self.window * embedding, 
                                      n=self.embedding_dim).real**2)
        freq = freq / torch.clamp(freq.amax(dim=-1, keepdim=True), min=eps)
        return freq

    def get_nearest_neighbour(self, X_flatten: Tensor) -> Tensor:
        assert X_flatten.ndim == 2
        if self.distance_type == 'dot':
            X_flatten_freq = self.transform_freq(X_flatten)
            embedding_freq = self.transform_freq(self.embedding.weight)
            dot_similarity = X_flatten_freq @ embedding_freq.T 
            distances = 1 - dot_similarity / X_flatten_freq.size(-1)
        else: # euclidean            
            distances = (torch.sum(X_flatten**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(X_flatten, self.embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
        return encoding_indices
    
    def quantize(self, X: Tensor) -> Tensor:
        quantized, _, _, _ = self.forward(X)
        return quantized
    
    def compute_perplexity(self, encodings: Tensor) -> Tensor:
        eps = torch.finfo(encodings.dtype).eps
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + eps)))
        return perplexity
    
    def compute_vq_loss(self, quantized: Tensor, X: Tensor) -> Tensor:
        e_latent_loss = F.mse_loss(quantized.detach(), X)
        q_latent_loss = F.mse_loss(quantized, X.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        return vq_loss

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, VqLoss]:
        X = rearrange(X, 'b h c k -> b h c k', k=self.embedding_dim)

        X_flatten = X.view(-1, self.embedding_dim)
        encoding_indices = self.get_nearest_neighbour(X_flatten)
        encodings = torch.zeros(
            (X_flatten.shape[0], self.num_embeddings),
            dtype=X.dtype, device=X.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(X.shape)
        loss = VqLoss(perplexity=self.compute_perplexity(encodings), 
                      loss_vq=self.compute_vq_loss(quantized, X))
        # reparametrization trick
        quantized = X + (quantized - X).detach()
        return quantized, encoding_indices, loss
    

class AudioVqVae(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,         
        num_embeddings: int, 
        embedding_dim: int, 
        commitment_cost: float,
        num_tokens_per_second: int=8, # recommended: 4, 8 or 16
        downsampling: int=8, # recommended: 5 or 8
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
        self.num_tokens_per_second = num_tokens_per_second
        self.sample_rate = sample_rate
        self.downsampling = downsampling
        self.stride = sample_rate // (num_tokens_per_second * downsampling)
        self.kernel_size = self.stride * 2
        self.distance_type = distance_type

        self.pad_layer = PadForConv(
            kernel_size=self.downsampling,
            pad_mode='mean')

        self.encoder = LocalPatternFilter(
            in_channels=in_channels,
            out_channels=out_channels,
            sample_mode=sample_mode
        )
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost,
            distance_type=distance_type
        )  
        self.decoder = FilterEnhance(
            in_channels=in_channels,
            out_channels=out_channels
        )
        
        self.utils = LocalPatterns
        self.audio_criterion = AudioOverlapInformationLoss(
            kernel_size=self.kernel_size, 
            stride=self.stride,
            downsampling=self.downsampling,
            loss_type='overlap', reduction=None
        )
        self.mutual_info_mask = AudioMutualInfoMask(
            kernel_size=self.kernel_size, 
            stride=self.stride,
            downsampling=self.downsampling)
        
    @property
    def embedding_filters(self) -> Tensor:
        return self.vq.embedding.weight
    
    def downsample(self, X: Tensor) -> Tensor:
        return reduce(
            self.pad_layer(X), 
            'b h (t ds) -> b t h', 'max', ds=self.downsampling)

    def project(self, X: Tensor) -> Tensor:
        filters = repeat(
            self.embedding_filters,
            'h n -> h c n', c=self.in_channels)
        X_filtered = F.conv1d(X, filters, padding='same')
        embedding_mask = self.mutual_info_mask(X_filtered, X)
        X_filtered = torch.einsum('bhn,bh->bhn', X_filtered, embedding_mask)
        X_filtered = self.downsample(X_filtered)
        return X_filtered
    
    def get_representation(self, X: Tensor) -> Tensor:
        # TODO: select important tokens from X envelope & return the projection
        pass

    def forward(self, X: Tensor) -> tuple[Tensor, VqVaeLoss]:
        Z = self.encoder(X)
        Z_quantized, _, vq_loss = self.vq(Z)
        X_filtered = self.decoder(X, Z_quantized)
        recon_loss = self.audio_criterion(X_filtered, X)
        return X_filtered, VqVaeLoss(*vq_loss, recon_loss)

    def generate_embedding_mask(self) -> Tensor:
        """
        Evaluate and decide whether the embedding filters are suitable
        for speech classification task based on heuristics.
        """
        zcs = self.utils.get_embedding_color_v2(self.embedding_filters)
        return rearrange(zcs > 0, 'c -> 1 c 1')