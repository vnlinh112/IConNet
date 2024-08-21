from typing import Literal, Optional
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from ..nn.activation import nl_relu
from ..conv.pad import PadForConv
from .local_filter import LocalPatternFilter, LocalPatterns, FilterEnhance
from .loss import AudioOverlapInformationLoss, AudioMutualInfoMask, SignalLoss
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
        num_embeddings: int, 
        embedding_dim: int, 
        commitment_cost: float=0.1,
        distance_type: Literal['euclidean', 'dot']='euclidean',
        learnable_codebook: bool=True
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
        self.norm_fn = lambda A: A / torch.clamp(
            A.abs().amax(dim=-1, keepdim=True), min=torch.finfo(A.dtype).eps)
        
        self.learnable_codebook = learnable_codebook

    def from_pretrained(self, path, freeze=True):
        if path is not None:
            self.learnable_codebook = not freeze
            self.embedding = self.embedding.from_pretrained(torch.load(path), freeze=freeze)
    
    def transform_freq(self, embedding: Tensor) -> Tensor:
        freq = nl_relu(torch.fft.rfft(self.window * embedding, 
                                      n=self.embedding_dim+1).real**2)[..., 1:]
        freq = self.norm_fn(freq)
        return freq

    def get_nearest_neighbour(self, X_flatten: Tensor) -> Tensor:
        assert X_flatten.ndim == 2
        if self.distance_type == 'dot':
            X_flatten_freq = self.transform_freq(X_flatten)
            embedding_freq = self.transform_freq(self.embedding.weight)
            dot_similarity = X_flatten_freq @ embedding_freq.T 
            distances = 1 - dot_similarity / X_flatten_freq.shape[-1]
            distances = 1 - dot_similarity / X_flatten_freq.shape[-1]
        else: # euclidean            
            distances = (torch.sum(X_flatten**2, dim=-1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(X_flatten, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=-1, keepdim=True)
        encoding_indices = torch.argmin(distances, dim=-1, keepdim=True)
        return encoding_indices
    
    def quantize(self, X: Tensor) -> Tensor:
        quantized, _, _, _ = self.forward(X)
        return quantized
    
    def compute_perplexity(self, encodings: Tensor) -> Tensor:
        eps = torch.finfo(encodings.dtype).eps
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs+eps)))
        return perplexity
    
    def compute_vq_loss(self, quantized: Tensor, X: Tensor) -> Tensor:
        if self.learnable_codebook:
            e_latent_loss = F.mse_loss(quantized.detach(), X)
            q_latent_loss = F.mse_loss(quantized, X.detach())
            vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        else:
            vq_loss = F.mse_loss(quantized.detach(), X.detach())
        return vq_loss

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, VqLoss]:
        assert X.shape[-1] == self.embedding_dim, f'Expect input last dim={self.embedding_dim}, got {X.shape[-1]} instead'
        assert X.ndim == 3 or X.ndim == 4
        assert X.shape[-1] == self.embedding_dim, f'Expect input last dim={self.embedding_dim}, got {X.shape[-1]} instead'
        assert X.ndim == 3 or X.ndim == 4

        X_flatten = X.view(-1, self.embedding_dim)
        encoding_indices = self.get_nearest_neighbour(X_flatten)
        encodings = torch.zeros(
            (X_flatten.shape[0], self.num_embeddings),
            dtype=X.dtype, device=X.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(X.shape)
        vq_loss = self.compute_vq_loss(quantized, X)
        loss = VqLoss(perplexity=self.compute_perplexity(encodings), 
                      loss_vq=vq_loss)
        if self.learnable_codebook:# reparametrization trick
            quantized = X + (quantized - X).detach()
        else:
            quantized = quantized.detach()
            encoding_indices = encoding_indices.detach()
        encoding_indices = rearrange(
            encoding_indices, '(b n) 1 -> b n', b=X.shape[0])
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
        distance_type: Literal['euclidean', 'dot']='euclidean',
        projector_mask: bool=False, 
        loss_type: Literal['overlap', 'minami', 
                            'maxami', 'mami',
                            'signal_loss']='signal_loss',     
        codebook_pretrained_path: Optional[str]=None,     
        freeze_codebook: bool=False         
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
            sample_mode=sample_mode,
            kernel_size=embedding_dim
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

        self.load_codebook(
            path=codebook_pretrained_path,
            freeze=freeze_codebook)

        if loss_type == 'signal_loss':
            self.audio_criterion = SignalLoss()
        else:
            self.audio_criterion = AudioOverlapInformationLoss(
                kernel_size=self.kernel_size, 
                stride=self.stride,
                downsampling=self.downsampling,
                loss_type=loss_type, reduction='mean'
            )

        self.projector_mask = projector_mask 

        # if self.projector_mask:
        #     self.mutual_info_mask = AudioMutualInfoMask(
        #         kernel_size=self.kernel_size, 
        #         stride=self.stride,
        #         downsampling=self.downsampling)
        
    @property
    def embedding_filters(self) -> Tensor:
        return self.vq.embedding.weight

    def load_codebook(self, path, freeze=True):
        self.codebook_pretrained_path = path
        self.freeze_codebook = freeze
        if path is not None:
            self.vq.from_pretrained(
                path, freeze=freeze)
    
    def downsample(self, X: Tensor) -> Tensor:
        return reduce(
            self.pad_layer(X), 
            'b h (t ds) -> b t h', 'max', ds=self.downsampling)

    def project(self, X: Tensor) -> Tensor:
        filters = repeat(
            self.embedding_filters,
            'h n -> h c n', c=self.in_channels)
        X_filtered = F.conv1d(X, filters, padding='same')
        if self.projector_mask:
            embedding_mask = self.generate_embedding_mask()
            X_filtered = torch.einsum('bhn,h->bhn', X_filtered, embedding_mask)
        # X_filtered = self.downsample(X_filtered) # TODO: bug
        return X_filtered
    
    def get_representation(self, X: Tensor) -> Tensor:
        # TODO: select important tokens from X envelope & return the projection
        pass

    def train_embedding_ssl(self, X: Tensor) -> VqVaeLoss:
        Z = self.encoder(X)
        Z_quantized, _, vq_loss = self.vq(Z)
        X_filtered = self.decoder(X, Z_quantized)
        recon_loss = self.audio_criterion(X_filtered, X)
        return VqVaeLoss(*vq_loss, recon_loss)

    def forward(self, X: Tensor) -> Tensor:
        Z = self.encoder(X)
        Z_quantized, _, _ = self.vq(Z)
        X_filtered = self.decoder(X, Z_quantized)
        return X_filtered
    
    def generate_embedding_mask(self) -> Tensor:
        """
        Evaluate and decide whether the embedding filters are suitable
        for speech classification task based on heuristics.
        """
        zcs = self.utils.get_embedding_color_v2(self.embedding_filters)
        mask = zcs > 0
        return mask
    
    def generate_embedding_mask_v2(self) -> Tensor:
        emb_fft_db = nl_relu(torch.fft.rfft(self.embedding_filters).real**2)
        fft_score = torch.where(emb_fft_db<=0.5, 1, 0).sum(dim=-1)
        zcs = self.utils.get_embedding_color_v2(self.embedding_filters)
        mask = torch.logical_and(zcs > 0, fft_score > 100)
        return mask