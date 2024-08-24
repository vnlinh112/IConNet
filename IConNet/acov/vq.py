import torch
from torch import nn, Tensor 
from torch.nn import functional as F
from einops import rearrange

class VectorQuantizer(nn.Module):
    def __init__(
            self, 
            num_embeddings, 
            embedding_dim, 
            commitment_cost, 
            eps=1e-10):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        self.eps = eps

    def forward(self, inputs):
        inputs =  rearrange(inputs, 'b c n -> b n c')
        b,n,c = inputs.shape
        h = self._embedding_dim
        v = self._num_embeddings
        # flat_input = inputs.view(-1, self._embedding_dim)
        flat_input = rearrange(inputs, 'b n c -> k h', b=b, n=n, c=c, h=h)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, 
            device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        # quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # h=embedding_dim, v=num_embedding
        quantized = torch.einsum('hv,vh->vh',
            encodings, self._embedding.weight)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))
        
        quantized = rearrange(quantized, 'b n c -> b c n')
        return loss, quantized, perplexity
    

from .vqt import SimpleVectorQuantize
from einops import pack

class VQuantizer(nn.Module):
    """ https://arxiv.org/abs/2202.01855 """

    def __init__(
        self,
        codebook_dim,
        codebook_size,
        norm = True
    ):
        super().__init__()
        self.norm = nn.LayerNorm(
            codebook_dim, 
            elementwise_affine=False) if norm else nn.Identity()
        
        self.vq = SimpleVectorQuantize(codebook_dim, codebook_size)

    def forward(self, x):
        x = self.norm(x)
        quantized, indices, loss = self.vq(x, indices = indices)
        return quantized, indices, loss

def test():    
    quantizer = VQuantizer(
        codebook_dim = 256,      # codebook dimension
        codebook_size = 4096     # codebook size
    )
    x = torch.randn(16, 62, 1024)
    quantized, indices, loss = quantizer(x) 
    assert quantized.shape == (16, 62) # (16, 62) - (batch, seq)
    return quantized, indices, loss


class SimpleVectorQuantize(nn.Module):
    def __init__(
        self,
        codebook_dim,
        codebook_size,
        eps = 1e-5,
        commitment_weight = 1.,
        learnable_codebook = True
    ):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.eps = eps
        self.commitment_weight = commitment_weight
        self.learnable_codebook = learnable_codebook

        self.codebook = nn.Embedding(
            embedding_dim = codebook_dim,
            num_embeddings = codebook_size
        )        

    def forward(self, x):
        x = rearrange(x, 'b n d -> b n d')
        quantize, embed_ind, distances = self.codebook(x)
        loss = F.mse_loss(quantize, x) * self.commitment_weight         
        quantize = x + (quantize - x).detach()
        return quantize, embed_ind, loss