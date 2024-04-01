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
    

from .vqt import VectorQuantize
from einops import pack

class RandomProjectionQuantizer(nn.Module):
    """ https://arxiv.org/abs/2202.01855 """

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        codebook_dim,
        num_codebooks = 1,
        norm = True,
        **kwargs
    ):
        super().__init__()
        self.num_codebooks = num_codebooks

        rand_projs = torch.empty(num_codebooks, dim, codebook_dim)
        nn.init.xavier_normal_(rand_projs)

        self.register_buffer('rand_projs', rand_projs)

        # in section 3 of https://arxiv.org/abs/2202.01855
        # "The input data is normalized to have 0 mean and standard deviation of 1 ... to prevent collapse"

        self.norm = nn.LayerNorm(dim, elementwise_affine = False) if norm else nn.Identity()

        self.vq = VectorQuantize(
            dim = codebook_dim * num_codebooks,
            heads = num_codebooks,
            codebook_size = codebook_size,
            use_cosine_sim = True,
            separate_codebook_per_head = True,
            **kwargs
        )

    def forward(
        self,
        x,
        indices = None
    ):
        return_loss = indices is not None

        x = self.norm(x)
        x = torch.einsum('b n d, h d e -> b n h e', x, self.rand_projs)
        x, ps = pack([x], 'b n *')
        self.vq.eval()
        quantized, indices, loss = self.vq(x, indices = indices)

        if return_loss:
            return loss

        return indices