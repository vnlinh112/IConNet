from .zero_crossing import (
    zero_crossings, zero_crossing_rate, 
    zero_crossing_score, samples_like,
    signal_distortion_ratio, signal_loss)
from ..nn.activation import gumbel_softmax

import torch 
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from typing import Literal, Optional, Union
from ..conv.pad import PadForConv
from ..nn.mamba import FeedForward

from .visualize import (
    visualize_speech_codebook, get_embedding_color, 
    visualize_embedding_umap, visualize_training_curves
) 

import gc
import numpy as np


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        eps = torch.finfo(inputs.dtype).eps
        device = inputs.device
        inputs = rearrange(inputs, 'b h c k -> b h c k')
        assert self.embedding_dim ==  inputs.shape[-1]

        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
        encodings = torch.zeros(
            encoding_indices.shape[0], 
            self.num_embeddings, 
            device=device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(
            encodings, self.embedding.weight).view(inputs.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + eps)))
        return loss, quantized, perplexity, encodings
    

class LocalPatternFilter(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int=1023,
        filter_selection_mode: Literal['fixed', 'auto']='auto'
    ) -> None:
        """
        filter_selection_mode:
            - `fixed`: linspace
            - `zerocrossing`: zerocrossing with gumbel softmax selection 
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.filter_selection_mode = filter_selection_mode
        self.position_concentration = nn.Parameter(torch.rand(self.out_channels))  
        self.pad_layer = PadForConv(
                    kernel_size=self.kernel_size,
                    pad_mode='zero')

    @staticmethod
    def compute_autocovariance(
            X: Tensor,
            positions,
            kernel_size) -> Tensor:
        acov = []
        for position in positions:
            end_pos = position + kernel_size
            frame = X[..., position:end_pos]
            acov += [LocalPatternFilter.autocovariance(frame)]
        return torch.stack(acov, dim=1)

    @staticmethod
    def autocovariance(filters: Tensor) -> Tensor:
        eps = torch.finfo(filters.dtype).eps
        n = filters.shape[-1]
        filters_max = reduce(filters.abs(), '... n -> ... ()', 'max')
        filters_max = torch.where(filters_max == 0, eps, filters_max)
        win = torch.hann_window(
            window_length=n, 
            dtype=filters.dtype, device=filters.device)
        windowed_data = filters * win / filters_max # normalize
        spectrum = torch.fft.rfft(windowed_data, n=n)**2
        acov = torch.fft.ifftshift(torch.fft.irfft(spectrum.abs(), n=n))
        return acov

    @staticmethod
    def random_filter_positions(
        X: Tensor,
        n_positions: int,
        kernel_size: int,
        end: int,
        mode: Literal['fixed', 'auto']='fixed',
    ):
        if mode=='fixed':
            positions = repeat(
                torch.linspace(
                    start=0, 
                    end=end,
                    steps=n_positions,
                    dtype=torch.int,
                    device=X.device),
                'h -> b h', b=X.shape[0]
            )
        else: # auto: based on zerocrossing
            scores = zero_crossing_score(
                X, n_fft=kernel_size, stride=kernel_size//4)
            scores = gumbel_softmax(scores, tau=1)
            samples = samples_like(
                scores, 
                n_fft=kernel_size, 
                stride=kernel_size//4,
                offset=0, 
                max_sample=end,
                keepdim=False) # (T)
            
            frame_positions = torch.topk(
                scores, n_positions, dim=-1).indices.sort().values
            
            positions = []
            for frame_pos in frame_positions.split(1):
                frame_pos = frame_pos.squeeze()
                positions += [samples.index_select(dim=0, index=frame_pos)]
            positions = torch.stack(positions, dim=0) 
        return positions.contiguous()

    def _generate_filters_idx(self, X: Tensor) -> Tensor:
        length = X.shape[-1]
        end = length - self.kernel_size - 1 - self.kernel_size # TODO: check why -kernel
        n_positions = self.out_channels
        positions = self.random_filter_positions(
            X, n_positions, self.kernel_size,
            end, self.filter_selection_mode)
        positions = rearrange(positions, '... t -> ... t 1')
        filters_idx = torch.concat(
            [positions + i for i in range(self.kernel_size)], 
            dim=-1)
        # print(f'X: {X.shape} | filters_idx: [{filters_idx.min()}, {filters_idx.max()}]')
        return filters_idx

    # def _extract_filters(
    #     self, X: Tensor, filters_idx: Tensor) -> Tensor:
    #     filters = []
    #     z = torch.zeros(
    #         filters_idx.shape[0], filters_idx.shape[1],
    #         X.shape[-1] - self.kernel_size, device=X.device)
    #     filters_idx = torch.concat([filters_idx, z], dim=-1).type(torch.int64)
    #     for idx in filters_idx.split(1, dim=1):
    #         indices = repeat(idx, 'b 1 n -> b c n', c=self.in_channels)
    #         f = torch.gather(
    #             X, dim=-1, index=indices, 
    #             sparse_grad=False)[...,:self.kernel_size]
    #         filters.append(f)
    #     filters = torch.stack(filters, dim=1).contiguous()
    #     return filters

    def _extract_filters(
        self, X: Tensor, filters_idx: Tensor) -> Tensor:
        filters = []
        b, h, n = filters_idx.shape
        for i in range(b):
            filter = []
            for j in range(h):
                filter += [X[i].index_select(dim=-1, index=filters_idx[i, j, :])]
            filter = torch.stack(filter, dim=0)
            filters.append(filter)
        filters = torch.stack(filters, dim=0).contiguous()
        return filters
        
    def _generate_filters(self, X) -> Tensor:
        filters_idx = self._generate_filters_idx(self.pad_layer(X))
        filters = self._extract_filters(X, filters_idx)
        filters = self.autocovariance(filters)
        return filters
    
    def forward(self, X: Tensor) -> Tensor:
        batch_size = X.shape[0]
        filters = self._generate_filters(X)
        assert filters.shape == (batch_size,
                                 self.out_channels, 
                                 self.in_channels, 
                                 self.kernel_size)
        return filters

class FilterEnhance(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, X, filters):
        recon_loss = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        n_outputs = self.out_channels
        n_batch = X.shape[0]
        for i in range(n_batch):
            X_outs = F.conv1d(
                X, filters[i], padding='same').chunk(n_outputs, dim=1)
            loss = torch.tensor(0., dtype=torch.float64, requires_grad=True)
            for k in range(n_outputs):
                loss = loss + signal_loss(X, X_outs[k])
            recon_loss = recon_loss + loss / n_outputs
        return recon_loss.sum()
    

class Model(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,         
        num_embeddings: int, 
        embedding_dim: int, 
        commitment_cost: float,
        num_tokens: int, 
        num_classes: int, 
        filter_selection_mode: Literal['fixed', 'auto']='auto'
    ):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.cls_features = num_embeddings
        self.num_classes = num_classes

        self.encoder = LocalPatternFilter(
            in_channels = in_channels,
            out_channels = out_channels,
            filter_selection_mode=filter_selection_mode
        )
        self.vq_vae = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost)
        self.decoder = FilterEnhance(
            in_channels = in_channels,
            out_channels = out_channels
        )
        self.ffn = FeedForward(self.num_embeddings)
        self.ln1 = nn.LayerNorm(self.num_embeddings)
        self.cls_head = nn.Linear(
            self.cls_features, num_classes)
        self.pad_layer = PadForConv(
                    kernel_size=self.num_tokens,
                    pad_mode='mean')

    def project(self, X):
        filters = repeat(
            self.vq_vae.embedding.weight,
            'h n -> h c n', c=self.in_channels)
        X = F.conv1d(X, filters)
        return X

    def classify(self, X):
        X = reduce(rearrange(self.pad_layer(X), 
                      'b h (l t) -> b l t h', l=self.num_tokens),
                  'b l t h -> b l h', 'max')
        X = reduce(self.ffn(self.ln1(X)),
                     'b l h -> b h', 'mean')
        X = self.cls_head(X)
        return X

    def forward(self, X):
        Z = self.encoder(X)
        vq_loss, Z_quantized, perplexity, _ = self.vq_vae(Z)
        recon_loss = self.decoder(X, Z_quantized)
        logits = self.classify(self.project(X))
        return logits, vq_loss, recon_loss, perplexity

  


@torch.no_grad()
def test(model, iter):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        logits, _,_,_ = model(data)
        del data
        gc.collect()
        torch.cuda.empty_cache()
        probs = F.softmax(logits.squeeze(), dim=-1)
        pred = probs.argmax(dim=-1)
        correct += pred.eq(target).sum().item()
        del target
        gc.collect()
        torch.cuda.empty_cache()
    acc = correct / test_loader_length
    print(f"Test iter: {iter}\tAccuracy: {correct}/{test_loader_length} ({100. * acc:.0f}%)\n")
    return acc

def main():
    num_training_updates = 1000
    log_interval = 100
    test_interval = 100

    in_channels = 1
    out_channels = 8
    embedding_dim = 1023
    num_embeddings = 384

    commitment_cost = 0.25

    learning_rate = 1e-3
    num_tokens = 8
    num_classes = 4

    model = Model(in_channels, out_channels,
                num_embeddings, embedding_dim,
                commitment_cost, num_tokens, 
                num_classes, filter_selection_mode='auto').to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)


    test_loader_length = len(test_loader.dataset)
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    train_res_cls_loss = []
    test_accuracy = []

    for i in range(num_training_updates):
        (data, target) = next(iter(train_loader))
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        logits, vq_loss, recon_loss, perplexity = model(data)
        cls_loss = F.cross_entropy(logits.squeeze(), target)
        loss = vq_loss + recon_loss + 0.15*cls_loss
        loss.backward()
        optimizer.step()

        train_res_recon_error.append(recon_loss.item())
        train_res_perplexity.append(perplexity.item())
        train_res_cls_loss.append(cls_loss.item())

        del data, target
        gc.collect()
        torch.cuda.empty_cache() 

        if (i+1) % log_interval == 0:
            print('%d iterations' % (i+1))
            print('cls_loss: %.3f' % np.mean(train_res_cls_loss[-log_interval:]))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-log_interval:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-log_interval:]))
            print()

        if (i+1) % test_interval == 0:
            model.eval()
            acc = test(model, i+1)
            test_accuracy.append(acc)
            model.train()

    visualize_training_curves(
        test_accuracy, train_res_cls_loss,
        train_res_recon_error, train_res_perplexity
    )
    
    emb_data = model.vq_vae.embedding.weight.data.cpu()
    emb_color, zcs = get_embedding_color(emb_data)
    k = 2
    min_dist=0.9
    visualize_embedding_umap(emb_data, colors=emb_color, edgecolors=None,
            n_neighbors=k, min_dist=min_dist, metric='euclidean', 
            title=f'n_neighbors={k}, min_dist={min_dist}')
    
    visualize_speech_codebook(emb_data.numpy(), n=64, colors=emb_color)

    selected_emb_topk = torch.topk(zcs, 32)
    selected_emb = emb_data.index_select(dim=0, index=selected_emb_topk.indices)
    selected_emb_color = torch.stack(
            [torch.full_like(selected_emb_topk.values, 0.5), 
            selected_emb_topk.values, selected_emb_topk.values], 
            dim=-1).numpy()
    visualize_speech_codebook(selected_emb.numpy(), n=64, colors=selected_emb_color)


