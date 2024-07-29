from typing import Literal, Optional
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from ..nn.activation import gumbel_softmax
from ..conv.pad import PadForConv
from .zero_crossing import zero_crossings, zero_crossing_score, samples_like
from einops import reduce, rearrange, repeat
from .loss import AudioMutualInfoMask, AudioMutualInfo

class LocalPatterns(nn.Module):
    def __init__(self):
        super().__init__() 

    @staticmethod
    def compute_autocovariance(
            X: Tensor,
            positions,
            kernel_size) -> Tensor:
        acov = []
        for position in positions:
            end_pos = position + kernel_size
            frame = X[..., position:end_pos]
            acov += [LocalPatterns.autocovariance(frame)]
        return torch.stack(acov, dim=1)

    @staticmethod
    def autocovariance(filters: Tensor) -> Tensor:
        eps = torch.finfo(filters.dtype).eps
        n = filters.shape[-1]
        filters = filters / torch.clamp(
            filters.abs().amax(dim=-1, keepdim=True), min=eps)
        win = torch.hann_window(
            window_length=n, 
            dtype=filters.dtype, 
            device=filters.device).reshape(1,1,-1)
        windowed_data = filters * win
        spectrum = torch.fft.rfft(windowed_data, n=n)**2
        acov = torch.fft.ifftshift(
            torch.fft.irfft(spectrum.real, n=n))
        return acov

    @staticmethod
    def sample_using_zero_crossing(
        X: Tensor,
        n_positions: int,
        kernel_size: int,
        end: int,
    ):
        stride = kernel_size // 4
        scores = zero_crossing_score(
                X, n_fft=kernel_size, stride=stride)
        scores = gumbel_softmax(scores, tau=1)
        samples = samples_like(
            scores, 
            n_fft=kernel_size, 
            stride=stride,
            offset=0, 
            max_sample=end,
            keepdim=False) # (T)
        
        frame_positions = torch.topk(
            scores, n_positions, dim=-1
        ).indices.sort().values
        
        positions = []
        for frame_pos in frame_positions.split(1):
            frame_pos = frame_pos.squeeze()
            positions += [samples.index_select(dim=0, index=frame_pos)]
        positions = torch.stack(positions, dim=0) 
        return positions

    @staticmethod
    def sample_using_envelope(
        X: Tensor,
        n_positions: int,
        kernel_size: int,
        end: Optional[int]=None,
    ):
        eps = torch.finfo(X.dtype).eps
        stride = kernel_size // 4
        X_envelope = F.avg_pool1d(
                X**2, kernel_size=kernel_size, stride=stride)
        X_envelope = torch.clamp(F.softsign(X_envelope), min=eps)
        X_envelope = X_envelope.view(-1, X_envelope.size(-1))
        
        positions = []
        for x in X_envelope:
            d = torch.distributions.Dirichlet(x)
            positions += [d.sample().topk(n_positions).indices*stride]
        positions = torch.stack(positions, dim=0).type(torch.long)
        positions = positions.view(*X.shape[:-1], n_positions)
        return positions

    @staticmethod
    def sample_filter_positions(
        X: Tensor,
        n_positions: int,
        kernel_size: int,
        sample_mode: Literal['fixed', 
                    'zero_crossing', 
        'envelope']='envelope'
    ):
        if sample_mode=='envelope':
            positions = LocalPatterns.sample_using_envelope(
                X, n_positions, kernel_size)
        elif sample_mode=='zero_crossing':
            positions = LocalPatterns.sample_using_zero_crossing(
                X, n_positions, kernel_size)
        else: # fixed
            length = X.shape[-1]
            end = length - 2*kernel_size - 1
            positions = repeat(
                torch.linspace(
                    start=0, 
                    end=end,
                    steps=n_positions,
                    dtype=torch.int,
                    device=X.device),
                'h -> b h', b=X.shape[0]
            )
        return positions.contiguous()
    
    @staticmethod
    def get_embedding_color_v2(
            y: Tensor, 
            mask_ratio=0.2, 
            score_offset=0.5) -> Tensor:
        y = y / repeat(reduce(y, 'b n -> b' , 'max'), 'b -> b n', n=1)
        crossings = zero_crossings(y, dtype=torch.float)
        zcrate = crossings.mean(dim=-1)
        length = y.shape[-1]
        mid_q = length // 4   
        crossings_mid = crossings[:, mid_q:-mid_q]
        nonzero_mean = torch.where(
            crossings>0, crossings, 0.0).mean(dim=-1)
        nonzero_mean_mid = torch.where(
            crossings_mid>0, crossings_mid, 0.0).mean(dim=-1)
        kernel_size = (length + 1) // 8
        y_pos_avg = F.avg_pool1d(
            torch.clamp(y, min=0), 
            kernel_size=kernel_size, stride=kernel_size)
        emb_data_cond = torch.diff(y_pos_avg) > 0
        cond1 = emb_data_cond[:, 1:3].sum(dim=-1) == 2
        cond2 = emb_data_cond[:, -2:].sum(dim=-1) == 0
        cond3 = y_pos_avg[:, 3] - y_pos_avg[:, -1] > 0
        cond4 = y_pos_avg[:, 4] - y_pos_avg[:, 0] > 0
        emb_data_mask = cond1 * cond2 * cond3 * cond4
        score0 = F.sigmoid(1 - 8*zcrate + nonzero_mean + nonzero_mean_mid)
        score1 = mask_ratio * emb_data_mask
        score = score0 + score1*(1-score0) - score_offset
        score_sig = torch.clamp(
            F.softplus(score, beta=40.0, threshold=0.5)*10, max=1)
        return score

class LocalPatternFilter(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int=1023,
        sample_mode: Literal['fixed', 
                            'zero_crossing', 
                            'envelope']='envelope'
    ) -> None:
        """
        sample_mode:
            - `fixed`: linspace
            - `zero_crossing`: zero crossing with gumbel softmax sampling 
            - `envelope`: envelope with dirichlet sampling
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_mode = sample_mode
        self.pad_layer = PadForConv(
                    kernel_size=self.kernel_size,
                    pad_mode='zero')
        self.utils = LocalPatterns
        self.norm_fn = lambda A: A / torch.clamp(A.abs().amax(dim=-1, keepdim=True), min=torch.finfo(A.dtype).eps)

    def _generate_filters_idx(self, X: Tensor) -> Tensor:
        n_positions = self.out_channels
        positions = self.utils.sample_filter_positions(
            X, n_positions, self.kernel_size,
            self.sample_mode)
        positions = rearrange(positions, '... t -> ... t 1')
        filters_idx = torch.concat(
            [positions + i for i in range(self.kernel_size)], 
            dim=-1)
        return filters_idx
        
    def _extract_filters(
        self, X: Tensor, filters_idx: Tensor) -> Tensor:
        filters = []
        b, c, h, n = filters_idx.shape
        for i in range(b):
            filter = [X[i].index_select(
                        dim=-1, 
                        index=filters_idx[i, k, j, :]) for k in range(c) for j in range(h)]
            filter = torch.stack(filter, dim=0)
            filters.append(filter)
        filters = torch.stack(filters, dim=0).contiguous()
        return filters
        
    def _generate_filters(self, X) -> Tensor:
        filters_idx = self._generate_filters_idx(self.pad_layer(X))
        filters = self._extract_filters(X, filters_idx)
        filters = self.utils.autocovariance(filters)
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
        out_channels
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, X, filters):
        eps = torch.finfo(X.dtype).eps
        X /= torch.clamp(X.max(dim=-1, keepdim=True).values, min=eps)
        filters = rearrange(filters, 'b h c n -> (b h) c n', c=1)
        X_outs = F.conv1d(X, filters, padding='same')
        return X_outs
    

class SpeechSegmentSelector(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            kernel_size: int=1023,
            stride: int=128,
            max_num_tokens: int=2048
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size 
        self.stride = stride
        self.max_num_tokens = max_num_tokens

        self.ami = AudioMutualInfo(
            kernel_size=512, stride=250, downsampling=8)
        self.mutual_information = self.ami.mutual_information
        self.compute_probs = self.ami.compute_probs
        self.utils = LocalPatterns
        self.norm_fn = lambda A: A / torch.clamp(A.abs().amax(dim=-1, keepdim=True), min=torch.finfo(A.dtype).eps)
        self.pad_fn = PadForConv(kernel_size=self.stride, pad_mode='zero')

    
    def _extract_all_filters(self, X: Tensor) -> Tensor:
        if self.stride == self.kernel_size:
            filters = rearrange(
                self.pad_fn(X), 'b c (h kz) -> b h c kz', 
                kz=self.kernel_size, c=1)
        else:
            X_pad = self.pad_fn(X)
            n_filters = X_pad.shape[-1] // self.stride
            X_pad = F.pad(X_pad, pad=(0, self.kernel_size*2), mode='constant', value=0)
            filters_positions = torch.arange(n_filters) * self.stride
            filters = [X_pad[..., p:p+self.kernel_size] for p in filters_positions]
            filters = rearrange(torch.stack(filters, dim=2), 'b c h kz -> b h c kz')        
        filters = self.utils.autocovariance(filters)
        filters = self.norm_fn(filters)
        return filters

    def compute_segment_score(
            self, x: Tensor, filters: Tensor, hard: bool=True) -> Tensor:
        X = x[None, ...]
        X_filtered = F.conv1d(X, filters, padding='same')  # b h n
        X_filtered = self.norm_fn(X_filtered)
        if hard==True:
            px = self.compute_probs(X)
            px_cz = self.compute_probs(X_filtered)
            out = reduce(px.ge(px_cz), 'b h n -> b h', 'all')
        else:
            out, (_, _) = self.mutual_information(X_filtered, X)
        return out

    def _generate_filters(self, X: Tensor) -> Tensor:
        X = self.norm_fn(X)
        filters = self._extract_all_filters(X)
        segment_mask = torch.concat(
            [self.compute_segment_score(x, filters[i]) for i, x in enumerate(X)], 
            dim=0)
        filters = torch.einsum('bhcn,bh->bhn', filters, segment_mask)
        # filters = rearrange(filters, 'b h c n -> b (c h) n', c=1)
        return filters

    def forward(self, X: Tensor) -> Tensor: 
        filters = self._generate_filters(X)
        return filters