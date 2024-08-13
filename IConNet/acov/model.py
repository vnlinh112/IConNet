from typing import Literal, Optional, Union, Callable
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from .audio_vqvae import AudioVqVae, VqVaeLoss, VqVaeClsLoss
from einops import reduce, rearrange, repeat
from ..nn.classifier import FeedForward, FeedForwardAddNorm
from .audio_vqmix import AudioVQEncoder, VqClsLoss
from .audio_augmentor import AudioAugmentor
from ..nn.activation import nl_relu
from .scb_win_conv import SCBWinConv
from ..nn.sequential import Seq2SeqBlocks, Seq2OneBlocks, Seq2MBlocks
from ..nn.activation import NLReLU
from ..trainer.model_wrapper import ModelWrapper
from ..conv.pad import PadForConv

class SCB(nn.Module):
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
        out_channel: Optional[int]=None,
        kernel_size: Optional[int]=None,
        downsampling: Optional[int]=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.cls_dim = cls_dim
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.stride = stride
        self.sample_mode = sample_mode
        self.distance_type = distance_type
        self.loss_type = loss_type

        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.downsampling = downsampling

        self.encoder: Union[AudioVQEncoder, AudioAugmentor] = None
         
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.cls_dim),
            nn.Linear(self.num_embeddings, self.cls_dim, bias=False),
            nn.PReLU(self.cls_dim),
            nn.Linear(self.cls_dim, num_classes)
        )

        zero_loss = torch.tensor(0., dtype=torch.float64)
        self.register_buffer('zero_loss', zero_loss)
        
    @property
    def embedding_filters(self) -> Tensor:
        return self.encoder.embedding_filters
    
    def classify(self, X: Tensor) -> tuple[Tensor, Tensor]:
        X = nl_relu(self.encoder.project(X))
        latent = reduce(X,'b h n -> b h', 'mean')
        logits = self.classifier(latent)
        return logits, latent

    def train_embedding(self, X: Tensor) -> VqVaeClsLoss:
        vq_loss = self.encoder.train_embedding_ssl(X)
        return VqVaeClsLoss(*vq_loss, loss_cls=self.zero_loss)
    
    def train_projector(self, X: Tensor) -> VqVaeClsLoss:
        vq_loss = self.encoder.train_projector_ssl(X)
        return VqVaeClsLoss(*vq_loss, loss_cls=self.zero_loss)
    
    def train_embedding_projector(self, X: Tensor) -> VqVaeClsLoss:
        vq_loss = self.train_embedding(X)
        recon_loss = vq_loss.loss_recon + self.train_projector(X).loss_recon
        loss = vq_loss._replace(loss_recon=recon_loss, 
                                loss_cls=self.zero_loss)
        return loss
    
    def train_embedding_cls(
            self, X: Tensor, Y: Optional[Tensor]=None
        ) -> tuple[Tensor, VqVaeClsLoss]:

        vq_loss = self.train_embedding(X)
        logits, Z = self.classify(X)
        if Y is not None:
            loss_cls = F.cross_entropy(logits.squeeze(), Y)
        else:
            loss_cls = self.encoder.zero_loss
        loss = vq_loss._replace(loss_cls=loss_cls)
        return Z, loss
    
    def train_projector_cls(
            self, X: Tensor, Y: Optional[Tensor]=None
        ) -> tuple[Tensor, VqVaeClsLoss]:

        vq_loss = self.train_projector(X)
        logits, Z = self.classify(X)
        if Y is not None:
            loss_cls = F.cross_entropy(logits.squeeze(), Y)
        else:
            loss_cls = self.encoder.zero_loss
        loss = vq_loss._replace(loss_cls=loss_cls)
        return Z, loss
    
    def train_embedding_projector_cls(
            self, X: Tensor, Y: Optional[Tensor]=None
        ) -> tuple[Tensor, VqVaeClsLoss]:

        vq_loss = self.train_embedding(X)
        recon_loss = vq_loss.loss_recon + self.train_projector(X).loss_recon
        logits, Z = self.classify(X)
        if Y is not None:
            loss_cls = F.cross_entropy(logits.squeeze(), Y)
        else:
            loss_cls = self.encoder.zero_loss
        loss = vq_loss._replace(loss_recon=recon_loss, loss_cls=loss_cls)
        return Z, loss
    
    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.forward(X)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs

    def forward(self, X: Tensor) -> Tensor:
        logits, _ = self.classify(X)
        return logits

class SCB7(SCB):
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
        num_tokens: int=256,
        cls_dim: int=500,
        sample_rate: int=16000,
        sample_mode: Literal['fixed', 
                            'zero_crossing', 
                            'envelope']='zero_crossing',
        distance_type: Literal['euclidean', 'dot']='euclidean',
        loss_type: Literal['overlap', 'minami', 
                            'maxami', 'mami',
                            'signal_loss']='signal_loss',
        codebook_pretrained_path: Optional[str]=None,
        freeze_codebook: bool=False
    ):
        super().__init__(
            in_channels=in_channels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            cls_dim=cls_dim,
            num_classes=num_classes,
            sample_rate=sample_rate,
            downsampling=downsampling,
            sample_mode=sample_mode,
            distance_type=distance_type,
            commitment_cost=commitment_cost
        )
        self.num_tokens = num_tokens
        self.out_channel = out_channels
        self.encoder = AudioVqVae(
            in_channels=in_channels,
            out_channels=out_channels,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost,
            sample_rate=sample_rate,
            num_tokens_per_second=num_tokens_per_second,
            downsampling = downsampling,
            sample_mode=sample_mode,
            distance_type=distance_type,
            projector_mask=True,
            loss_type=loss_type,
            codebook_pretrained_path=codebook_pretrained_path,
            freeze_codebook=freeze_codebook
        )  
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.num_embeddings),
            FeedForward(self.num_embeddings)
        )
        self.cls_head = nn.Linear(
            self.num_embeddings, num_classes)
        self.pad_layer = PadForConv(
                    kernel_size=self.num_tokens,
                    pad_mode='mean')

    def classify(self, X):
        X = self.encoder.project(X)
        X = reduce(self.pad_layer(X), 
                'b h (l t) ->  b l h', 'max', l=self.num_tokens)
        latent = reduce(self.ffn(X),
                     'b l h -> b h', 'mean')
        logits = self.cls_head(latent)
        return logits, latent


class SCB8(SCB):
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
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            cls_dim=cls_dim,
            num_classes=num_classes,
            num_tokens_per_second=num_tokens_per_second,
            sample_rate=sample_rate,
            downsampling=downsampling,
            sample_mode=sample_mode,
            distance_type=distance_type,
            commitment_cost=commitment_cost
        )

        self.encoder = AudioVqVae(
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
        self.ffn = nn.Sequential(
            nn.Linear(self.num_embeddings, self.cls_dim),
            nn.LayerNorm(self.cls_dim),
            FeedForward(self.cls_dim)
        )
        self.ln1 = nn.LayerNorm(self.cls_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.cls_dim),
            nn.Linear(self.cls_dim, num_classes)
        )
        
        self.total_downsampling = self.downsampling * self.encoder.stride
    
    def classify(self, X: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encoder.project(X)
        # TODO: mask before mean or ?
        Z = reduce(self.ffn(latent),
                     'b l h -> b h', 'mean')
        logits = self.classifier(Z)
        return logits, latent
    

class SCB11(SCB):
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
            commitment_cost=commitment_cost
        )

        self.encoder = AudioAugmentor(
            in_channels=in_channels,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            kernel_size=embedding_dim,
            stride=stride,
            commitment_cost=commitment_cost,
            distance_type=distance_type,
            loss_type=loss_type
        )  
         
        self.classifier = nn.Sequential(
            nn.Linear(self.num_embeddings, self.cls_dim),
            FeedForward(self.cls_dim),
            nn.Linear(self.cls_dim, num_classes)
        )

        self.zero_loss = self.encoder.zero_loss
        
    
    def classify(self, X: Tensor) -> tuple[Tensor, Tensor]:
        X = self.encoder.project(X)
        X = F.max_pool1d(X, kernel_size=128, stride=32)
        latent = reduce(X,'b h n -> b h', 'mean')
        logits = self.classifier(latent)
        return logits, latent

    # def classify(self, X: Tensor) -> tuple[Tensor, Tensor]:
    #     X_filtered = self.encoder.project(X)
    #     ami = AudioMutualInfo(
    #         kernel_size=512, 
    #         stride=250,
    #         downsampling=8
    #     )
    #     mi_score, _ = ami.mutual_information(X_filtered, X)
    #     latent = mi_score
    #     logits = self.classifier(latent)
    #     return logits, latent


class SCB12(SCB):
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
            commitment_cost=commitment_cost 
        )

        self.encoder = AudioVQEncoder(
            in_channels=in_channels,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            kernel_size=embedding_dim,
            stride=stride,
            commitment_cost=commitment_cost,
            distance_type=distance_type,
            codebook_type='wave',
            positional_encoding_type='learnable',
            num_tokens_per_second=num_tokens_per_second,
            max_num_tokens=max_num_tokens, 
            sample_rate=sample_rate, 
        )

        self.init_codebooks(codebook_pretrained_path, freeze_codebook)

        self.linear_projector = nn.Linear(
            embedding_dim, cls_dim, bias=False)
        self.num_mamba_block = num_mamba_block
        self.ssl_head = nn.Linear(cls_dim, num_embeddings)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_embeddings),
            nn.Linear(self.num_embeddings, self.cls_dim, bias=False),
            nn.PReLU(self.cls_dim),
            nn.Linear(self.cls_dim, num_classes)
        )

        self.seq_blocks = nn.LSTM(
                input_size=cls_dim, 
                hidden_size=cls_dim, 
                num_layers=num_mamba_block, 
                batch_first=True)

    @property
    def embedding_filters(self) -> Tensor:
        return self.encoder.vq.embedding.embedding.weight
    
    @property
    def filter_codebook(self) -> Tensor:
        return self.encoder.vq.embedding.embedding.weight
    
    @property
    def latent_codebook(self) -> Tensor:
        return self.encoder.embedding.weight

    def init_codebooks(self, path, freeze=True):
        self.codebook_pretrained_path = path
        self.freeze_codebook = freeze
        if path is not None:
            self.encoder.vq.embedding.from_pretrained(
                path, freeze=freeze)
            self.encoder.embedding.weight.data = self.embedding_filters.detach().clone()

    def seq_forward(self, X: Tensor) -> Tensor:
        X, _ = self.seq_blocks(X)
        return X

    def encode(self, X: Tensor) -> Tensor:
        encodings, loss = self.encoder.encode(X)
        latent = self.ssl_head(
            self.seq_forward(
                self.linear_projector(encodings)))
        return latent

    def classify(self, X: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(X)
        if latent.ndim == 3:
            latent = reduce(latent,'b n h -> b h', 'mean', h=self.num_embeddings)
        logits = self.classifier(latent)
        return logits, latent
    
    def tokenize(self, X: Tensor) -> tuple[Tensor, VqClsLoss]:
        tokens, vq_loss = self.encoder.vq(self.encoder.tokenizer(X))
        return tokens, vq_loss
    
    def train_next_token_prediction(self, X: Tensor, Y: Tensor) -> Tensor:
        X = self.encoder.embedding(X) 
        X = X + self.encoder.generate_positional_encoding(X.shape[1])
        X_next = self.ssl_head(self.seq_forward(self.linear_projector(X)))
        X_next = rearrange(X_next, 'b n h -> (b n) h')
        Y = rearrange(Y, 'b n -> (b n)')
        loss = F.cross_entropy(X_next, Y).mean() 
        return loss

    def train_embedding(self, X: Tensor) -> VqVaeClsLoss:
        """Bigram token prediction"""
        tokens, vq_loss = self.tokenize(X)
        inputs = tokens[..., :-1]
        targets = tokens[..., 1:]
        recon_loss = self.train_next_token_prediction(inputs, targets)
        return VqVaeClsLoss(
            perplexity=vq_loss.perplexity,
            loss_vq=vq_loss.loss_vq, 
            loss_recon=recon_loss,
            loss_cls=self.zero_loss)
    
    def train_embedding_cls(
            self, X: Tensor, Y: Optional[Tensor]=None
        ) -> tuple[Tensor, VqVaeClsLoss]:

        vq_loss = self.train_embedding(X)
        logits, Z = self.classify(X)
        if Y is not None:
            loss_cls = F.cross_entropy(logits.squeeze(), Y)
        else:
            loss_cls = self.encoder.zero_loss
        loss = vq_loss._replace(loss_cls=loss_cls)
        return Z, loss
    

class SCB13(nn.Module):
    """
    SCB Augmentation + IConNet
    """
    def __init__(
        self,
        in_channels: int,    
        num_embeddings: int, 
        embedding_dim: int, 
        num_classes: int, 
        window_k: int=5,
        stride: int=1,
        cls_dim: int=500,
        sample_rate: int=16000,
        distance_type: Literal['euclidean', 'dot']='euclidean',
        loss_type: Literal['overlap', 'minami', 
                            'maxami', 'mami']='overlap',
        codebook_pretrained_path: Optional[str]=None,
        freeze_codebook: bool=False,
        num_tokens_per_second: int=64,
        max_num_tokens: int=768, 
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.cls_dim = cls_dim
        self.window_k = window_k

        self.encoder = SCBWinConv(
            in_channels=in_channels,
            out_channels=num_embeddings,
            kernel_size=embedding_dim, 
            stride=stride,
            learnable_windows=True,
            shared_window=False,
            window_k=window_k,
            filter_init=codebook_pretrained_path,
            sample_rate=sample_rate, 
        )

        self.classifier = nn.Sequential(
            NLReLU(),
            nn.LayerNorm(self.num_embeddings),
            nn.Linear(self.num_embeddings, self.cls_dim, bias=False),
            nn.PReLU(self.cls_dim),
            nn.Linear(self.cls_dim, self.cls_dim, bias=False),
            nn.PReLU(self.cls_dim),
            nn.Linear(self.cls_dim, num_classes, bias=False)
        )

    @property
    def embedding_filters(self) -> Tensor:
        return self.encoder.filters
    
    @property
    def filter_codebook(self) -> Tensor:
        return self.encoder.filters * self.encoder.windows

    def encode(self, X: Tensor) -> Tensor:
        return self.encoder(X)

    def classify(self, X: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(X)
        latent = reduce(latent,'b h n -> b h', 'mean', h=self.num_embeddings)
        logits = self.classifier(latent)
        return logits
    
    def forward(self, X: Tensor) -> Tensor:
        logits = self.classify(X)
        return logits
    
    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.forward(X)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs
    

class SCB14(SCB):
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
                            'maxami', 'mami']='maxami',
        codebook_pretrained_path: Optional[str]=None,
        freeze_codebook: bool=False,
        iconnet_config = None
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
            commitment_cost=commitment_cost
        )
        self.codebook_pretrained_path = codebook_pretrained_path
        self.freeze_codebook = freeze_codebook

        self.encoder = AudioAugmentor(
            in_channels=in_channels,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            kernel_size=embedding_dim,
            stride=stride,
            commitment_cost=commitment_cost,
            distance_type=distance_type,
            loss_type=loss_type
        )  

        if codebook_pretrained_path is not None:
            self.encoder.load_codebook(
                codebook_pretrained_path, freeze_codebook)

        self.iconnet_config = iconnet_config 
        self.classifier = ModelWrapper(
            iconnet_config.name).init_model(
                iconnet_config, n_input=num_embeddings, n_output=num_classes)
        
    @property
    def embedding_filters(self) -> Tensor:
        return self.encoder.embedding_filters
    
    def classify(self, X: Tensor) -> tuple[Tensor, Tensor]:
        X = self.encoder.project(X)
        X = F.max_pool1d(nl_relu(X), kernel_size=128, stride=32)
        latent = reduce(X, 'b h n -> b h', 'mean')
        logits = self.classifier(X)
        return logits, latent
    
    def train_embedding(self, X: Tensor) -> VqVaeClsLoss:
        if self.freeze_codebook:
            return VqVaeClsLoss(
                perplexity=self.zero_loss, loss_vq=self.zero_loss,
                loss_recon=self.zero_loss, loss_cls=self.zero_loss)
        vq_loss = self.encoder.train_embedding_ssl(X)
        return VqVaeClsLoss(*vq_loss, loss_cls=self.zero_loss)
    

class SCB15(SCB14):
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
                            'maxami', 'mami']='maxami',
        codebook_pretrained_path: Optional[str]=None,
        freeze_codebook: bool=False,
        iconnet_config = None,
        num_tokens_per_second: int=16,
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
            iconnet_config=iconnet_config
        )

        self.classifier = ModelWrapper(
            iconnet_config.name).init_model(
                iconnet_config, n_input=num_embeddings, n_output=num_classes)
        
        self.num_tokens_per_second = num_tokens_per_second
        self.pad_token_layer = PadForConv(kernel_size=num_tokens_per_second)

    # def iconnet_forward(self, X: Tensor):
    #     X = rearrange(X, 'b h n -> (b h) 1 n')
    #     X = self.classifier.fe_blocks(X)
    #     X = reduce(
    #         self.pad_token_layer(X), 
    #         'bh c (m t) -> (bh m) c', 'mean', 
    #         t=self.num_tokens_per_second)
    #     logits = self.classifier.cls_head(X)
    #     logits = reduce(logits, '(b h m) l -> (b h) l', 'max')
    #     logits = reduce(logits, '(b h) l -> b l', 'mean')
    #     return logits
    
    def iconnet_forward(self, X: Tensor):
        batch_size = X.shape[0]
        X = self.classifier.fe_blocks(X)
        X = reduce(
            self.pad_token_layer(X), 
            'b h (m t) -> (b m) h', 'mean', 
            t=self.num_tokens_per_second)
        logits = self.classifier.cls_head(X)
        logits = reduce(logits, '(b m) l -> b l', 'max', b=batch_size)
        return logits
    
    def classify(self, X: Tensor) -> tuple[Tensor, Tensor]:
        X = self.encoder.project(X)
        latent = reduce(
            F.max_pool1d(nl_relu(X), kernel_size=128, stride=32), 
            'b h n -> b h', 'mean')
        logits = self.iconnet_forward(X)
        return logits, latent