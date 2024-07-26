from typing import Optional, Literal
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from ..conv.pad import PadForConv

# TODO: LossLogger or LossWrapper
# to calculate OOD filters z and OOD samples X 
# for transfer learning case & monitoring. 

class AudioMutualInfo(nn.Module):
    def __init__(
        self,
        kernel_size: int=1024,
        stride: int=500,
        downsampling: int=8, 
        rho_value: float=10.,
        envelope_type: Literal['mean']='mean',
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.downsampling = downsampling 
        self.rho_value = rho_value
        self.pad_layer = PadForConv(
            kernel_size=downsampling, pad_mode='mean')
        self.envelope_type = envelope_type
        window = torch.hann_window(self.kernel_size)
        self.register_buffer('window', window)

    def get_envelope(self, X: Tensor):
        '''
        Input:  audio waveform shape=(B C N) 
        Output: envelope [0.,1.] shape=(B C N//stride)
        '''
        eps = torch.finfo(X.dtype).eps
        env = F.avg_pool1d(
            X.abs(), kernel_size=self.kernel_size, 
            stride=self.stride)
        env = env / torch.clamp(env.amax(dim=-1, keepdim=True), min=eps)
        return env
        
    def compute_probs(self, X: Tensor):
        '''
        Input:  audio waveform shape=(B C N) 
        Output: probability [0.,1.] shape=(B C n_tokens)
                with n_tokens = (N//stride) // downsampling
        '''
        X = self.pad_layer(self.get_envelope(X))
        px = reduce(X, '... (n ds) -> ... n', 'mean', ds=self.downsampling)
        return px
    
    def compute_entropy(self, px: Tensor):
        eps = torch.finfo(px.dtype).eps
        return -torch.sum(px*torch.log(px+eps), dim=-1)

    def mutual_information(self, X_filtered: Tensor, X: Tensor):
        '''
        Args:
            X_filtered: (X|z)  B H N , H = num_positions
            X:          (X)    B C M , C = 1
        Returns:
            MI(X,z): Mutual information of P(X) and P(z)
            H(X):    Entropy of P(X)
            H(X|z):  Conditional entropy of P(X|z)
        '''
        H = X_filtered.shape[1]
        px = repeat(
            self.compute_probs(X), 
            'b 1 n -> b h n', h=H)
        px_cz = self.compute_probs(X_filtered)
        entropy_x = self.compute_entropy(px)
        conditional_entropy_x_cz = self.compute_entropy(px_cz)
        mutual_info_xz = entropy_x - conditional_entropy_x_cz
        return mutual_info_xz, (entropy_x, conditional_entropy_x_cz)
    
    def non_positive_flip(self, X):
        """Flip sign for non positive values and multiply with rho_value:
            x > 0: 0
            x = 0: 0.5*rho_value
            x < 0: rho_value 
        """
        eps = torch.finfo(X.dtype).eps
        big_num = 100
        flip_sign = torch.sigmoid(big_num*(-X)/(X.abs()+eps))
        out = flip_sign*self.rho_value
        return out
    
    def forward(self, X_filtered: Tensor, X: Tensor):
        mutual_info, _ = self.mutual_information(X_filtered, X)
        return mutual_info

class AudioMutualInfoMask(AudioMutualInfo):
    def __init__(self,
        kernel_size: int=1024,
        stride: int=500,
        downsampling: int=8,
        threshold: float=0.
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            downsampling=downsampling,
            rho_value=1)
        self.threshold = threshold # TODO: learnable?
        
    def forward(self, X_filtered, X):
        mi, (_, _) = self.mutual_information(X_filtered, X)
        mi = 1 - self.non_positive_flip(mi-self.threshold)
        return mi


class AudioOverlapInformationLoss(AudioMutualInfo):
    def __init__(
        self, 
        kernel_size: int=1024,
        stride: int=500,
        downsampling: int=8, 
        rho_value: float=10.,
        lambda_value: float=0.2,
        loss_type: Literal['overlap', 'minami', 
                            'maxami', 'mami']='overlap',
        ensure_non_negative: bool=True,
        reduction: Optional[Literal['mean', 'sum']]=None
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            downsampling=downsampling,
            rho_value=rho_value
        )
        self.lambda_value = lambda_value
        self.loss_type = loss_type
        self.ensure_non_negative = ensure_non_negative
        self.reduction = reduction 

        if self.loss_type == 'mami':
            self.loss_fn = self.mami_loss
        elif self.loss_type == 'minami':
            self.loss_fn = self.minami_loss
        elif self.loss_type == 'maxami':
            self.loss_fn = self.maxami_loss
        else:
            self.loss_fn = self.overlap_loss
        
    def overlap_loss(self, X_filtered, X):
        ''' Loss I-solution1: minimizing overlap ratio 
                to find local features (pitch):
                
            L = MI(X,z)/[H(X)-MI(X,z)]
            MinAMI = L + penalty(L)
        '''
        mutual_info_xz, (
            entropy_x, _
        ) = self.mutual_information(
            X_filtered, X
        )
        loss = mutual_info_xz / (entropy_x - mutual_info_xz)
        loss = loss + self.non_positive_flip(loss)
        return loss

    def minami_loss(self, X_filtered, X):
        '''Loss I-solution2: minimizing overlap ratio 
                to find local features (pitch)
                
            MinAMI = MI(X,z) + penalty(H(X|z))
            robust_MinAMI = MI(X,z) + penalty(MI(X,z)*H(X|z))
        '''
        mutual_info_xz, (
            _, conditional_entropy_x_cz
        ) = self.mutual_information(
            X_filtered, X
        )
        if self.ensure_non_negative == True:
            penalty = self.non_positive_flip(
                conditional_entropy_x_cz * mutual_info_xz)
        else:
            penalty = self.non_positive_flip(
                conditional_entropy_x_cz)
        loss = mutual_info_xz + penalty
        return loss

    def maxami_loss(self, X_filtered, X):
        '''Loss II: maximizing overlap ratio 
                to find global features (timbre)
            
            MaxAMI = H(X|z) + penalty(MI(X,z)*H(X|z))
        '''
        mutual_info_xz, (
            _, conditional_entropy_x_cz
        ) = self.mutual_information(
            X_filtered, X
        )
        penalty = self.non_positive_flip(
            conditional_entropy_x_cz * mutual_info_xz)
        loss = conditional_entropy_x_cz + penalty
        return loss

    def mami_loss(self, X_filtered, X):
        '''Loss III: try balancing overlap ratio 
                    to find both local and global features 
                    (pitch & timbre)

            L(lambda>=0) = MI(X,z) - [lambda * H(X|z)]
            MAMI = L + penalty(MI(X,z)*H(X|z))
            robust_MAMI = L + penalty(L)
        '''
        mutual_info_xz, (
            _, conditional_entropy_x_cz
        ) = self.mutual_information(
            X_filtered, X
        )
        loss = mutual_info_xz - self.lambda_value * conditional_entropy_x_cz
        if self.ensure_non_negative == True:
            penalty = self.non_positive_flip(
                loss)
        else:
            penalty = self.non_positive_flip(
                conditional_entropy_x_cz * mutual_info_xz)
        loss = loss + penalty
        return loss

    def reduce(self, loss):
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    def forward(self, X_filtered, X):
        '''
        Args:
            X_filtered: (X|z)  B H N , H = num_positions
            X:          (X)    B C M , C = 1
        '''
        loss = self.loss_fn(X_filtered, X)
        loss = self.reduce(loss)
        return loss
