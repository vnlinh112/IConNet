from .signal import visualize_window, general_cosine_window
from .FIRConv import Downsample2
from einops import rearrange, reduce
import torch
import torch.nn as nn

class M9(nn.Module):
    def __init__(self, n_input=1, n_output=35, 
                 stride=16, n_channel=128, eps=1e-12):
        super().__init__()
        self.conv1 = Downsample2(n_input, n_channel, 
                                kernel_size=511, stride=2, window_k=2)
        self.conv3 = Downsample2(n_channel, n_channel, 
                                kernel_size=127, stride=4, window_k=3)
        self.layer_norm = nn.LayerNorm(3*n_channel)
        self.eps = eps
        self.cls_head = nn.Sequential(
            nn.Linear(3*n_channel, 2*n_channel),
            nn.PReLU(2*n_channel),
            nn.Linear(2*n_channel, n_output)
        )

        self.act = NLReLU(beta=1.0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.act(x1)
        x3 = self.act(self.conv3(x2))
        x1 = reduce(x1, 'b c n -> b () c', 'mean')
        x2 = reduce(x2, 'b c n -> b () c', 'mean')
        x3 = reduce(x3, 'b c n -> b () c', 'mean')
        x = torch.cat([x1, x2, x3], dim=-1)
        x = rearrange(x, 'b 1 c -> b c')
        x = self.layer_norm(x)
        x = self.cls_head(x)
        return x 

class NLReLU(nn.Module):
    def __init__(self, beta=1.):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return nl_relu(x, self.beta)
        
def nl_relu(x, beta=1.):
    return torch.log(1 + beta * F.relu(x)) 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# model = M9(n_input=1, n_output=n_classes)
# print(model)

# print("Number of parameters: %s" % count_parameters(model))