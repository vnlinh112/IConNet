import torch 
import torch.nn as nn

class SeqBlocks(nn.Module):
    """
    C: n_input_channel
    H: n_output_channel
    N: input sequence length
    (B C N) => (B H N)
    Output: logits
    """
    def __init__(
            self, 
            n_block: int, 
            n_input_channel: int, 
            n_output_channel: int):
        super().__init__()
        self.blocks = nn.LSTM(
                input_size=n_input_channel, 
                hidden_size=n_output_channel, 
                num_layers=n_block, 
                batch_first=True)
    
    def forward(self, x):
        x, _ = self.blocks(x)
        return x