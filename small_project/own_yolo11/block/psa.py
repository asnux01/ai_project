# import library
import torch
import torch.nn as nn

from ..layer import Conv
from .attention import Attention

class PSABlock(nn.Module):
    
    # initialized
    def __init__(
        self,
        in_channels,
        out_channels,
        attn_ratio=0.5,
        num_heads=4,
        shortcut=True
    ):
        
        # nn.Module reset to use PyTorch
        super(PSABlock, self).__init__()
        
        # parameter
        ch_i = in_channels          # input channels
        ch_h = 2 * out_channels     # hidden channels
        ch_o = out_channels         # output channels
        
        # Attention
        self.attention = Attention(
            in_channels=ch_i,
            out_channels=ch_o,
            num_heads=num_heads,
            attn_ratio=attn_ratio
        )
        
        # FFN Conv0
        self.ffn_conv0 = Conv(
            in_channels=ch_o,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # FFN Conv1
        self.ffn_conv1 = Conv(
            in_channels=ch_h,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0,
            activation="identity"
        )
        
        # Attention shortcut
        self.attention_shortcut = (
            shortcut
            and ch_i == ch_o
        )
        
        # FFN shortcut
        self.ffn_shortcut = shortcut
    
    # forward
    def forward(self, x):
        
        # Attention residual
        res = x
        
        # Attention
        x = self.attention(x)
        
        # Attention residual
        if self.attention_shortcut:
            x = res + x
        
        # FFN residual
        res = x
        
        # FFN
        x = self.ffn_conv0(x)
        x = self.ffn_conv1(x)
        
        # FFN residual
        if self.ffn_shortcut:
            x = res + x
        
        # return
        return x