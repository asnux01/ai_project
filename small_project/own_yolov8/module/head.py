# insert library
import torch
import torch.nn as nn

from ..layer import ConvBNSiLU

class Head(nn.Module):
    
    def __init__(
        self,
        in_channels,
        activation="silu"
    ):
        
        # parameter
        out_channels = 3    # fake value
        
        # nn.Module reset to use PyTorch
        super(Head, self).__init__()
        
        # Conv block
        self.conv_block = ConvBNSiLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=activation
        )
        
        # Conv2d
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
    # Forward
    def forward(self, x):
        # Conv block1
        x = self.conv_block(x)
        
        # Conv block2
        x = self.conv_block(x)
        
        # Conv2d
        x = self.conv2d(x)
        
        # return
        return x