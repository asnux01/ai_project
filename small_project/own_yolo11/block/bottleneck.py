# import library
import torch
import torch.nn as nn

from ..layer import Conv

class Bottleneck(nn.Module):
    
    # initialize
    def __init__(
        self,
        channels,
        shortcut=True,
        activation="silu"
    ):
        
        # nn.Module reset to use PyTorch
        super(Bottleneck, self).__init__()
        
        # parameter
        ch0 = channels
          
        # Conv
        self.conv0 = Conv(
            in_channels=ch0,
            out_channels=ch0,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=activation
        )
        
        self.conv1 = Conv(
            in_channels=ch0,
            out_channels=ch0,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=activation
        )
        
        # Shortcut parameter
        self.shortcut = shortcut
        
    # forward
    def forward(self, x):
        
        # Residual
        res = x
        
        # Conv
        x = self.conv0(x)
        x = self.conv1(x)
        
        # detect shortcut
        if self.shortcut:
            x = res + x
        
        # return
        return x