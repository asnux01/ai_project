# import library
import torch
import torch.nn as nn

from ..layer import Conv

class Bottleneck(nn.Module):
    
    # initialize
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True
    ):
        
        # nn.Module reset to use PyTorch
        super(Bottleneck, self).__init__()
        
        # parameter
        ch_i = in_channels
        ch_h = in_channels
        ch_o = out_channels
          
        # Conv
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_h,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        self.conv1 = Conv(
            in_channels=ch_h,
            out_channels=ch_o,
            kernel_size=3,
            stride=1,
            padding=1,
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