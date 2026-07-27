# import library
import torch
import torch.nn as nn

from ..layer import Conv
from .bottleneck import Bottleneck
from .c3k import C3K

class C3K2(nn.Module):
    
    # initialized
    def __init__(
        self,
        in_channels,
        out_channels,
        c3k,
        n,
        shortcut,
        e=0.5
    ):
        
        # nn.Module reset to use PyTorch
        super(C3K2, self).__init__()
        
        # parameter
        ch_i = in_channels              # input channels
        ch_o = out_channels             # output channels
        ch_h = int(out_channels * e)    # hidden channels
        ch_s = 2 * ch_h                 # channels before split
        ch_c = (n + 2) * ch_h           # channels after concat
        bn_cnt = n                      # bottleneck counter
        self.c3k = c3k                  # choose c3k or bottleneck
        
        # Conv before Split 
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_s,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # C3K
        self.c3k0 = C3K(
            in_channels=ch_h,
            out_channels=ch_h,
            n=bn_cnt,
            shortcut=shortcut
        )
        
        # Bottleneck
        # Bottleneck list
        self.bottlenecks = nn.ModuleList()
                                
        # Bottleneck append
        for _ in range(bn_cnt):
            bottleneck = Bottleneck(
                in_channels=ch_h,
                out_channels=ch_h,
                shortcut=shortcut
            )
                        
            self.bottlenecks.append(bottleneck)

        # Conv after Concat
        self.conv1 = Conv(
            in_channels=ch_s,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    # forward
    def forward(self, x):
        
        # Conv before split
        x = self.conv0(x)
        
        # Split x0, x1
        # x0: not pass C3K or Bottleneck
        # x1: pass C3K or Bottleneck
        x0, x1 = self.conv0(x).chunk(
            chunks=2,
            dim=1
        )
        
        # C3K or Bottleneck
        if self.c3k is True:
            