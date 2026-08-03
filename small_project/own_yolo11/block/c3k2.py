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
        shortcut=True,
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
        
        # Conv before Split 
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_s,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Bottleneck or C3K
        # if ck3 is true: ck3
        # if ck3 is false: bottleneck
        # Ck3 and Bottleneck list
        self.blocks = nn.ModuleList()
                                
        # Both append
        for _ in range(bn_cnt):
            if c3k:
                block = C3K(
                    in_channels=ch_h,
                    out_channels=ch_h,
                    n=2,
                    shortcut=shortcut,
                )
            else:
                block = Bottleneck(
                    in_channels=ch_h,
                    out_channels=ch_h,
                    shortcut=shortcut
                )
                        
            self.blocks.append(block)

        # Conv after Concat
        self.conv1 = Conv(
            in_channels=ch_c,
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
        # x0: not passing C3K or Bottleneck
        # x1: passing C3K or Bottleneck
        x0, x1 = x.chunk(
            chunks=2,
            dim=1
        )
        
        # make list
        y = [x0, x1]
        
        # passing C3K or Bottleneck
        for block in self.blocks:
            x1 = block(x1)
            y.append(x1)
            
        # Concat
        x = torch.cat(
            y,
            dim=1
        )
        
        # Conv after Concat
        x = self.conv1(x)
        
        # return
        return x