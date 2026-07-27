# import library
import torch
import torch.nn as nn

from ..layer import Conv
from .bottleneck import Bottleneck

class C3K (nn.Module):
    
    # initialized
    def __init__(
        self,
        in_channels,
        out_channels,
        n,
        shortcut=True
    ):
        
        # nn.Module reset to use PyTorch
        super(C3K, self).__init__()
        
        # parameter
        ch_io = in_channels         # I/O channels
        ch_h = out_channels // 2    # hidden channels
        bn_cnt = n                  # bottleneck counter
        
        # branch0 Conv: bottleneck X
        self.conv0 = Conv(
            in_channels=ch_io,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # branch1 Conv: bottleneck O
        self.conv1 = Conv(
            in_channels=ch_io,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
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
    
        # After concat Conv
        self.conv2 = Conv(
            in_channels=ch_io,
            out_channels=ch_io,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    # forward
    def forward(self, x):
        
        # branch0 Conv
        x0 = self.conv0(x)
        
        # branch1 Conv
        x1 = self.conv1(x)
        
        # Bottleneck
        for bottleneck in self.bottlenecks:
            x1 = bottleneck(x1)
        
        # Concat
        x = torch.cat(
            [x0, x1],
            dim=1
        )
        
        # after concat Conv
        x = self.conv2(x)
        
        # return
        return x