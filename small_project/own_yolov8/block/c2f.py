# 라이브러리 삽입
import torch
import torch.nn as nn

from ..layer import ConvBNSiLU
from .bottleneck import Bottleneck

class C2F(nn.Module):
    
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_count = 1,
        shortcut = True,
        activation = "silu"
    ):
        
        # nn.Module reset to use PyTorch
        super(C2F, self).__init__()
        
        # split channels
        split_channels = out_channels // 2
        
        # input Conv
        self.conv0 = ConvBNSiLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation
        )
        
        # Bottleneck list
        self.bottlenecks = nn.ModuleList()
                
        # Bottleneck append
        for _ in range(bottleneck_count):
            bottleneck = Bottleneck(
                in_channels = split_channels,
                out_channels = split_channels,
                shortcut = shortcut,
                activation = activation
            )
        
            self.bottlenecks.append(bottleneck)

         # concated channels
        concat_channels = split_channels * (bottleneck_count + 2)
                
        # output Conv
        self.conv1 = ConvBNSiLU(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation
        )
        
    # forward
    def forward(self, x):
        
        # input Conv
        x = self.conv0(x)
        
        # Split
        x1, x2 = x.chunk(2, dim=1)
        features = [x1, x2]
        
        # bottleneck
        for bottleneck in self.bottlenecks:
            out = bottleneck(features[-1])
            
            features.append(out)
            
        # Concat
        x = torch.cat(
            features,
            dim=1
        )
        
        # output Conv
        x = self.conv1(x)
        
        # return
        return x