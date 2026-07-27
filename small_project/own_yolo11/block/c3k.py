# import library
import torch
import torch.nn as nn

from ..layer import Conv
from .bottleneck import Bottleneck

class C3K (nn.Module):
    
    # initialized
    def __init__(
        self,
        channels,
        n,
        shortcut=True,
        activation="silu"
    ):
        
        # nn.Module reset to use PyTorch
        super(C3K, self).__init__()
        
        # parameter
        ch0 = channels
        ch1 = channels // 2
        cnt = n
        
        # branch0 Conv: bottleneck X
        self.conv0 = Conv(
            in_channels=ch0,
            out_channels=ch1,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation
        )
        
        # branch1 Conv: bottleneck O
        self.conv1 = Conv(
            in_channels=ch0,
            out_channels=ch1,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation
        )
        
        # Bottleneck list
        self.bottlenecks = nn.ModuleList()
                        
        # Bottleneck append
        for _ in range(cnt):
            bottleneck = Bottleneck(
                channels=ch1,
                shortcut=shortcut,
                activation=activation
            )
                
            self.bottlenecks.append(bottleneck)
    
        # After concat Conv
        self.conv2 = Conv(
            in_channels=ch0,
            out_channels=ch0,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation
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