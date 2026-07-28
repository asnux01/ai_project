# import library
import torch
import torch.nn as nn

from ..layer import Conv
from ..block import C3K2, C2PSA, SPPF

class Backbone(nn.Module):
    
    # initialized
    def __init__(
        self,
        depth_factor,
        width_factor,
        max_channels,
        shortcut=True
    ):
        
        # nn.Module reset to use PyTorch
        super(Backbone, self).__init__()
        
        # parameter
        # channels
        ch_min64 = min(64, max_channels) * width_factor
        ch_min128 = min(128, max_channels) * width_factor
        ch_min256 = min(256, max_channels) * width_factor
        ch_min512 = min(512, max_channels) * width_factor
        ch_min1024 = min(1024, max_channels) * width_factor
        
        # counter factor
        n = 2 * depth_factor
        
        # output channels
        self.out_channels = [
            ch_min256,
            ch_min512,
            ch_min1024
        ]
        
        # stage 1
        # Conv
        self.conv0 = Conv(
            in_channels=3,
            out_channels=ch_min64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # stage 2
        # Conv
        self.conv1 = Conv(
            in_channels=ch_min64,
            out_channels=ch_min128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # C3K2
        self.c3k2_0 = C3K2(
            in_channels=ch_min128,
            out_channels=ch_min256,
            c3k=False,
            n=n,
            e=0.25
        )
        
        # stage 3
        # Conv
        self.conv2 = Conv(
            in_channels=ch_min256,
            out_channels=ch_min256,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # C3K2
        self.c3k2_1 = C3K2(
            in_channels=ch_min256,
            out_channels=ch_min512,
            c3k=False,
            n=n,
            e=0.25
        )
        
        # stage 4
        # Conv
        self.conv3 = Conv(
            in_channels=ch_min512,
            out_channels=ch_min512,
            kernel_size=3,
            stride=2,
            padding=1
        )
                
        # C3K2
        self.c3k2_2 = C3K2(
            in_channels=ch_min512,
            out_channels=ch_min512,
            c3k=True,
            n=n
        )
        
        # stage 5
        # Conv
        self.conv4 = Conv(
            in_channels=ch_min512,
            out_channels=ch_min1024,
            kernel_size=3,
            stride=2,
            padding=1
        )
                        
        # C3K2
        self.c3k2_3 = C3K2(
            in_channels=ch_min1024,
            out_channels=ch_min1024,
            c3k=True,
            n=n
        )
        
        # stage 6
        # SPPF
        self.sppf = SPPF(
            in_channels=ch_min1024,
            out_channels=ch_min1024
        )
        
        # C2PSA
        self.c2psa = C2PSA(
            in_channels=ch_min1024,
            out_channels=ch_min1024,
            n=n,
            shortcut=shortcut,
        )
        
    # forward
    def forward(self, x):
        
        # stage 1
        x = self.conv0(x)
        
        # stage 2
        x = self.conv1(x)
        x = self.c3k2_0(x)
        
        # stage 3
        x = self.conv2(x)
        x = self.c3k2_1(x)
        y0 = x
        
        # stage 4
        x = self.conv3(x)
        x = self.c3k2_2(x)
        y1 = x
        
        # stage 5
        x = self.conv4(x)
        x = self.c3k2_3(x)
        
        # stage 6
        x = self.sppf(x)
        x = self.c2psa(x)
        y2 = x
        
        # return
        return [y0, y1, y2]