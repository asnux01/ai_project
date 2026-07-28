# import library
import torch
import torch.nn as nn

from ..layer import Conv
from ..block import C3K2

class Neck(nn.Module):
    
    # initialized
    def __init__(
        self,
        channels,
        depth_factor,
        shortcut=True
    ):
        
        # nn.Module reset to use PyTorch
        super(Neck, self).__init__()
        
        # parameter
        ch_min256 = channels[0]             # channels of min256
        ch_min512 = channels[1]             # channels of min512
        ch_min1024 = channels[2]            # channels of min1024
        ch_c_15 = ch_min512 + ch_min1024    # concated channels min1024 + min512
        ch_c_25 = ch_min256 + ch_min512     # concated channels min256 * min512
        ch_c_55 = ch_min512 + ch_min512     # concated channels min512 + min512
        
        # counter factor
        n = 2 * depth_factor
        
        # output channels
        self.out_channels = [
            ch_min256,
            ch_min512,
            ch_min1024
        ]

        # Upsampling
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )
        
        # Topdown1
        # C3K2
        self.c3k2_0 = C3K2(
            in_channels=ch_c_15,
            out_channels=ch_min512,
            c3k=False,
            n=n,
            shortcut=shortcut
        )
        
        # Topdown2
        # C3K2
        self.c3k2_1 = C3K2(
            in_channels=ch_c_55,
            out_channels=ch_min256,
            c3k=False,
            n=n,
            shortcut=shortcut
        )
        
        # Bottomup1
        # Conv
        self.conv0 = Conv(
            in_channels=ch_min256,
            out_channels=ch_min256,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # C3K2
        self.c3k2_2 = C3K2(
            in_channels=ch_c_25,
            out_channels=ch_min512,
            c3k=False,
            n=n,
            shortcut=shortcut
        )
        
        # Bottomup1
        # Conv
        self.conv1 = Conv(
            in_channels=ch_min512,
            out_channels=ch_min512,
            kernel_size=3,
            stride=2,
            padding=1
        )
                
        # C3K2
        self.c3k2_3 = C3K2(
            in_channels=ch_c_15,
            out_channels=ch_min1024,
            c3k=True,
            n=n,
            shortcut=shortcut
        )
    
    # forward
    def forward(self, x):
        
        # transport to Bottomup
        trans0 = x[2]
        
        # Topdown1
        # first Upsampling
        y = self.upsample(x[2])
        
        # Concat
        y = torch.cat(
            [y, x[1]],
            dim=1
        )
        
        # C3K2
        y = self.c3k2_0(y)
        trans1 = y
        
        # Topdown2
        # second Upsampling
        y = self.upsample(y)
                
        # Concat
        y = torch.cat(
            [y, x[0]],
            dim=1
        )
                
        # C3K2
        y = self.c3k2_1(y)
        
        # transport to head
        head0 = y
        
        # Bottomup1
        # Conv
        y = self.conv0(y)
        
        # Concat
        y = torch.cat(
            [y, trans1],
            dim=1
        )
        
        # C3K2
        y = self.c3k2_2(y)
        
        # transport to head
        head1 = y
        
        # Bottomup1
        # Conv
        y = self.conv1(y)
                
        # Concat
        y = torch.cat(
            [y, trans0],
            dim=1
        )
                
        # C3K2
        y = self.c3k2_3(y)
        
        # transport to head
        head2 = y
        
        # return
        return [head0, head1, head2]