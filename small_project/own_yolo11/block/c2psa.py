# import library
import torch
import torch.nn as nn

from ..layer import Conv
from .psa import PSABlock


class C2PSA(nn.Module):
    
    # initialize
    def __init__(
        self,
        in_channels,
        out_channels,
        n,
        e=0.5
    ):
        
        # nn.Module reset to use PyTorch
        super(C2PSA, self).__init__()
        
        # parameter
        ch_i = in_channels          # input channels
        ch_o = out_channels         # output channels
        ch_h = int(ch_o * e)        # hidden channels
        ch_s = 2 * ch_h             # channels before split
        ch_c = 2 * ch_h             # channels after concat
        psa_cnt = n                 # PSABlock counter
        
        # save hidden channels
        self.hidden_channels = ch_h
        
        # Attention head counter
        head_cnt = ch_h // 64

        
        # Conv before Split
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_s,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # PSABlock 
        # PSABlock list
        self.blocks = nn.ModuleList()
        
        # PSABlock append
        for _ in range(psa_cnt):
            block = PSABlock(
                in_channels=ch_h,
                out_channels=ch_h,
                attn_ratio=0.5,
                num_heads=head_cnt,
                shortcut=True
            )
            
            self.blocks.append(block)
        
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
        
        # Conv before Split
        x = self.conv0(x)
        
        # Split x0 and x1
        x0, x1 = x.split(
            [
                self.hidden_channels,
                self.hidden_channels
            ],
            dim=1
        )
        
        # PSABlock
        for block in self.blocks:
            x1 = block(x1)
        
        # Concat
        x = torch.cat(
            [x0, x1],
            dim=1
        )
        
        # Conv after Concat
        x = self.conv1(x)
        
        # return
        return x