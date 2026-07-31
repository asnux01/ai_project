# import library
import torch
import torch.nn as nn

from ..layer import Conv

class BoxBranch(nn.Module):
    
    # initialized
    def __init__(
        self,
        in_channels,
        hidden_channels,
        reg_max=16
    ):
        
        # nn.Module reset to use PyTorch
        super(BoxBranch, self).__init__()
        
        # parameter
        ch_i = in_channels
        ch_h = hidden_channels
        ch_o = 4 * reg_max
        
        # Conv0
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_h,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv1
        self.conv1 = Conv(
            in_channels=ch_h,
            out_channels=ch_h,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv2d
        self.conv2d = nn.Conv2d(
            in_channels=ch_h,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
    # forward
    def forward(self, x):
        
        # box branch
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2d(x)
        
        # return
        return x