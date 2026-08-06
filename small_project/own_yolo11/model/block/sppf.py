# import library
import torch
import torch.nn as nn

from ..layer import Conv

class SPPF(nn.Module):
    
    # initialize
    def __init__(
        self,
        in_channels,
        out_channels,
        k=5
    ):
        
        # nn.Module reset to use PyTorch
        super(SPPF, self).__init__()
        
        # parameter
        ch_i = in_channels          # in channels
        ch_o = out_channels         # out channels
        ch_h = out_channels // 2    # hidden channels
        ch_c = ch_h * 4             # concated channels
        mp_k = k                    # maxpool kernel size
        
        # Conv0
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Maxpool2d
        self.maxpool2d = nn.MaxPool2d(
            kernel_size=mp_k,
            stride=1,
            padding=2
        )
        
        # Concated channels
        
        # Conv1
        self.conv1 = Conv(
            in_channels=ch_c,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    # forward
    def forward(self, x):
        
        # Conv0
        x = self.conv0(x)
        
        # Maxpool
        x1 = self.maxpool2d(x)
        x2 = self.maxpool2d(x1)
        x3 = self.maxpool2d(x2)
        
        # Concat
        x = torch.cat(
            [x, x1, x2, x3],
            dim=1
        )
        
        # Conv1
        x = self.conv1(x)
        
        # return
        return x