# import library
import torch
import torch.nn as nn

from ..layer import Conv

class SPPF(nn.Module):
    
    # initialize
    def __init__(
        self,
        channels,
        activation="silu"
    ):
        
        # nn.Module reset to use PyTorch
        super(SPPF, self).__init__()
        
        # parameter
        ch0 = channels
        ch1 = channels // 2
        ch2 = ch1 * 4
        
        # Conv0
        self.conv0 = Conv(
            in_channels=ch0,
            out_channels=ch1,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation
        )
        
        # Maxpool2d
        self.maxpool2d = nn.MaxPool2d(
            kernel_size=5,
            stride=1,
            padding=2
        )
        
        # Concated channels
        
        # Conv1
        self.conv1 = Conv(
            in_channels=ch2,
            out_channels=ch0,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation
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