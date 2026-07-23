# insert library
import torch
import torch.nn as nn

from ..layer import ConvBNSiLU

class SPPF(nn.Module):
    
    def __init__(
        self,
        in_channels,
        out_channels,
        activation="silu"
    ):
        
        # nn.Module reset to use PyTorch
        super(SPPF, self).__init__()
        
        # hidden channels
        hidden_channels = in_channels // 2
        
        # input Conv
        self.conv0 = ConvBNSiLU(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation
        )
        
        # maxpool
        self.maxpool = nn.MaxPool2d(
            kernel_size=5,
            stride=1,
            padding=2
        )
        
        # concated channels
        concat_channels = hidden_channels * 4
        
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
        
        # Maxpool
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        
        # Concat
        x = torch.cat(
            (x, x1, x2, x3),
            dim=1
        )
        
        # output Conv
        x = self.conv1(x)
        
        # return
        return x