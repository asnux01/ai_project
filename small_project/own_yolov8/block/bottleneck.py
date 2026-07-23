# insert library
import torch.nn as nn

from ..layer import ConvBNSiLU

class Bottleneck(nn.Module):
    
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        activation="silu"
    ):
        
        # nn.Module reset to use PyTorch
        super(Bottleneck, self).__init__()
        
        # Conv0
        self.conv0 = ConvBNSiLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=activation
        )
        
        # Conv1
        self.conv1 = ConvBNSiLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=activation
        )
        
        # shortcut
        self.shortcut = shortcut
        
    # forward
    def forward(self, x):
        
        # residual
        y = x
        
        # Conv -> Conv
        x = self.conv0(x)
        x = self.conv1(x)
        
        # Shortcut
        if self.shortcut is True:
            x = x + y
        
        # return
        return x
        