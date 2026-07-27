# import library
import torch
import torch.nn as nn

class Conv(nn.Module):
    
    # initialize
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation="silu"
    ):
        
        # nn.Module reset to use PyTorch
        super(Conv, self).__init__()
        
        # Conv2d
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        # BatchNorm
        self.bn = nn.BatchNorm2d(
            num_features=out_channels
        )
        
        # Activation
        activation = activation.lower()
        
        if activation == "silu":
            self.act = nn.SiLU
        elif activation == "relu":
            self.act = nn.ReLU
        else:
            self.act = nn.Identity()
        
    # Forward
    def forward(self, x):
        
        # Conv
        x = self.conv(x)
        
        # BatchNorm
        x = self.bn(x)
        
        # Activation
        x = self.act(x)
        
        # return
        return x