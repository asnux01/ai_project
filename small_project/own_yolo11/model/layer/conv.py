# import library
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
        groups=1,
        activation="silu"
    ):
        
        # nn.Module reset to use PyTorch
        super(Conv, self).__init__()
        
        # parameter
        ch_i = in_channels      # input channels
        ch_o = out_channels     # output channels
        
        # Conv2d
        self.conv = nn.Conv2d(
            in_channels=ch_i,
            out_channels=ch_o,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        
        # BatchNorm
        self.bn = nn.BatchNorm2d(
            num_features=ch_o
        )
        
        # Activation
        activation = activation.lower()
        
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        
    # forward
    def forward(self, x):
        
        # Conv
        x = self.conv(x)
        
        # BatchNorm
        x = self.bn(x)
        
        # Activation
        x = self.act(x)
        
        # return
        return x