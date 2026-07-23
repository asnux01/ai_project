# insert library
import torch.nn as nn

# Conv → BatchNorm → activation
class ConvBNSiLU(nn.Module):

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
        super(ConvBNSiLU, self).__init__()

        # Conv
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = False
        )

        # BatchNorm
        self.bn = nn.BatchNorm2d(
            num_features = out_channels
        )

        # Activation
        activation = activation.lower()
        
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        
        # Conv → BatchNorm → Activation
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x