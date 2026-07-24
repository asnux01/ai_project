# insert library
import torch
import torch.nn as nn

from ..layer import ConvBNSiLU
from ..block import C2F

class Neck(nn.Module):
    
    def __init__(
        self,
        in_channels,
        deepen_factor,
        shortcut=False,
        activation="silu"
    ):
        
        # nn.Module reset to use PyTorch
        super(Neck, self).__init__()
        
        # channel parameter
        channels_256w = in_channels[0]
        channels_512w = in_channels[1]
        channels_512wr = in_channels[2]
        channels_512w1pr = in_channels[2] + in_channels[1]
        channels_768w = in_channels[0] + in_channels[1]
        
        # bottleneck counter parameter
        bottleneck_count = 3 * deepen_factor
        
        # Upsampling
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )
        
        # TopDown Layer1
        self.c2f0 = C2F(
            in_channels=channels_768w,
            out_channels=channels_256w,
            bottleneck_count=bottleneck_count,
            shortcut=shortcut,
            activation=activation
        )
        
        # TopDown Layer2
        self.c2f1 = C2F(
            in_channels=channels_512w1pr,
            out_channels=channels_512w,
            bottleneck_count=bottleneck_count,
            shortcut=shortcut,
            activation=activation
        )
        
        # Down Sample0
        self.conv0 = ConvBNSiLU(
            in_channels=channels_256w,
            out_channels=channels_256w,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation
        )
        
        # BottomUp Layer0
        self.c2f2 = C2F(
            in_channels=channels_768w,
            out_channels=channels_512w,
            bottleneck_count=bottleneck_count,
            shortcut=shortcut,
            activation=activation
        )
        
        # Down Sample1
        self.conv1 = ConvBNSiLU(
            in_channels=channels_512w,
            out_channels=channels_512w,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation
        )
        
        # BottomUp Layer0
        self.c2f3 = C2F(
            in_channels=channels_512w1pr,
            out_channels=channels_512wr,
            bottleneck_count=bottleneck_count,
            shortcut=shortcut,
            activation=activation
        )
        
        self.out_channels = [
            in_channels[0],
            in_channels[1],
            in_channels[2]
        ]
        
    # forward
    def forward(self, x):
        
        # first Upsampling
        y = self.upsample(x[2])
        
        # first Concat
        y = torch.cat(
            [y, x[1]],
            dim=1
        )
        
        # TopDown Layer2
        y = self.c2f1(y)
        tmp = y
        
        # second Upsampling
        y = self.upsample(y)
        
        # second Concat
        y = torch.cat(
            [y, x[0]],
            dim=1
        )
        
        # TopDown Layer1
        y = self.c2f0(y)
        head0 = y
        
        # Down Sample0
        y = self.conv0(y)
        
        # third Concat
        y = torch.cat(
            [y, tmp],
            dim=1
        )
        
        # BottomUp Layer0
        y = self.c2f2(y)
        head1 = y
        
        # Down Sample1
        y = self.conv1(y)
        
        # third Concat
        y = torch.cat(
            [y, x[2]],
            dim=1
        )
        
        # BottomUp Layer0
        y = self.c2f3(y)
        head2 = y
        
        # return
        return [head0, head1, head2]