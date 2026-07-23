# insert library
import torch.nn as nn

from ..layer import ConvBNSiLU
from ..block import C2F, SPPF

class Backbone(nn.Module):
    
    def __init__(
        self,
        deepen_factor,
        widen_factor,
        ratio,
        shortcut=True,
        activation="silu"
    ):
        # nn.Module reset to use PyTorch
        super(Backbone, self).__init__()
                
        # channel parameter
        input_channels = 3
        hidden_channels = 64
        stem_channels = hidden_channels * widen_factor
        stage1_channels = 2 * hidden_channels * widen_factor
        stage2_channels = 4 * hidden_channels * widen_factor
        stage3_channels = 8 * hidden_channels * widen_factor
        stage4_channels = 8 * hidden_channels * widen_factor * ratio
        
        # bottleneck counter parameter
        bottleneck_count3 = 3 * deepen_factor
        bottleneck_count6 = 6 * deepen_factor
        
        # Stem layer
        # Conv
        self.conv0 = ConvBNSiLU(
            in_channels=input_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation
        )
        
        # Stage Layer1
        # Conv
        self.conv1 = ConvBNSiLU(
            in_channels=stem_channels,
            out_channels=stage1_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation
        )
        
        # C2F
        self.c2f0 = C2F(
            in_channels=stage1_channels,
            out_channels=stage1_channels,
            bottleneck_count=bottleneck_count3,
            shortcut=shortcut,
            activation=activation
        )
        
        # Stage Layer2
        # Conv
        self.conv2 = ConvBNSiLU(
            in_channels=stage1_channels,
            out_channels=stage2_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation
        )
        
        # C2F
        self.c2f1 = C2F(
            in_channels=stage2_channels,
            out_channels=stage2_channels,
            bottleneck_count=bottleneck_count6,
            shortcut=shortcut,
            activation=activation
        )
        
        # Stage Layer3
        # Conv
        self.conv3 = ConvBNSiLU(
            in_channels=stage2_channels,
            out_channels=stage3_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation
        )
                
        # C2F
        self.c2f2 = C2F(
            in_channels=stage3_channels,
            out_channels=stage3_channels,
            bottleneck_count=bottleneck_count6,
            shortcut=shortcut,
            activation=activation
        )
        
        # Stage Layer4
        # Conv
        self.conv4 = ConvBNSiLU(
            in_channels=stage3_channels,
            out_channels=stage4_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation
        )
                
        # C2F
        self.c2f3 = C2F(
            in_channels=stage4_channels,
            out_channels=stage4_channels,
            bottleneck_count=bottleneck_count3,
            shortcut=shortcut,
            activation=activation
        )
        
        # SPPF
        self.sppf = SPPF(
            in_channels=stage4_channels,
            out_channels=stage4_channels,
            activation=activation
        )
        
        self.out_channels = [
            stage2_channels,
            stage3_channels,
            stage4_channels
        ]
        
    # forward
    def forward(self, x):
        
        # Stem layer
        x = self.conv0(x)
        
        # Stage Layer1
        x = self.conv1(x)
        x = self.c2f0(x)
        
        # Stage Layer2
        x = self.conv2(x)
        x = self.c2f1(x)
        stage2_out = x
        
        # Stage Layer3
        x = self.conv3(x)
        x = self.c2f2(x)
        stage3_out = x
        
        # Stage Layer4
        x = self.conv4(x)
        x = self.c2f3(x)
        x = self.sppf(x)
        stage4_out = x
        
        # return
        return (stage2_out, stage3_out, stage4_out)