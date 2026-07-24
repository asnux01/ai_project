# insert library
import torch.nn as nn

from ..module import Backbone, Neck, Head

class Yolov8(nn.Module):
    
    def __init__(
        self,
        deepen_factor,
        widen_factor,
        ratio,
    ):
        
        # nn.Module reset to use PyTorch
        super(Yolov8, self).__init__()
        
        # parameter
        activation = "silu"
        shortcut_Ture = True
        shortcut_False = False
        
        # Backbone
        self.backbone = Backbone(
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            ratio=ratio,
            shortcut=shortcut_Ture,
            activation=activation
        )
        
        # Neck
        self.neck = Neck(
            in_channels=self.backbone.out_channels,
            deepen_factor=deepen_factor,
            shortcut=shortcut_False,
            activation=activation
        )
        
        # 256xw Head
        self.head0 = Head(
            in_channels=self.neck.out_channels[0],
            activation=activation,
        )
        
        # 512xw Head
        self.head1 = Head(
            in_channels=self.neck.out_channels[1],
            activation=activation,
        )
        
        # 512xwxr Head
        self.head2 = Head(
            in_channels=self.neck.out_channels[2],
            activation=activation,
        )
        
    # forward
    def forward(self, x):
        
        # Backbone
        x = self.backbone(x)
        
        # Neck
        x = self.neck(x)
        
        # 256xw head
        y0 = self.head0(x[0])
        
        # 512xw head
        y1 = self.head1(x[1])
        
        # 512xw head
        y2 = self.head2(x[2])
        
        # return
        return [y0, y1, y2]