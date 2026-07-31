# import library
import torch.nn as nn

from ..module import Backbone, Neck

class Yolov11(nn.Module):
    
    # initialized
    def __init__(
        self,
        depth_factor,
        width_factor,
        max_channels,
    ):
        
        # nn.Module reset to use PyTorch
        super(Yolov11, self).__init__()
        
        # Backbone
        self.backbone = Backbone(
            depth_factor=depth_factor,
            width_factor=width_factor,
            max_channels=max_channels
        )
        
        # Neck
        self.neck = Neck(
            channels=self.backbone.out_channels,
            depth_factor=depth_factor
        )
        
        # Head
        # self.head = ~~~
        
    # forward
    def forward(self, x):
        
        # Backbone
        x = self.backbone(x)
        
        # Neck
        x = self.neck(x)
        
        # Head
        # x = self.head(x)
        
        # return
        return x
        
        