# import library
import torch.nn as nn

from .module import Backbone, Neck, Head

class Yolov11(nn.Module):
    
    # initialized
    def __init__(
        self,
        num_classes,
        depth_factor,
        width_factor,
        max_channels,
        reg_max=16,
        strides=(8,16,32)
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
        
        # Detect Head
        self.head = Head(
            num_classes=num_classes,
            in_channels=self.neck.out_channels,
            reg_max=reg_max,
            strides=strides
        )
        
    # forward
    def forward(self, x):
        
        # Backbone
        features = self.backbone(x)
        
        # Neck
        features = self.neck(features)
        
        # Head
        x = self.head(features)
        
        # return
        return x
        
        