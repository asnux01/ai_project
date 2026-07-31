# import library
import torch
import torch.nn as nn

from ..layer import Conv

class ClassBranch(nn.Module):
    
    # initialized
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes
    ):
        
        # nn.Module reset to use PyTorch
        super(ClassBranch, self).__init__()
        
        # parameter
        ch_i = in_channels
        ch_h = hidden_channels
        ch_o = num_classes
        self.nc = num_classes
        
        # DWConv0
        self.dwconv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_i,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=ch_i
        )
                
        # Conv0
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
        )
                
        # DWConv1
        self.dwconv1 = Conv(
            in_channels=ch_h,
            out_channels=ch_h,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=ch_h
        )
                        
        # Conv1
        self.conv1 = Conv(
            in_channels=ch_h,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
        )
                
        # Conv2d
        self.conv2d = nn.Conv2d(
            in_channels=ch_h,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
    # forward
    def forward(self, x):
        
        # class branch
        x = self.dwconv0(x)
        x = self.conv0(x)
        x = self.dwconv1(x)
        x = self.conv1(x)
        x = self.conv2d(x)
        
        # return
        return x