# import library
import torch
import torch.nn as nn

from ..layer import Conv

class DetectBranch(nn.Module):
    
    # initialized
    def __init__(
        self,
        in_channels,
        box_hidden_channels,
        class_hidden_channels,
        nc,
        reg_max=16
    ):
        
        # nn.Module reset to use PyTorch
        super(DetectBranch, self).__init__()
        
        # parameter
        ch_i = in_channels
        ch_bh = box_hidden_channels
        ch_ch = class_hidden_channels
        ch_bo = 4 * reg_max
        ch_co = nc
        
        # Box Branch
        # Conv0
        self.box_conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_bh,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv1
        self.box_conv1 = Conv(
            in_channels=ch_bh,
            out_channels=ch_bh,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv2d
        self.box_conv2d = nn.Conv2d(
            in_channels=ch_bh,
            out_channels=ch_bo,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        # Class Branch
        # DWConv0
        self.class_dwconv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_i,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=ch_i
        )
        
        # Conv0
        self.class_conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_ch,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # DWConv1
        self.class_dwconv1 = Conv(
            in_channels=ch_ch,
            out_channels=ch_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=ch_ch
        )
                
        # Conv1
        self.class_conv1 = Conv(
            in_channels=ch_ch,
            out_channels=ch_ch,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Conv2d
        self.class_conv2d = nn.Conv2d(
            in_channels=ch_ch,
            out_channels=ch_co,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
    # forward
    def forward(self, x):
        
        # box branch
        y = self.box_conv0(x)
        y = self.box_conv1(y)
        y = self.box_conv2d(y)
        o0 = y
        
        # class branch
        y = self.class_dwconv0(x)
        y = self.class_conv0(y)
        y = self.class_dwconv1(y)
        y = self.class_conv1(y)
        y = self.class_conv2d(y)
        o1 = y
        
        # return
        return [o0, o1]