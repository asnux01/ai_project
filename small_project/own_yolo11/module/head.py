# import library
import torch
import torch.nn as nn

from ..layer import Conv

class Head(nn.Module):
    
    # initialized
    def __init__(
        self,
        channels,
        nc
    ):
        
        # nn.Module reset to use PyTorch
        super(Head, self).__init__()
        
        # parameter
        self.nc = nc
        self.reg_max = 16
        self.nl = len(channels)
        self.no = self.nc + 4 * self.reg_max
        
        reg_max = 16
        ch_min256 = channels[0]
        ch_min512 = channels[1]
        ch_min1024 = channels[2]
        ch_box_out = 4 * reg_max
        ch_box_h = max(16, ch_min256 // 4, ch_box_out)
        ch_cls_h = max(ch_min256, min(nc, 100))