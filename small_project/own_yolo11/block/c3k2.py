# import library
import torch
import torch.nn as nn

from ..layer import Conv
from .bottleneck import Bottleneck
from .c3k import C3K

class C3K2(nn.Module):
    
    # initialized
    def __init__(
        self,
        in_channels,
        out_channels,
        c3k,
        n,
        shortcut,
        e,
        activation="silu"
    ):
        
        # nn.Module reset to use PyTorch
        super(C3K2, self).__init__()
        
        # parameter
        ch0 = channels,
        ch1 = 