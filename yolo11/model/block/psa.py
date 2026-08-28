#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch.nn as nn

from ..layer import Conv
from .attention import Attention

class PSABlock(nn.Module):
    
    # 초기화
    def __init__(
        self,
        in_channels,
        out_channels,
        attn_ratio=0.5,
        num_heads=4,
        shortcut=True
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(PSABlock, self).__init__()
        
        # 파라미터
        ch_i = in_channels          # 입력 채널 수
        ch_h = 2 * out_channels     # hidden 채널 수
        ch_o = out_channels         # 출력 채널 수
        
        # Attention
        self.attention = Attention(
            in_channels=ch_i,
            out_channels=ch_o,
            num_heads=num_heads,
            attn_ratio=attn_ratio
        )
        
        # FFN Conv0
        self.ffn_conv0 = Conv(
            in_channels=ch_o,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # FFN Conv1
        self.ffn_conv1 = Conv(
            in_channels=ch_h,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0,
            activation="identity"
        )
        
        # Attention shortcut
        self.attention_shortcut = (
            shortcut
            and ch_i == ch_o
        )
        
        # FFN shortcut
        self.ffn_shortcut = shortcut
    
    # 포워드
    def forward(self, x):
        
        # Attention residual
        res = x
        
        # Attention
        x = self.attention(x)
        
        # Attention residual
        if self.attention_shortcut:
            x = res + x
        
        # FFN residual
        res = x
        
        # FFN
        x = self.ffn_conv0(x)
        x = self.ffn_conv1(x)
        
        # FFN residual
        if self.ffn_shortcut:
            x = res + x
        
        # 반환
        return x