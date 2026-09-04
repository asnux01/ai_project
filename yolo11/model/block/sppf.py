#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch
import torch.nn as nn

from ..layer import Conv

class SPPF(nn.Module):
    
    # 초기화
    def __init__(
        self,
        in_channels,
        out_channels,
        k=5
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(SPPF, self).__init__()
        
        # parameter
        ch_i = in_channels          # 입력 채널 수 
        ch_o = out_channels         # 출력 채널 수
        ch_h = out_channels // 2    # hidden 채널 수
        ch_c = ch_h * 4             # concat 후 채널 수
        mp_k = k                    # maxpool 커널 크기
        
        # Conv0
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0,
            activation="identity"
        )
        
        # Maxpool2d
        self.maxpool2d = nn.MaxPool2d(
            kernel_size=mp_k,
            stride=1,
            padding=2
        )
        
        # Conv1
        self.conv1 = Conv(
            in_channels=ch_c,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    # 포워드
    def forward(self, x):
        
        # Conv0
        x = self.conv0(x)
        
        # Maxpool
        x1 = self.maxpool2d(x)
        x2 = self.maxpool2d(x1)
        x3 = self.maxpool2d(x2)
        
        # Concat
        x = torch.cat(
            [x, x1, x2, x3],
            dim=1
        )
        
        # Conv1
        x = self.conv1(x)
        
        # 반환
        return x
    
    
    