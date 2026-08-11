#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch.nn as nn

from ..layer import Conv

class BoxBranch(nn.Module):
    
    # 초기화
    def __init__(
        self,
        in_channels,
        hidden_channels,
        reg_max=16
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(BoxBranch, self).__init__()
        
        # 채널 파라미터
        ch_i = in_channels
        ch_h = hidden_channels
        ch_o = 4 * reg_max
        
        # 첫 번째Conv
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_h,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # 두 번째 Conv
        self.conv1 = Conv(
            in_channels=ch_h,
            out_channels=ch_h,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv2d
        # Left, top, right, bottom의 logit 출력
        self.conv2d = nn.Conv2d(
            in_channels=ch_h,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
    # 포워드
    def forward(self, x):
        
        # box branch
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2d(x)
        
        # 반환
        return x