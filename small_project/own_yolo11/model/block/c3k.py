#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch
import torch.nn as nn

from ..layer import Conv
from .bottleneck import Bottleneck

class C3K (nn.Module):
    
    # 초기화
    def __init__(
        self,
        in_channels,
        out_channels,
        n,
        shortcut=True
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(C3K, self).__init__()
        
        # 파라미터
        ch_i = in_channels          # 입력 채널 수
        ch_o = out_channels         # 출력 채널 수   
        bn_cnt = n                  # bottleneck 카운터
        
        # 파라미터 유효성 검사
        # 0채널 Conv가 생성되는 것을 방지
        if (ch_i <= 0 or ch_o <= 0):
            raise ValueError(
                "in_channels와 out_channels는 "
                "1 이상이어야 합니다."
            )
        
        # Bottleneck이 한 번도 실행되지 않는 것을 방지
        if bn_cnt <= 0:
            raise ValueError(
                "n은 1 이상의 정수여야 합니다."
            )
        
        ch_h = out_channels // 2    # hidden 채널 수
        ch_c = ch_h * 2             # concat 채널 수 
        
        # 첫 번째 분기 Conv: bottleneck 통과 안 함
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # 두 번째 Conv: bottleneck 통과
        self.conv1 = Conv(
            in_channels=ch_i,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Bottleneck
        # Bottleneck list
        self.bottlenecks = nn.ModuleList()
                        
        # Bottleneck append
        for _ in range(bn_cnt):
            bottleneck = Bottleneck(
                in_channels=ch_h,
                out_channels=ch_h,
                shortcut=shortcut
            )
                
            self.bottlenecks.append(bottleneck)
    
        # concat 이후 Conv
        self.conv2 = Conv(
            in_channels=ch_c,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    # forward
    def forward(self, x):
        
        # 첫 번째 분기 Conv
        x0 = self.conv0(x)
        
        # 두 번째 분기 Conv
        x1 = self.conv1(x)
        
        # Bottleneck
        for bottleneck in self.bottlenecks:
            x1 = bottleneck(x1)
        
        # Concat
        x = torch.cat(
            [x0, x1],
            dim=1
        )
        
        # concat 이후 Conv
        x = self.conv2(x)
        
        # 반환
        return x