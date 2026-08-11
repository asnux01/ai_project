#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch
import torch.nn as nn

from ..layer import Conv
from .bottleneck import Bottleneck
from .c3k import C3K

class C3K2(nn.Module):
    
    # 초기화
    def __init__(
        self,
        in_channels,
        out_channels,
        c3k,
        n,
        shortcut=True,
        e=0.5
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(C3K2, self).__init__()
        
        # parameter
        ch_i = in_channels              # 입력 채널 수
        ch_o = out_channels             # 출력 채널 수
        bn_cnt = n                      # bottleneck 카운터
        
        # 파라미터 유효성 검사
        # 0채널 Conv가 만들어지는 것을 방지한다.
        if (ch_i <= 0 or ch_o <= 0):
            raise ValueError(
                "in_channels와 out_channels는 "
                "1 이상이어야 합니다."
            )

        # 반복 블록이 완전히 사라지는 것을 방지한다.
        if bn_cnt <= 0:
            raise ValueError(
                "n은 1 이상의 정수여야 합니다."
            )

        # Hidden 채널 비율의 유효 범위를 검사한다.
        if not 0 < e <= 1:
            raise ValueError(
                "e는 0보다 크고 1 이하여야 합니다."
            )
        
        ch_h = max(int(out_channels * e), 1)    # hidden 채널 수
        ch_s = 2 * ch_h                         # 스플릿 전 채널 수 
        ch_c = (bn_cnt + 2) * ch_h              # concat 후 채널 수
        
        # 스플릿 전 Conv 
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_s,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Bottleneck or C3K
        # Ck3 and Bottleneck list
        self.blocks = nn.ModuleList()
                                
        # Both append
        for _ in range(bn_cnt):
            # c3k가 True면 C3K를 사용
            if c3k:
                block = C3K(
                    in_channels=ch_h,
                    out_channels=ch_h,
                    n=2,
                    shortcut=shortcut,
                )
                # c3k가 False면 Bottleneck을 사용
            else:
                block = Bottleneck(
                    in_channels=ch_h,
                    out_channels=ch_h,
                    shortcut=shortcut
                )
                        
            self.blocks.append(block)

        # concat 이후 Conv
        self.conv1 = Conv(
            in_channels=ch_c,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    # 포워드
    def forward(self, x):
        
        # 스플릿 전 Conv
        x = self.conv0(x)
        
        # x0와 x1로 스플릿
        # x0: C3K 혹은 Bottleneck 통과 안 함
        # x1: C3K 혹은 Bottleneck 통과
        x0, x1 = x.chunk(
            chunks=2,
            dim=1
        )
        
        # concat용 리스트
        y = [x0, x1]
        
        # C3K 혹은 Bottleneck 통과
        for block in self.blocks:
            x1 = block(x1)
            y.append(x1)
            
        # Concat
        x = torch.cat(
            y,
            dim=1
        )
        
        # Concat 후 Conv
        x = self.conv1(x)
        
        # 반환
        return x