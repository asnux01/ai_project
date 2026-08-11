#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch
import torch.nn as nn

from ..layer import Conv
from .psa import PSABlock


class C2PSA(nn.Module):
    
    # 초기화
    def __init__(
        self,
        in_channels,
        out_channels,
        n,
        shortcut=True,
        e=0.5
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(C2PSA, self).__init__()
        
        # 파라미터
        ch_i = in_channels          # 입력 채널 수
        ch_o = out_channels         # 출력 채널 수
        ch_h = int(ch_o * e)        # hidden 채널 수
        psa_cnt = n                 # PSABlock counter
        
        # 파라미터 유효성 검사
        # 0채널 Conv가 만들어지는 것을 방지
        if ( ch_i <= 0 or ch_o <= 0):
            raise ValueError(
                "in_channels와 out_channels는 "
                "1 이상이어야 합니다."
            )

        # PSA가 한 번도 실행되지 않는 것을 방지
        if psa_cnt <= 0:
            raise ValueError(
                "n은 1 이상의 정수여야 합니다."
            )

        # Hidden 채널 비율의 유효 범위를 검사
        if not 0 < e <= 1:
            raise ValueError(
                "e는 0보다 크고 1 이하여야 합니다."
            )
        
        ch_s = 2 * ch_h             # 스플릿 전 채널 수
        ch_c = 2 * ch_h             # concat 후 채널 수
        
        # 포워드 접근 가능 파라미터
        # hidden 채널 수
        self.hidden_channels = ch_h
        
        # Attention head 카운터
        head_cnt = max(ch_h // 64, 1)

        # Attention의 head별 reshape가 성립하려면
        # hidden 채널이 head 수로 나누어떨어져야 함
        if ch_h % head_cnt != 0:
            raise ValueError(
                f"hidden 채널({ch_h})은 "
                f"attention head 수({head_cnt})로 "
                "나누어떨어져야 합니다."
            )
        
        # 스플릿 전 Conv
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_s,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # PSABlock 
        # PSABlock list
        self.blocks = nn.ModuleList()
        
        # PSABlock append
        for _ in range(psa_cnt):
            block = PSABlock(
                in_channels=ch_h,
                out_channels=ch_h,
                attn_ratio=0.5,
                num_heads=head_cnt,
                shortcut=shortcut
            )
            
            self.blocks.append(block)
        
        # Concat 후 Conv
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
        
        # x0와 x1 스플릿
        # x0: PSABlock을 거치지 않는 경로
        # x1: PSABlock을 거치는 경로
        x0, x1 = x.split(
            [
                self.hidden_channels,
                self.hidden_channels
            ],
            dim=1
        )
        
        # PSABlock
        for block in self.blocks:
            x1 = block(x1)
        
        # Concat
        x = torch.cat(
            [x0, x1],
            dim=1
        )
        
        # Concat 후 Conv
        x = self.conv1(x)
        
        # 반환
        return x