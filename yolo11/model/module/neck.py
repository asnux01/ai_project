#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch
import torch.nn as nn

from ..layer import Conv
from ..block import C3K2

class Neck(nn.Module):
    
    # 초기화
    def __init__(
        self,
        channels,
        depth_factor,
        shortcut=True
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(Neck, self).__init__()
        
        # 파라미터 유효성 검사
        # Neck은 P3, P4, P5 세 특징을 입력받으므로
        # 채널 값도 정확히 세 개여야 한다.
        if (not isinstance(channels, (tuple, list)) 
            or len(channels) != 3):
            raise ValueError(
                "channels는 P3, P4, P5 채널 값 "
                "3개여야 합니다."
            )

        # 모든 특징 맵의 채널 수는 양수여야 한다.
        if any(int(channel) <= 0 for channel in channels):
            raise ValueError(
                "모든 channels 값은 "
                "1 이상이어야 합니다."
            )

        if depth_factor <= 0:
            raise ValueError(
                "depth_factor는 0보다 커야 합니다."
            )
        
        # 파라미터
        ch_min256 = int(channels[0])        # min256의 채널 수
        ch_min512 = int(channels[1])        # min512의 채널 수
        ch_min1024 = int(channels[2])       # min1024의 채널 수
        ch_c_15 = ch_min512 + ch_min1024    # P5와 P4를 concat한 채널 수
        ch_c_25 = ch_min256 + ch_min512     # P4와 P3를 concat한 채널 수
        ch_c_55 = ch_min512 + ch_min512     # P5와 P5를 concat한 채널 수
        
        # 카운터
        n = max(int(2 * depth_factor), 1)
        
        # 포워드 접근 가능 파라미터
        # 출력 채널 수
        self.out_channels = [
            ch_min256,
            ch_min512,
            ch_min1024
        ]

        # 업샘플링
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )
        
        # Topdown1
        # C3K2
        self.c3k2_0 = C3K2(
            in_channels=ch_c_15,
            out_channels=ch_min512,
            c3k=False,
            n=n,
            shortcut=shortcut
        )
        
        # Topdown2
        # C3K2
        self.c3k2_1 = C3K2(
            in_channels=ch_c_55,
            out_channels=ch_min256,
            c3k=False,
            n=n,
            shortcut=shortcut
        )
        
        # Bottomup1
        # Conv
        self.conv0 = Conv(
            in_channels=ch_min256,
            out_channels=ch_min256,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # C3K2
        self.c3k2_2 = C3K2(
            in_channels=ch_c_25,
            out_channels=ch_min512,
            c3k=False,
            n=n,
            shortcut=shortcut
        )
        
        # Bottomup1
        # Conv
        self.conv1 = Conv(
            in_channels=ch_min512,
            out_channels=ch_min512,
            kernel_size=3,
            stride=2,
            padding=1
        )
                
        # C3K2
        self.c3k2_3 = C3K2(
            in_channels=ch_c_15,
            out_channels=ch_min1024,
            c3k=True,
            n=n,
            shortcut=shortcut
        )
    
    # 포워드
    def forward(self, x):
        
        # Bottomup 줄기
        trans0 = x[2]
        
        # Topdown1
        # 첫 번째 업샘플링
        y = self.upsample(x[2])
        
        # Concat
        y = torch.cat(
            [y, x[1]],
            dim=1
        )
        
        # C3K2
        y = self.c3k2_0(y)
        trans1 = y
        
        # Topdown2
        # 두 번째 업샘플링
        y = self.upsample(y)
                
        # Concat
        y = torch.cat(
            [y, x[0]],
            dim=1
        )
                
        # C3K2
        y = self.c3k2_1(y)
        
        # 헤드 출력 0
        head0 = y
        
        # Bottomup1
        # Conv
        y = self.conv0(y)
        
        # Concat
        y = torch.cat(
            [y, trans1],
            dim=1
        )
        
        # C3K2
        y = self.c3k2_2(y)
        
        # 헤드 출력 1
        head1 = y
        
        # Bottomup1
        # Conv
        y = self.conv1(y)
                
        # Concat
        y = torch.cat(
            [y, trans0],
            dim=1
        )
                
        # C3K2
        y = self.c3k2_3(y)
        
        # 헤드 출력 2
        head2 = y
        
        # 반환
        return [head0, head1, head2]