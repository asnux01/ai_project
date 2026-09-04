#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch.nn as nn

from ..layer import Conv
from ..block import C3K2, C2PSA, SPPF

class Backbone(nn.Module):
    
    # 초기화
    def __init__(
        self,
        depth_factor,
        width_factor,
        max_channels,
        shortcut=True
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(Backbone, self).__init__()
        
        # 파라미터
        # 파라미터 유효성 검사
        # 잘못된 depth 설정으로 반복 블록이 사라지는 것을 방지
        if depth_factor <= 0:
            raise ValueError(
                "depth_factor는 0보다 커야 합니다."
            )

        # 잘못된 width 설정으로 0채널 Conv가 생성되는 것을 방지
        if width_factor <= 0:
            raise ValueError(
                "width_factor는 0보다 커야 합니다."
            )

        if max_channels <= 0:
            raise ValueError(
                "max_channels는 0보다 커야 합니다."
            )
        
        # 채널 파라미터
        # 채널 수가 0이 되는 것을 방지
        ch_min64 = max(int(min(64, max_channels) * width_factor), 1)
        ch_min128 = max(int(min(128, max_channels) * width_factor), 1)
        ch_min256 = max(int(min(256, max_channels) * width_factor), 1)
        ch_min512 = max(int(min(512, max_channels) * width_factor), 1)
        ch_min1024 = max(int(min(1024, max_channels) * width_factor), 1)
        
        # 카운터
        # 정수로 변환하고 반복 횟수를 최소 1로 설정
        n = max(int(2 * depth_factor), 1)
        
        # 포워드 접근 가능 파라미터
        # 출력 채널 수
        self.out_channels = [
            ch_min256,
            ch_min512,
            ch_min1024
        ]
        
        # stage 1
        # Conv
        self.conv0 = Conv(
            in_channels=3,
            out_channels=ch_min64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # stage 2
        # Conv
        self.conv1 = Conv(
            in_channels=ch_min64,
            out_channels=ch_min128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # C3K2
        self.c3k2_0 = C3K2(
            in_channels=ch_min128,
            out_channels=ch_min256,
            c3k=False,
            n=n,
            e=0.25
        )
        
        # stage 3
        # Conv
        self.conv2 = Conv(
            in_channels=ch_min256,
            out_channels=ch_min256,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # C3K2
        self.c3k2_1 = C3K2(
            in_channels=ch_min256,
            out_channels=ch_min512,
            c3k=False,
            n=n,
            e=0.25
        )
        
        # stage 4
        # Conv
        self.conv3 = Conv(
            in_channels=ch_min512,
            out_channels=ch_min512,
            kernel_size=3,
            stride=2,
            padding=1
        )
                
        # C3K2
        self.c3k2_2 = C3K2(
            in_channels=ch_min512,
            out_channels=ch_min512,
            c3k=True,
            n=n
        )
        
        # stage 5
        # Conv
        self.conv4 = Conv(
            in_channels=ch_min512,
            out_channels=ch_min1024,
            kernel_size=3,
            stride=2,
            padding=1
        )
                        
        # C3K2
        self.c3k2_3 = C3K2(
            in_channels=ch_min1024,
            out_channels=ch_min1024,
            c3k=True,
            n=n
        )
        
        # stage 6
        # SPPF
        self.sppf = SPPF(
            in_channels=ch_min1024,
            out_channels=ch_min1024
        )
        
        # C2PSA
        self.c2psa = C2PSA(
            in_channels=ch_min1024,
            out_channels=ch_min1024,
            n=n,
            shortcut=shortcut,
        )
        
    # 포워드
    def forward(self, x):
        
        # stage 1
        x = self.conv0(x)
        
        # stage 2
        x = self.conv1(x)
        x = self.c3k2_0(x)
        
        # stage 3
        x = self.conv2(x)
        x = self.c3k2_1(x)
        y0 = x
        
        # stage 4
        x = self.conv3(x)
        x = self.c3k2_2(x)
        y1 = x
        
        # stage 5
        x = self.conv4(x)
        x = self.c3k2_3(x)
        
        # stage 6
        x = self.sppf(x)
        x = self.c2psa(x)
        y2 = x
        
        # 반환
        return [y0, y1, y2]