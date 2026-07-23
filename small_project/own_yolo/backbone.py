import torch
import torch.nn as nn

# 필요한 모듈 가져옴
from convbnsilu import ConvBNSiLU
from c3 import C3
from sppf import SPPF

# C3 & convbnsilu & SPPF를 이용한 Backbone 구조를 구현한 모듈
class Backbone(nn.Module):
    
    def __init__(
        self,
        in_channels = 3,        # 입력 특징 맵의 채널 수
        activation = "silu"     # 활성화 함수 적용 여부
    ):
        
        # PyTorch를 사용하기 위해 nn.Module을 초기화
        super(Backbone, self).__init__()
        
        # 스테이지별 내부 채널 수
        stage1_channels = 64
        stage2_channels = stage1_channels * 2
        stage3_channels = stage2_channels * 2
        stage4_channels = stage3_channels * 2
        stage5_channels = stage4_channels * 2
        
        # Neck으로 전달할 채널 리스트
        self.out_channels = [
            stage3_channels,
            stage4_channels,
            stage5_channels
        ]
        
        # 스테이지 1: P1
        # P1 ConvBNSiLU
        # 640 x 640 x 3 -> 320 x 320 x 64
        self.stage1_P1_conv = ConvBNSiLU(
            in_channels = in_channels,
            out_channels = stage1_channels,
            kernel_size = 6,
            stride = 2,
            padding = 2,
            activation = activation
        )
        
        # 스테이지 2: P2 -> C3
        # P2 ConvBNSiLU
        # 320 x 320 x 64 -> 160 x 160 x 128
        self.stage2_P2_conv = ConvBNSiLU(
            in_channels = stage1_channels,
            out_channels = stage2_channels,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            activation = activation
        )
        
        # C3
        self.stage2_C3 = C3(
            in_channels = stage2_channels,
            out_channels = stage2_channels,
            bottleneck_count = 3,
            shortcut = True,
            activation = activation
        )
        
        # 스테이지 3: P3 -> C3
        # P3 ConvBNSiLU
        # 160 x 160 x 128 -> 80 x 80 x 256
        self.stage3_P3_conv = ConvBNSiLU(
            in_channels = stage2_channels,
            out_channels = stage3_channels,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            activation = activation
        )
        
        # C3
        self.stage3_C3 = C3(
            in_channels = stage3_channels,
            out_channels = stage3_channels,
            bottleneck_count = 6,
            shortcut = True,
            activation = activation 
        )
        
        # 스테이지 4: P4 -> C3
        # P4 ConvBNSiLU
        # 80 x 80 x 256 -> 40 x 40 x 512
        self.stage4_P4_conv = ConvBNSiLU(
            in_channels = stage3_channels,
            out_channels = stage4_channels,
            kernel_size = 3,
            stride = 2,
            padding = 1,    
            activation = activation
        )
        
        # 네 번째 C3
        self.stage4_C3 = C3(
            in_channels = stage4_channels,
            out_channels = stage4_channels,
            bottleneck_count = 9,
            shortcut = True,
            activation = activation
        )
        
        # 스테이지 5: P5 -> C3
        # P5 ConvBNSiLU
        # 40 x 40 x 512 -> 20 x 20 x 1024
        self.stage5_P5_conv = ConvBNSiLU(
            in_channels = stage4_channels,
            out_channels = stage5_channels,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            activation = activation
        )
        
        # C3
        self.stage5_C3 = C3(
            in_channels = stage5_channels,
            out_channels = stage5_channels,
            bottleneck_count = 3,
            shortcut = True,
            activation = activation
        )
        
        # 스테이지 6: SPPF
        # SPPF
        self.stage6_SPPF = SPPF(
            in_channels = stage5_channels,
            out_channels = stage5_channels,
            activation = activation
        )
        
    def forward(self, x):
        # 스테이지 1: P1
        x = self.stage1_P1_conv(x)
        
        # 스테이지 2: P2 -> C3
        x = self.stage2_P2_conv(x)
        x = self.stage2_C3(x)
        
        # 스테이지 3: P3 -> C3
        # neck에서 사용할 특징 맵 반환
        x = self.stage3_P3_conv(x)
        x = self.stage3_C3(x)
        stage3_fm = x
        
        # 스테이지 4: P4 -> C3
        x = self.stage4_P4_conv(x)
        x = self.stage4_C3(x)
        stage4_fm = x
        
        # 스테이지 5: P5 -> C3
        x = self.stage5_P5_conv(x)
        x = self.stage5_C3(x)
        
        # 스테이지 6: SPPF
        x = self.stage6_SPPF(x)
        stage6_fm = x

        return [stage3_fm, stage4_fm, stage6_fm]