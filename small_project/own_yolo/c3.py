import torch
import torch.nn as nn

# 필요한 모듈 가져옴
from convbnsilu import ConvBNSiLU
from bottleneck import Bottleneck


# C3 모듈
# Convbnsilu → Bottleneck → Concat → Convbnsilu 구조를 가지는 모듈
class C3(nn.Module):

    def __init__(
        self, 
        in_channels,            # 입력 채널 수
        out_channels,           # 출력 채널 수
        bottleneck_count = 1,   # Bottleneck 모듈의 개수
        shortcut = True,        # residual 연결 여부
        activation = "silu"     # 활성화 함수
    ):

        # PyTorch를 사용하기 위해 nn.Module을 초기화
        super(C3, self).__init__()

        # 두 분기에 사용할 채널 수
        hidden_channels = in_channels // 2
        
        # 첫 번째 분기 Conv
        # Bottleneck을 통과하지 않는 Conv
        self.branch1_conv = ConvBNSiLU(
            in_channels = in_channels,
            out_channels = hidden_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            activation = activation
        )

        # 두 번째 분기 Conv
        # Bottleneck을 통과하는 Conv
        self.branch2_conv = ConvBNSiLU(
            in_channels = in_channels,
            out_channels = hidden_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            activation = activation
        )

        # Bottleneck 저장 리스트
        self.bottlenecks = nn.ModuleList()
        
        # Bottleneck을 지정된 개수만큼 생성
        for _ in range(bottleneck_count):
            bottleneck = Bottleneck(
                channels = hidden_channels,
                shortcut = shortcut,
                activation = activation
            )

            self.bottlenecks.append(bottleneck)
        
        # Concat 이후 마지막 Conv
        self.output_conv = ConvBNSiLU(
            in_channels = hidden_channels * 2,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            activation = activation
        )

        
    def forward(self, x):
        # 첫 번째 분기
        branch1 = self.branch1_conv(x)

        # 두 번째 분기
        branch2 = self.branch2_conv(x)

        # Bottleneck들을 순서대로 통과
        for bottleneck in self.bottlenecks:
            branch2 = bottleneck(branch2)

        # 두 분기를 채널 방향으로 연결
        x = torch.cat(
            (branch1, branch2),
            dim = 1
        )

        # Concat 결과를 ConvBNSiLU에 통과
        x = self.output_conv(x)

        return x