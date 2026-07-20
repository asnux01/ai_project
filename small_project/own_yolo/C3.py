import torch.nn as nn

# 각각 별도의 파일에 작성된 모듈을 가져온다.
from convbnsilu import ConvBNSiLU
from bottleneck import Bottleneck1
from concat import Concat


# 사진의 C3 구조를 구현한 모듈
class C3(nn.Module):

    def __init__(self, in_channels, branch_channels, out_channels):

        # 부모 클래스인 nn.Module을 초기화한다.
        super(C3, self).__init__()

        # 왼쪽 경로의 1×1 ConvBNSiLU
        #
        # 사진에서는:
        # 입력 채널 128 → 출력 채널 64
        #
        # k1: kernel_size=1
        # s1: stride=1
        # p0: padding=0
        self.left_conv = ConvBNSiLU(
            in_channels,
            branch_channels,
            1,
            1,
            0
        )

        # 오른쪽 경로의 1×1 ConvBNSiLU
        #
        # 왼쪽 경로와 같은 입력을 받지만,
        # 서로 다른 컨볼루션 가중치를 사용한다.
        #
        # 사진에서는:
        # 입력 채널 128 → 출력 채널 64
        self.right_conv = ConvBNSiLU(
            in_channels,
            branch_channels,
            1,
            1,
            0
        )

        # 사진의 BottleNeck 1 × 3에 해당한다.
        #
        # 각 Bottleneck1은 채널 수를 변경하지 않는다.
        # 따라서 64채널이 계속 유지된다.
        self.bottleneck1 = Bottleneck1(branch_channels)
        self.bottleneck2 = Bottleneck1(branch_channels)
        self.bottleneck3 = Bottleneck1(branch_channels)

        # 왼쪽 경로와 오른쪽 경로의 결과를
        # 채널 방향으로 연결한다.
        self.concat = Concat()

        # Concat 이후의 1×1 ConvBNSiLU
        #
        # 왼쪽 경로 채널 수 + 오른쪽 경로 채널 수가
        # 입력 채널 수가 된다.
        #
        # 사진에서는:
        # 64 + 64 = 128채널
        #
        # 마지막 출력도 128채널이다.
        self.output_conv = ConvBNSiLU(
            branch_channels * 2,
            out_channels,
            1,
            1,
            0
        )

    def forward(self, x):

        # 입력 x를 왼쪽 경로로 전달한다.
        #
        # 사진 기준:
        # [배치, 128, 160, 160]
        #            ↓
        # [배치, 64, 160, 160]
        left = self.left_conv(x)

        # 같은 입력 x를 오른쪽 경로로 전달한다.
        #
        # 사진 기준:
        # [배치, 128, 160, 160]
        #            ↓
        # [배치, 64, 160, 160]
        right = self.right_conv(x)

        # 오른쪽 경로를 Bottleneck1에 세 번 통과시킨다.
        #
        # Bottleneck1은 채널 수와 가로·세로 크기를
        # 변경하지 않는다.
        right = self.bottleneck1(right)
        right = self.bottleneck2(right)
        right = self.bottleneck3(right)

        # 왼쪽 결과와 오른쪽 결과를
        # 채널 방향으로 연결한다.
        #
        # 64채널 + 64채널 = 128채널
        x = self.concat(left, right)

        # Concat 결과를 마지막 1×1 ConvBNSiLU에 통과시킨다.
        #
        # 사진 기준:
        # 128채널 → 128채널
        x = self.output_conv(x)

        # C3 모듈의 최종 결과를 반환한다.
        return x