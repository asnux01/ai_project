import torch.nn as nn

# 별도의 convbnsilu.py 파일에 작성된
# ConvBNSiLU 클래스를 가져온다.
from convbnsilu import ConvBNSiLU


# 그림의 BottleNeck 1 구조를 구현한 클래스
class Bottleneck1(nn.Module):

    def __init__(self, channels):

        # 부모 클래스인 nn.Module을 초기화한다.
        super(Bottleneck1, self).__init__()

        # 첫 번째 ConvBNSiLU 모듈
        #
        # 그림의 설정:
        # k1, s1, p0, c
        #
        # kernel_size = 1
        # stride      = 1
        # padding     = 0
        #
        # 입력 채널과 출력 채널을 모두 channels로 설정하므로
        # 채널 수는 바뀌지 않는다.
        #
        # 입력:
        # [배치, channels, 높이, 너비]
        #
        # 출력:
        # [배치, channels, 높이, 너비]
        self.conv1 = ConvBNSiLU(
            channels,    # 입력 채널 수 c
            channels,    # 출력 채널 수 c
            1,           # kernel_size
            1,           # stride
            0            # padding
        )

        # 두 번째 ConvBNSiLU 모듈
        #
        # 그림의 설정:
        # k3, s1, p1, c
        #
        # kernel_size = 3
        # stride      = 1
        # padding     = 1
        #
        # 3×3 컨볼루션을 사용하지만 padding=1이므로
        # 특징 맵의 높이와 너비가 유지된다.
        #
        # 입력 채널과 출력 채널도 모두 channels이므로
        # 채널 수도 그대로 유지된다.
        self.conv2 = ConvBNSiLU(
            channels,    # 입력 채널 수 c
            channels,    # 출력 채널 수 c
            3,           # kernel_size
            1,           # stride
            1            # padding
        )

    def forward(self, x):

        # 원래 입력 x를 저장한다.
        #
        # 이 값은 두 ConvBNSiLU를 통과한 결과와
        # 마지막에 더할 때 사용한다.
        residual = x

        # 첫 번째 ConvBNSiLU를 통과시킨다.
        #
        # k=1, s=1, p=0
        x = self.conv1(x)

        # 두 번째 ConvBNSiLU를 통과시킨다.
        #
        # k=3, s=1, p=1
        x = self.conv2(x)

        # 두 ConvBNSiLU를 통과한 결과에
        # 처음 저장한 원래 입력을 더한다.
        #
        # 그림에서 동그라미 안의 +에 해당한다.
        x = x + residual

        # 최종 결과를 반환한다.
        return x