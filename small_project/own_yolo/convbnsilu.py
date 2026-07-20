import torch
import torch.nn as nn


# Convolution → Batch Normalization → SiLU를
# 하나의 묶음으로 만든 모듈
class ConvBNSiLU(nn.Module):

    def __init__(
        self,
        in_channels,     # 입력 특징 맵의 채널 수
        out_channels,    # 출력 특징 맵의 채널 수
        kernel_size,     # 컨볼루션 커널 크기
        stride,          # 커널이 이동하는 간격
        padding          # 입력 가장자리에 추가할 패딩 크기
    ):
        # 부모 클래스인 nn.Module을 초기화한다.
        # 이 코드가 있어야 PyTorch가 아래의 Conv, BN, SiLU를
        # 모델을 구성하는 정식 레이어로 인식할 수 있다.
        super(ConvBNSiLU, self).__init__()

        # 1. 컨볼루션 연산
        #
        # 입력 특징 맵에서 선, 모서리, 무늬 등의 특징을 추출한다.
        #
        # in_channels:
        #   입력 특징 맵의 채널 수
        #
        # out_channels:
        #   컨볼루션이 생성할 출력 특징 맵의 개수
        #
        # kernel_size:
        #   컨볼루션 필터의 크기
        #   예: 3이면 3×3 필터
        #
        # stride:
        #   컨볼루션 필터가 이동하는 간격
        #   예: 2이면 두 칸씩 이동하므로
        #   출력의 가로와 세로가 보통 절반으로 줄어든다.
        #
        # padding:
        #   컨볼루션 전에 입력 가장자리에 추가하는 값의 크기
        #
        # bias=False:
        #   바로 뒤에서 BatchNorm이 값의 이동을 담당하므로
        #   Conv의 bias는 보통 사용하지 않는다.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        )

        # 2. 배치 정규화
        #
        # 컨볼루션이 만든 특징값의 분포를 정리해서
        # 모델이 안정적으로 학습하도록 돕는다.
        #
        # Conv의 출력 채널이 out_channels개이므로
        # BatchNorm에도 out_channels를 넣는다.
        self.bn = nn.BatchNorm2d(out_channels)

        # 3. SiLU 활성화 함수
        #
        # 특징값을 비선형적으로 변환한다.
        # 큰 양수는 대부분 유지하고,
        # 음수와 작은 값은 부드럽게 줄여서 전달한다.
        self.silu = nn.SiLU()

    def forward(self, x):

        # 입력 x를 컨볼루션에 통과시킨다.
        # 특징을 추출하고, 필요하면 크기와 채널 수도 변경한다.
        x = self.conv(x)

        # 컨볼루션이 만든 특징값의 분포를 정리한다.
        # 텐서의 높이, 너비, 채널 수는 바뀌지 않는다.
        x = self.bn(x)

        # SiLU 활성화 함수를 적용한다.
        # 텐서 크기는 바뀌지 않고 내부 값만 변한다.
        x = self.silu(x)

        # Conv → BN → SiLU를 모두 통과한 결과를 반환한다.
        return x