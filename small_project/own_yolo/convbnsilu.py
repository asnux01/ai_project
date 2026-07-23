import torch
import torch.nn as nn


# Convolution → Batch Normalization → 활성화 함수
# 하나의 묶음으로 만든 모듈
class ConvBNSiLU(nn.Module):

    def __init__(
        self,   
        in_channels,        # 입력 채널 수
        out_channels,       # 출력 채널 수
        kernel_size,        # 컨볼루션 커널 사이즈 nxn
        stride,             # 컨볼루션 스트라이드 간격
        padding,            # 컨볼루션 패딩 크기
        activation = "silu" # 활성화 함수
    ):
        # PyTorch를 사용하기 위해 nn.Module을 초기화
        super(ConvBNSiLU, self).__init__()

        # Conv
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = False
        )

        # BatchNorm
        self.bn = nn.BatchNorm2d(
            num_features = out_channels
        )

        # Activation Ft
        # 원하는 활성화 함수 선택
        activation = activation.lower()
        
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        # 입력 x를 Conv → BatchNorm → Activation 순으로 통과
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x