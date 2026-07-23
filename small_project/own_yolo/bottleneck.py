import torch
import torch.nn as nn

# ConvBNSiLU 가져옴
from convbnsilu import ConvBNSiLU

# Bottleneck 구조를 구현한 모듈
class Bottleneck(nn.Module):
    
    def __init__(
        self, 
        channels,               # 입출력 채널 수
        shortcut = True,        # residual 연결 여부
        activation = "silu"     # 활성화 함수
    ):
        # PyTorch를 사용하기 위해 nn.Module을 초기화
        super(Bottleneck, self).__init__()
        
        # 1x1 Conv
        self.conv1 = ConvBNSiLU(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            activation = activation
        )
        
        # 3x3 Conv
        self.conv2 = ConvBNSiLU(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            activation = activation
        )
        
        # residual 연결 여부
        self.use_shortcut = shortcut
        
    def forward(self, x):
        # x를 residual에 저장
        residual = x
        
        # 1x1 Conv → 3x3 Conv
        x = self.conv1(x)
        x = self.conv2(x)
        
        # residual 연결이 활성화되면
        # Conv 결과 x와 residual을 합해줌
        if self.use_shortcut:
            x = x + residual

        return x