import torch
import torch.nn as nn

# 필요한 모듈 가져옴
from convbnsilu import ConvBNSiLU

class SPPF(nn.Module):
    
    def __init__(
        self,
        in_channels,            # 입력 채널 수
        out_channels,           # 출력 채널 수
        activation = "silu",    # 활성화 함수
    ):
        
        # PyTorch를 사용하기 위해 nn.Module을 초기화
        super(SPPF, self).__init__()
        
        # SPPF 내부 채널 수
        hidden_channels = in_channels // 2
        
        # Concat 이전 1x1 Conv
        self.conv1 = ConvBNSiLU(
            in_channels = in_channels,
            out_channels = hidden_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            activation = activation
        )
    
        # 3개의 MaxPool2d를 생성 
        self.maxpool = nn.MaxPool2d(
            kernel_size = 5,
            stride = 1,
            padding = 2
        )
        
        # Concat 이후 1x1 Conv
        self.conv2 = ConvBNSiLU(
            in_channels = hidden_channels * 4,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            activation = activation
        )
        
    def forward(self, x):
        # Concat 이전 1x1 Conv
        x = self.conv1(x)
        
        # 첫 번째 MaxPool
        x1 = self.maxpool(x)
        
        # 두 번째 MaxPool
        x2 = self.maxpool(x1)
        
        # 세 번째 MaxPool
        x3 = self.maxpool(x2)
        
        # MaxPool 결과와 x를 채널 방향으로 연결
        x = torch.cat(
            (x, x1, x2, x3), 
            dim = 1
        )
        
        # Concat 이후 1x1 Conv
        x = self.conv2(x)
        
        return x
        
        