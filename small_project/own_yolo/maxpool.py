import torch
import torch.nn as nn

# MaxPool을 수행하는 모듈
# 주변 값들 중 가장 큰 특징만 선택
class MaxPool2d(nn.Module):
    
    def __init__(
        self,
        kernel_size,    # MaxPool 필터의 크기
        stride,         # MaxPool 필터가 이동하는 간격
        padding         # MaxPool 전에 입력 가장자리에 추가하는 값의 크기
    ):
        # PyTorch를 사용하기 위해 nn.Module을 초기화
        super(MaxPool2d, self).__init__()

        # 5x5 MaxPool
        self.maxpool = nn.MaxPool2d(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        
    def forward(self, x):
        # 입력 x를 MaxPool에 통과
        x = self.maxpool(x)

        return x
