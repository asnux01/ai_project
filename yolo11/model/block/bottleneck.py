#----------------------------------------------
# 라이브러리
#----------------------------------------------
import torch.nn as nn

from ..layer import Conv

class Bottleneck(nn.Module):
    
    # 초기화
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        e=0.5
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(Bottleneck, self).__init__()
        
        # 파라미터
        ch_i = in_channels              # 입력 채널
        ch_h = int(out_channels * e)    # hidden 채널
        ch_o = out_channels             # 출력 채널
        
        # shortcut 사용 여부
        self.shortcut = shortcut and ch_i == ch_o
          
        # 첫 번째 Conv
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_h,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # 두 번째 Conv
        self.conv1 = Conv(
            in_channels=ch_h,
            out_channels=ch_o,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        
    # 포워드
    def forward(self, x):
        
        # Residual
        res = x
        
        # Conv
        x = self.conv0(x)
        x = self.conv1(x)
        
        # shortcut 사용 여부 구분
        if self.shortcut:
            x = res + x
        
        # 반환
        return x