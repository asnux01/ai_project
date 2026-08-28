#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch.nn as nn

from ..layer import Conv

class ClassBranch(nn.Module):
    
    # 초기화
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes
    ):
        
        # PyTorch 사용을 위해 nn.Module 초기화
        super(ClassBranch, self).__init__()
        
        # 파라미터
        ch_i = in_channels
        ch_h = hidden_channels
        ch_o = num_classes
        
        # 포워드 접근 가능 파라미터
        self.nc = num_classes
        
        # DWConv0
        self.dwconv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_i,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=ch_i
        )
                
        # Conv0
        self.conv0 = Conv(
            in_channels=ch_i,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
        )
                
        # DWConv1
        self.dwconv1 = Conv(
            in_channels=ch_h,
            out_channels=ch_h,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=ch_h
        )
                        
        # Conv1
        self.conv1 = Conv(
            in_channels=ch_h,
            out_channels=ch_h,
            kernel_size=1,
            stride=1,
            padding=0
        )
                
        # Conv2d
        # 각 위치의 sigmoid 적용 전 클래스 logit 출력
        self.conv2d = nn.Conv2d(
            in_channels=ch_h,
            out_channels=ch_o,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
    # 포워드
    def forward(self, x):
        
        # class branch
        x = self.dwconv0(x)
        x = self.conv0(x)
        x = self.dwconv1(x)
        x = self.conv1(x)
        x = self.conv2d(x)
        
        # 반환
        return x