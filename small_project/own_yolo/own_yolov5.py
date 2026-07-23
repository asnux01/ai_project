import torch
import torch.nn as nn

# 필요한 모듈 가져옴
from backbone import Backbone
from neck import Neck

# Backbone과 Neck을 결합한 YOLOv5 모듈
class OwnYOLOv5(nn.Module):

    def __init__(
        self,
        in_channels = 3,       # 입력 이미지 채널 수
        activation = "silu"    # 활성화 함수
    ):

        super(OwnYOLOv5, self).__init__()

        # Backbone 생성
        self.backbone = Backbone(
            in_channels = in_channels,
            activation = activation
        )

        # Backbone의 출력 채널 리스트를 Neck에 전달
        self.neck = Neck(
            in_channels = self.backbone.out_channels,
            activation = activation
        )

        # 이후 Head에 전달할 출력 채널 리스트
        self.out_channels = self.neck.out_channels

    def forward(self, x):

        # Backbone 결과 생성
        output = self.backbone(x)

        # Neck 결과 생성
        output = self.neck(output)

        return output