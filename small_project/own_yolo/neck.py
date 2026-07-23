import torch
import torch.nn as nn

from convbnsilu import ConvBNSiLU
from c3 import C3

# C3 & ConvBNSiLU & concat & Upsample을 사용한 Neck 구조를 구현한 모듈
class Neck(nn.Module):
    
    def __init__(
        self,
        in_channels,        # Backbone에서 넘겨받은 입력 채널 리스트
        activation="silu"   # 활성화 함수
    ):
        
        # PyTorch를 사용하기 위해 nn.Module을 초기화
        super(Neck, self).__init__()
        
        # class 수
        ncls = 80
        
        # Backbone 입력 채널
        in_channel_256 = in_channels[0]
        in_channel_512 = in_channels[1]
        in_channel_1024 = in_channels[2]
        c = (5 + ncls) * 3

        # Head로 전달할 채널 리스트
        self.out_channels = [c, c, c]
        
        # Bottom-UP 스테이지
        # 1024 -> 512 Conv
        self.stage_BU_conv0 = ConvBNSiLU(
            in_channels=in_channel_1024,
            out_channels=in_channel_512,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation
        )
        
        # Upsmapling
        self.up = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )
        
        # 1024 -> 512 C3
        self.stage_BU_c3_0 = C3(
            in_channels=in_channel_1024,
            out_channels=in_channel_512,
            bottleneck_count=3,
            shortcut=False,
            activation=activation
        )
        
        # 512 -> 256 Conv
        self.stage_BU_conv1 = ConvBNSiLU(
            in_channels=in_channel_512,
            out_channels=in_channel_256,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation
        )
        
        # 512 -> 256 C3
        self.stage_BU_c3_1 = C3(
            in_channels=in_channel_512,
            out_channels=in_channel_256,
            bottleneck_count=3,
            shortcut=False,
            activation=activation
        )
        
        # Top-Down 스테이지
        # 256, 80x80 -> 40x40 Conv
        self.stage_TD_conv0 = ConvBNSiLU(
            in_channels=in_channel_256,
            out_channels=in_channel_256,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation
        )
        
        # 512 -> 512 C3
        self.stage_TD_C3_0 = C3(
            in_channels=in_channel_512,
            out_channels=in_channel_512,
            bottleneck_count=3,
            shortcut=False,
            activation=activation
        )
        
        # 512, 40x40 -> 20x20 Conv
        self.stage_TD_conv1 = ConvBNSiLU(
            in_channels=in_channel_512,
            out_channels=in_channel_512,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation
        )
        
        # 1024 -> 1024 C3
        self.stage_TD_C3_1 = C3(
            in_channels=in_channel_1024,
            out_channels=in_channel_1024,
            bottleneck_count=3,
            shortcut=False,
            activation=activation
        )
        
        # Conv2d 스테이지
        # 80x80x256 Conv2d
        self.stage_2d_conv0 = nn.Conv2d(
            in_channels=in_channel_256,
            out_channels=c,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 40x40x512 Conv2d
        self.stage_2d_conv1 = nn.Conv2d(
            in_channels=in_channel_512,
            out_channels=c,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
                
        # 20x20x1024 Conv2d
        self.stage_2d_conv2 = nn.Conv2d(
            in_channels=in_channel_1024,
            out_channels=c,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
    def forward(self, backbone_out):
        
        # Backbone 출력 분리
        stage3_fm = backbone_out[0]
        stage4_fm = backbone_out[1]
        stage6_fm = backbone_out[2]
        
        # 1024 → 512 Conv
        x = self.stage_BU_conv0(stage6_fm)
        pass0 = x
        
        # 20×20 → 40×40 Upsampling
        x = self.up(x)
        
        # 스테이지 4 결과와 Concat
        x = torch.cat(
            (x, stage4_fm),
            dim = 1
        )
        
        # 1024 → 512 C3
        x = self.stage_BU_c3_0(x)
        
        # 512 → 256 Conv
        x = self.stage_BU_conv1(x)
        pass1 = x
        
        # 40×40 → 80×80 Upsampling
        x = self.up(x)
        
        # 스테이지 3 결과와 Concat
        x = torch.cat(
            (x, stage3_fm),
            dim = 1
        )
        
        # 512 → 256 C3
        x = self.stage_BU_c3_1(x)
        
        # Conv2d 전달: 80x80x256
        conv2d_0 = x
        
        # 256, 80×80 → 40×40
        x = self.stage_TD_conv0(x)
        
        # pass1과 연결
        x = torch.cat(
            (x, pass1),
            dim = 1
        )
        
        # 512 → 512
        x = self.stage_TD_C3_0(x)
        
        # Conv2d 전달: 40x40x512
        conv2d_1 = x

        # 512, 40×40 → 20×20
        x = self.stage_TD_conv1(x)

        # pass0와 연결
        x = torch.cat(
            (x, pass0),
            dim = 1
        )

        # 1024 → 1024
        x = self.stage_TD_C3_1(x)

        # Conv2d 전달: 20x20x1024
        conv2d_2 = x
        
        # Conv2d
        head0 = self.stage_2d_conv0(conv2d_0)
        head1 = self.stage_2d_conv1(conv2d_1)
        head2 = self.stage_2d_conv2(conv2d_2)
        
        return [head0, head1, head2]