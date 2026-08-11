#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch.nn as nn

from .module import Backbone, Neck, Head
from .scales import get_scale_factors, normalize_scale_name

class Yolov11(nn.Module):
    
    # 초기화
    def __init__(
        self,
        num_classes,
        scale="n",
        reg_max=16,
        strides=(8,16,32)
    ):
        
        # PyTorch 사용을 위해 nn.Module 쵝화
        super(Yolov11, self).__init__()
        
        # 유효성 검사
        # 클래스 수는 Head 출력 채널 수로 사용되므로
        # 1 이상의 정수여야 함
        if (isinstance(num_classes, bool) 
            or not isinstance(num_classes, int)):
            raise TypeError(
                "num_classes는 정수여야 합니다."
            )

        if num_classes <= 0:
            raise ValueError(
                "num_classes는 1 이상이어야 합니다."
            )

        # reg_max는 DFL 분포 구간 수로 사용되므로
        # 1 이상의 정수여야 함
        if (isinstance(reg_max, bool)
            or not isinstance(reg_max, int)):
            raise TypeError(
                "reg_max는 정수여야 합니다."
            )

        if reg_max <= 0:
            raise ValueError(
                "reg_max는 1 이상이어야 합니다."
            )

        # Head는 P3, P4, P5 세 특징을 사용하므로
        # stride 값도 정확히 세 개가 필요
        if (not isinstance(strides, (tuple,list))
            or len(strides) != 3):
            raise ValueError(
                "strides는 P3, P4, P5용 값 "
                "3개여야 합니다."
            )

        # 0 이하의 stride는 anchor 좌표 변환에 사용할 수 없음
        if any(float(stride) <= 0 for stride in strides):
            raise ValueError(
                "모든 stride 값은 0보다 커야 합니다."
            )
        
        # 입력된 스케일 이름을 검사하고 소문자로 정규화
        self.scale = normalize_scale_name(scale)
        
        # 스케일 이름을 실제 모델 생성 계수로 변환
        (depth_factor, width_factor, max_channels) = get_scale_factors(self.scale)
        
        # 외부 접근 가능 파라미터
        # Checkpoint 저장 및 추론 모델 복원에 필요한 설정
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        
        # Backbone
        self.backbone = Backbone(
            depth_factor=depth_factor,
            width_factor=width_factor,
            max_channels=max_channels
        )
        
        # Neck
        self.neck = Neck(
            channels=self.backbone.out_channels,
            depth_factor=depth_factor
        )
        
        # Detect Head
        self.head = Head(
            num_classes=num_classes,
            in_channels=self.neck.out_channels,
            reg_max=reg_max,
            strides=strides
        )
        
    # 포워드
    def forward(self, x):
        
        # Backbone
        features = self.backbone(x)
        
        # Neck
        features = self.neck(features)
        
        # Head
        x = self.head(features)
        
        # 반환
        return x
        
        