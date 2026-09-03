# 라이브러리
import math

import torch
import torch.nn as nn

from ..block import BoxBranch, ClassBranch

class Head(nn.Module):

    def __init__(
        self,
        num_classes,
        in_channels,
        reg_max=16,
        strides=(8, 16, 32)
    ):

        super(Head, self).__init__()

        # 파라미터
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.num_levels = len(in_channels)

        # Box hidden 채널 수
        ch_bh = max(
            16,
            in_channels[0] // 4,
            4 * reg_max
        )

        # Class hidden 채널 수
        ch_ch = max(
            in_channels[0],
            min(num_classes, 100)
        )

        # P3, P4, P5에 독립적인 박스 분기를 생성
        self.box_branches = nn.ModuleList()
        
        for channel in in_channels:

            box_branch = BoxBranch(
                in_channels=channel,
                hidden_channels=ch_bh,
                reg_max=reg_max
            )
            
            self.box_branches.append(box_branch)
            
        # P3, P4, P5에 독립적인 분류 분기를 생성
        self.class_branches = nn.ModuleList()

        for channel in in_channels:

            class_branch = ClassBranch(
                in_channels=channel,
                hidden_channels=ch_ch,
                num_classes=num_classes
            )

            self.class_branches.append(class_branch)
        
        # Detect head bias 초기화
        self._initialize_biases()

    def _initialize_biases(self):

        # P3, P4, P5의 stride와
        # 각각의 Box/Class branch를 함께 순회
        for (
            stride,
            box_branch,
            class_branch
        ) in zip(
            self.strides,
            self.box_branches,
            self.class_branches
        ):

            # Box branch bias 초기화
            nn.init.constant_(box_branch.conv2d.bias, 2.0)

            # Class branch bias 초기화
            class_bias = math.log(
                5.0 / self.num_classes
                / (640.0 / float(stride)) ** 2
            )

            # Class branch 마지막 Conv의 모든 class bias를
            # 동일한 초기값으로 설정한다.
            nn.init.constant_(
                class_branch.conv2d.bias,
                class_bias
            )
    
    # 포워드
    def forward(self, features):
        
        # 결과 저장 리스트
        outputs = []

        # 각 특징 단계의 Box/Class 분기를 실행
        for index in range(self.num_levels):

            # 현재 특징 맵 가져오기
            feature = features[index]

            # Box Distribution 예측
            box_branch = self.box_branches[index]
            box_output = box_branch(feature)

            # Class Logit 예측
            class_branch = self.class_branches[index]
            class_output = class_branch(feature)

            # Bbox와 Class Prediction 결합
            output = torch.cat(
                (box_output, class_output),
                dim=1
            )
            
            # 현재 Feature Level Prediction 저장
            outputs.append(output)
        
        return outputs