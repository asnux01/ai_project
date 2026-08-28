#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch
import torch.nn as nn
import math

from ..block import BoxBranch, ClassBranch, DFL
from ..utils import make_anchors, dist2bbox

class Head(nn.Module):

    # 초기화
    def __init__(
        self,
        num_classes,
        in_channels,
        reg_max=16,
        strides=(8, 16, 32)
    ):

        # PyTorch 사용을 위해 nn.Module 초기화
        super(Head, self).__init__()

        # 파라미터
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.num_levels = len(in_channels)

        # 포워드 접근 가능 파라미터
        # 박스 거리 분포 채널 수
        # reg_max=16
        # left, top, right, bottom × 16
        # 4 × 16 = 64
        self.box_channels = 4 * reg_max

        # 박스 분기의 hidden 채널 수
        ch_bh = max(
            16,
            in_channels[0] // 4,
            4 * reg_max
        )

        # 분류 분기의 hidden 채널 수
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

        # DFL 블록
        self.dfl = DFL(
            reg_max=reg_max
        )
        
        # Detect 출력층의 초기 prediction이
        # 객체 탐지 학습에 적합하도록
        # Box/Class bias를 초기화한다.
        self._initialize_biases()

    def _initialize_biases(
        self,
    ):
        """
        Box branch와 Class branch의
        마지막 Conv2d bias를 초기화한다.
        """

        # P3, P4, P5의 stride와
        # 각각의 Box/Class branch를 함께 순회한다.
        for (
            stride,
            box_branch,
            class_branch,
        ) in zip(
            self.strides,
            self.box_branches,
            self.class_branches,
        ):

            # --------------------------------------------------
            # Box branch bias 초기화
            # --------------------------------------------------
            #
            # Box branch의 마지막 Conv는
            #
            # [B, 4 * reg_max, H, W]
            #
            # 형태의 거리 분포를 출력한다.
            #
            # 초기 box prediction이 지나치게 작은 값에
            # 몰리지 않도록 출력 bias를 2.0으로 초기화한다.
            nn.init.constant_(
                box_branch.conv2d.bias,
                2.0,
            )

            # --------------------------------------------------
            # Class branch bias 초기화
            # --------------------------------------------------
            #
            # feature map의 대부분 위치에는 객체가 없으므로
            # 초기 class probability를 매우 낮게 설정한다.
            #
            # stride가 작을수록 feature map 위치 수가 많기 때문에
            # stride별로 서로 다른 초기 bias가 계산된다.
            class_bias = math.log(
                5.0
                / self.num_classes
                / (640.0 / float(stride)) ** 2
            )

            # Class branch 마지막 Conv의 모든 class bias를
            # 동일한 초기값으로 설정한다.
            nn.init.constant_(
                class_branch.conv2d.bias,
                class_bias,
            )
    
    # 포워드
    def forward(self, features):

        # features 순서
        # features[0] = P3
        # features[1] = P4
        # features[2] = P5

        # 배치 크기
        batch_size = features[0].shape[0]

        # 결과 저장 리스트
        box_outputs = []
        class_outputs = []

        # 각 특징 단계의 Box/Class 분기를 실행
        for index in range(self.num_levels):

            # 현재 특징 맵
            feature = features[index]

            # 현재 박스 분기
            box_branch = self.box_branches[index]

            # 현재 분류 분기
            class_branch = self.class_branches[index]

            # 박스 분기 실행
            box_output = box_branch(
                feature
            )

            # 분류 분기 실행
            class_output = class_branch(
                feature
            )

            # 박스 출력 저장
            box_outputs.append(
                box_output
            )

            # 분류 출력 저장 
            class_outputs.append(
                class_output
            )

        # 각 단계의 박스 출력 평탄화
        box_reshape_outputs = []

        for box_output in box_outputs:

            # [B, 4 × reg_max, H, W]
            #           ->
            # [B, 4 × reg_max, H × W]
            box_output = box_output.reshape(
                batch_size,
                self.box_channels,
                -1
            )

            # 평탄화된 박스 출력 저장
            box_reshape_outputs.append(
                box_output
            )

        # P3, P4, P5의 모든 박스 위치를 연결
        # P3: [B, 64, 6400]
        # P4: [B, 64, 1600]
        # P5: [B, 64,  400]
        # Result: [B, 64, 8400]
        box_logits = torch.cat(
            box_reshape_outputs,
            dim=2
        )

        # 각 단계의 분류 출력 평탄화
        class_reshape_outputs = []

        for class_output in class_outputs:

            # [B, num_classes, H, W]
            #          ->
            # [B, num_classes, H × W]
            class_output = class_output.reshape(
                batch_size,
                self.num_classes,
                -1
            )

            # 평탄화된 분류 출력 저장
            class_reshape_outputs.append(
                class_output
            )

        # P3, P4, P5의 모든 분류 위치를 연결
        # P3: [B, num_classes, 6400]
        # P4: [B, num_classes, 1600]
        # P5: [B, num_classes,  400]
        # Result: [B, num_classes, 8400]
        class_logits = torch.cat(
            class_reshape_outputs,
            dim=2
        )

        # 학습용 원본 데이터 저장

        raw_outputs = {
            "box_logits": box_logits,
            "class_logits": class_logits,
            "features": features
        }

        # 학습용 데이터 반환
        if self.training:
            return raw_outputs

        # Inference
        # DFL Projection
        # [B, 4 × reg_max, 8400]
        #           ->
        # [B, 4, 8400]
        # 거리 분포를 ltrb 거리로 변환
        distance = self.dfl(
            box_logits
        )

        # 중심 좌표와 stride 생성
        # anchor_points:
        # [8400, 2]
        # stride_tensor:
        # [8400, 1]
        anchor_points, stride_tensor = make_anchors(
            features=features,
            strides=self.strides,
            grid_cell_offset=0.5
        )

        # 중심 좌표 차원 변환
        # [8400, 2]
        #    ->
        # [2, 8400]
        anchor_points = anchor_points.transpose(
            0,
            1
        )

        # [2, 8400]
        #     ->
        # [1, 2, 8400]
        anchor_points = anchor_points.unsqueeze(
            0
        )

        # stride 차원 변환
        # [8400, 1]
        #    ->
        # [1, 8400]
        stride_tensor = stride_tensor.transpose(
            0,
            1
        )

        # [1, 8400]
        #     ->
        # [1, 1, 8400]
        stride_tensor = stride_tensor.unsqueeze(
            0
        )

        # 중심 좌표와 네 방향 거리를 xywh로 변환
        # Grid point
        #     +
        # left, top, right, bottom distances
        #               ->
        # x, y, weight, height
        boxes = dist2bbox(
            distance=distance,
            anchor_points=anchor_points,
            xywh=True
        )

        # stride로 feature-map 좌표를 픽셀 좌표로 변환
        # Feature-map coordinates
        #            ↓
        # Original-image pixel coordinates
        boxes = boxes * stride_tensor

        # 분류 logit을 0~1 범위의 확률로 변환
        # Raw class logits
        #         ↓
        # Class probabilities between 0 and 1
        class_probabilities = torch.sigmoid(
            class_logits
        )

        # 박스 출력과 분류 출력을 합침
        # boxes:
        # [B, 4, 8400]
        # class_probabilities:
        # [B, num_classes, 8400]
        # final_output:
        # [B, 4 + num_classes, 8400]
        final_output = torch.cat(
            (boxes, class_probabilities),
            dim=1
        )

        # 후처리 및 학습 코드로 반환
        return final_output, raw_outputs