# --------------------------------------------------
# 라이브러리
# --------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import (
    TaskAlignedAssigner,
    bbox2dist,
    dist2bbox,
    make_anchors,
)


# 연속 거리 target을 인접한 두 분포 구간으로 학습하는 DFL
class DistributionFocalLoss(nn.Module):

    def __init__(
        self,
        reg_max=16,
    ):
        super().__init__()

        self.reg_max = reg_max

    def forward(
        self,
        pred_distribution,
        target_distance,
    ):
        """
        Args:
            pred_distribution:
                각 거리의 확률분포 예측값

                shape:
                    [positive_count * 4, reg_max]

            target_distance:
                실제 left, top, right, bottom 거리

                shape:
                    [positive_count, 4]

        Returns:
            DFL 결과

            shape:
                [positive_count, 1]
        """

        # target이 DFL 범위를 벗어나지 않게 제한
        target_distance = target_distance.clamp(
            min=0,
            max=self.reg_max - 1 - 0.01,
        )

        # target 거리의 왼쪽 정수 위치
        target_left = target_distance.long()

        # target 거리의 오른쪽 정수 위치
        target_right = target_left + 1

        # 실제 target이 왼쪽 값에 가까운 정도
        weight_left = (
            target_right.float()
            - target_distance
        )

        # 실제 target이 오른쪽 값에 가까운 정도
        weight_right = (
            target_distance
            - target_left.float()
        )

        # 왼쪽 정수 위치에 대한 Cross Entropy
        loss_left = F.cross_entropy(
            pred_distribution,
            target_left.reshape(-1),
            reduction="none",
        )

        loss_left = loss_left.reshape(
            target_left.shape
        )

        # 오른쪽 정수 위치에 대한 Cross Entropy
        loss_right = F.cross_entropy(
            pred_distribution,
            target_right.reshape(-1),
            reduction="none",
        )

        loss_right = loss_right.reshape(
            target_right.shape
        )

        # 두 정수 위치의 Loss를 거리 비율에 따라 합침
        loss = (
            loss_left * weight_left
            + loss_right * weight_right
        )

        # left, top, right, bottom의 평균
        loss = loss.mean(
            dim=-1,
            keepdim=True,
        )

        return loss


# positive prediction의 CIoU와 DFL을 계산하는 박스 loss
class BoundingBoxLoss(nn.Module):

    def __init__(
        self,
        reg_max=16,
    ):
        super().__init__()

        self.reg_max = reg_max

        self.dfl_loss = DistributionFocalLoss(
            reg_max=reg_max,
        )

    def forward(
        self,
        pred_distribution,
        pred_boxes,
        anchor_points,
        target_boxes,
        target_scores,
        target_scores_sum,
        foreground_mask,
    ):
        """
        Args:
            pred_distribution:
                Box branch 원시 출력

                [B, N, 4 x reg_max]

            pred_boxes:
                decode된 예측 박스

                [B, N, 4]

            anchor_points:
                Grid 중심 좌표

                [N, 2]

            target_boxes:
                각 positive prediction에 배정된 정답 박스

                [B, N, 4]

            target_scores:
                TaskAlignedAssigner가 만든 target score

                [B, N, num_classes]

            foreground_mask:
                positive prediction 위치

                [B, N]
        """

        # 각 positive prediction의 중요도
        weight = target_scores[
            foreground_mask
        ].sum(
            dim=-1,
            keepdim=True,
        )

        # --------------------------------------------------
        # CIoU Box Loss
        # --------------------------------------------------

        iou = bbox_iou(
            pred_boxes[foreground_mask],
            target_boxes[foreground_mask],
            xywh=False,
            CIoU=True,
        )

        iou = iou.reshape(-1, 1)

        box_loss = (
            (1.0 - iou) * weight
        ).sum()

        box_loss = (
            box_loss
            / target_scores_sum
        )

        # --------------------------------------------------
        # DFL
        # --------------------------------------------------

        # 정답 박스를 anchor point 기준의
        # left, top, right, bottom 거리로 변환
        target_distance = bbox2dist(
            anchor_points,
            target_boxes,
            self.reg_max - 1,
        )

        # Positive prediction만 선택
        positive_distribution = pred_distribution[
            foreground_mask
        ]

        positive_distribution = (
            positive_distribution.reshape(
                -1,
                self.reg_max,
            )
        )

        positive_target_distance = target_distance[
            foreground_mask
        ]

        dfl_loss = self.dfl_loss(
            positive_distribution,
            positive_target_distance,
        )

        dfl_loss = (
            dfl_loss * weight
        ).sum()

        dfl_loss = (
            dfl_loss
            / target_scores_sum
        )

        return box_loss, dfl_loss


# target 할당과 세 loss 결합을 담당하는 전체 detection loss
class YOLO11DetectionLoss(nn.Module):

    def __init__(
        self,
        num_classes=80,
        reg_max=16,
        strides=(8, 16, 32),
        box_gain=7.5,
        cls_gain=0.5,
        dfl_gain=1.5,
        tal_topk=10,
    ):
        super().__init__()

        # 모델 출력 구조와 anchor 생성에 필요한 파라미터
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides

        # 최종 loss에 곱할 box, class, DFL 가중치
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain

        # 각 분류 logit에 적용할 BCE loss
        self.classification_loss = (
            nn.BCEWithLogitsLoss(
                reduction="none",
            )
        )

        # positive 박스의 CIoU 및 DFL을 계산하는 객체
        self.bounding_box_loss = BoundingBoxLoss(
            reg_max=reg_max,
        )

        # 예측과 정답을 연결하는 Task-Aligned Assigner
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=num_classes,
            alpha=0.5,
            beta=6.0,
            stride=list(strides),
        )

        # DFL 분포를 거리 기댓값으로 변환할 고정 구간 값
        # reg_max=16:
        # [0, 1, 2, ..., 15]
        self.register_buffer(
            "projection",
            torch.arange(
                reg_max,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    # 이미지별 가변 길이 target을 batch tensor 형식으로 변환
    def prepare_targets(
        self,
        targets,
        batch_size,
        device,
        dtype,
    ):
        """
        Dataset의 target 리스트를
        TaskAlignedAssigner가 사용할 Tensor로 변환한다.

        입력:

            targets = [
                {
                    "boxes": [N0, 4],
                    "labels": [N0],
                },
                {
                    "boxes": [N1, 4],
                    "labels": [N1],
                },
            ]

        출력:

            gt_labels:
                [B, max_objects, 1]

            gt_boxes:
                [B, max_objects, 4]

            valid_mask:
                [B, max_objects, 1]
        """

        # 한 이미지에 존재하는 최대 객체 수
        max_objects = max(
            (
                target["labels"].shape[0]
                for target in targets
            ),
            default=0,
        )

        # 클래스 정답 저장 공간
        gt_labels = torch.zeros(
            (
                batch_size,
                max_objects,
                1,
            ),
            device=device,
            dtype=torch.long,
        )

        # 박스 정답 저장 공간
        gt_boxes = torch.zeros(
            (
                batch_size,
                max_objects,
                4,
            ),
            device=device,
            dtype=dtype,
        )

        # 실제 객체가 존재하는 위치를 표시
        valid_mask = torch.zeros(
            (
                batch_size,
                max_objects,
                1,
            ),
            device=device,
            dtype=torch.bool,
        )

        # 각 이미지의 target을 Tensor에 저장
        for batch_index, target in enumerate(targets):

            object_count = target["labels"].shape[0]

            # 객체가 없는 이미지
            if object_count == 0:
                continue

            gt_labels[
                batch_index,
                :object_count,
                0,
            ] = target["labels"].to(
                device=device,
                dtype=torch.long,
            )

            gt_boxes[
                batch_index,
                :object_count,
            ] = target["boxes"].to(
                device=device,
                dtype=dtype,
            )

            valid_mask[
                batch_index,
                :object_count,
                0,
            ] = True

        return (
            gt_labels,
            gt_boxes,
            valid_mask,
        )

    # DFL 분포 logit을 feature-map 좌표의 xyxy 박스로 복원
    def decode_boxes(
        self,
        anchor_points,
        pred_distribution,
    ):
        """
        Box branch의 분포 예측을
        xyxy 박스로 변환한다.

        pred_distribution:
            [B, N, 4 x reg_max]

        반환:
            [B, N, 4]
        """

        batch_size = pred_distribution.shape[0]
        anchor_count = pred_distribution.shape[1]

        # [B, N, 4 x reg_max]
        # ↓
        # [B, N, 4, reg_max]
        pred_distribution = pred_distribution.reshape(
            batch_size,
            anchor_count,
            4,
            self.reg_max,
        )

        # 각 거리 구간에 대한 확률
        pred_distribution = pred_distribution.softmax(
            dim=-1
        )

        projection = self.projection.to(
            dtype=pred_distribution.dtype
        )

        # 확률분포의 기대값을 계산
        # [B, N, 4, reg_max]
        # ↓
        # [B, N, 4]
        distance = torch.matmul(
            pred_distribution,
            projection,
        )

        # anchor point와 ltrb 거리를 이용하여
        # xyxy 박스를 생성
        boxes = dist2bbox(
            distance,
            anchor_points,
            xywh=False,
        )

        return boxes

    # raw model output과 target으로 전체 detection loss를 계산
    def forward(
        self,
        predictions,
        targets,
    ):
        """
        Args:
            predictions:
                네 YOLO11 Head의 학습 출력

                {
                    "box_logits":
                        [B, 4 x reg_max, N],

                    "class_logits":
                        [B, num_classes, N],

                    "features":
                        [P3, P4, P5],
                }

            targets:
                Dataset이 반환한 target 리스트

        Returns:
            total_loss:
                역전파에 사용할 전체 Loss

            loss_items:
                출력 및 기록용 개별 Loss
        """

        box_logits = predictions[
            "box_logits"
        ]

        class_logits = predictions[
            "class_logits"
        ]

        features = predictions[
            "features"
        ]

        # Box/Class 출력을 위치가 두 번째 차원이 되도록 변환
        # [B, 4 × reg_max, N]
        # ↓
        # [B, N, 4 × reg_max]
        pred_distribution = box_logits.permute(
            0,
            2,
            1,
        ).contiguous()

        # [B, num_classes, N]
        # ↓
        # [B, N, num_classes]
        pred_scores = class_logits.permute(
            0,
            2,
            1,
        ).contiguous()

        device = pred_scores.device
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        # P3, P4, P5의 anchor 중심점과 stride 값을 생성
        anchor_points, stride_tensor = make_anchors(
            features,
            self.strides,
            0.5,
        )

        anchor_points = anchor_points.to(
            device=device,
            dtype=dtype,
        )

        stride_tensor = stride_tensor.to(
            device=device,
            dtype=dtype,
        )

        # 이미지별 ground truth를 batch tensor로 정리
        (
            gt_labels,
            gt_boxes,
            valid_mask,
        ) = self.prepare_targets(
            targets=targets,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # 예측 분포를 feature-map 좌표의 박스로 복원
        # Feature-map 좌표계의 xyxy 박스
        pred_boxes = self.decode_boxes(
            anchor_points=anchor_points,
            pred_distribution=pred_distribution,
        )

        # Task-Aligned Assigner로 positive prediction의 정답을 결정
        # TaskAlignedAssigner에는 원본 이미지 픽셀 좌표로
        # 예측 박스와 anchor point를 전달한다.
        (
            _,
            assigned_boxes,
            assigned_scores,
            foreground_mask,
            _,
        ) = self.assigner(
            pred_scores.detach().sigmoid(),

            (
                pred_boxes.detach()
                * stride_tensor
            ).to(gt_boxes.dtype),

            anchor_points * stride_tensor,

            gt_labels,
            gt_boxes,
            valid_mask,
        )

        assigned_scores = assigned_scores.to(
            dtype=dtype
        )

        # Positive score의 전체 합
        assigned_scores_sum = (
            assigned_scores.sum()
        ).clamp(
            min=1.0
        )

        # 전체 위치에 대한 classification BCE loss를 계산
        cls_loss = self.classification_loss(
            pred_scores,
            assigned_scores,
        )

        cls_loss = (
            cls_loss.sum()
            / assigned_scores_sum
        )

        # Box와 DFL Loss의 초기값
        # 0이지만 pred_distribution과 연결돼 있어
        # gradient graph가 유지된다.
        box_loss = (
            pred_distribution.sum() * 0.0
        )

        dfl_loss = (
            pred_distribution.sum() * 0.0
        )

        # Positive prediction이 있을 때만 box와 DFL을 계산
        if foreground_mask.any():

            # assigned_boxes는 원본 이미지 픽셀 좌표이므로
            # stride로 나누어 feature-map 좌표로 변환
            assigned_boxes = (
                assigned_boxes
                / stride_tensor
            )

            box_loss, dfl_loss = (
                self.bounding_box_loss(
                    pred_distribution=pred_distribution,
                    pred_boxes=pred_boxes,
                    anchor_points=anchor_points,
                    target_boxes=assigned_boxes,
                    target_scores=assigned_scores,
                    target_scores_sum=assigned_scores_sum,
                    foreground_mask=foreground_mask,
                )
            )

        # 각 loss에 설정한 가중치를 적용
        weighted_box_loss = (
            box_loss * self.box_gain
        )

        weighted_cls_loss = (
            cls_loss * self.cls_gain
        )

        weighted_dfl_loss = (
            dfl_loss * self.dfl_gain
        )

        # 세 loss를 합쳐 역전파에 사용할 전체 loss를 생성
        total_loss = (
            weighted_box_loss
            + weighted_cls_loss
            + weighted_dfl_loss
        )

        # 기록용 값은 gradient graph에서 분리한
        loss_items = {
            "box_loss": (
                weighted_box_loss.detach()
            ),
            "cls_loss": (
                weighted_cls_loss.detach()
            ),
            "dfl_loss": (
                weighted_dfl_loss.detach()
            ),
            "total_loss": (
                total_loss.detach()
            ),
        }

        return total_loss, loss_items