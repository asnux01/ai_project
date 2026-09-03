# 라이브러리
import math

import torch
import torch.nn as nn


# Bounding Box Loss
class BoundingBoxLoss(nn.Module):

    def __init__(
        self,
        eps=1e-7
    ):

        # nn.Module 초기화
        super().__init__()

        # 0으로 나누는 것을 방지하기 위한 값
        self.eps = eps


    def forward(
        self,
        pred_bboxes,
        target_bboxes,
        target_scores,
        foreground_mask
    ):

        # Positive sample이 없는 경우
        if not foreground_mask.any():
            return pred_bboxes.sum() * 0.0

        # Positive prediction Bbox 선택
        foreground_pred_bboxes = pred_bboxes[foreground_mask]

        # Positive target Bbox 선택
        foreground_target_bboxes = target_bboxes[foreground_mask]

        # Positive sample의 가중치 계산
        bbox_weights = (
            target_scores.sum(
                dim=-1
            )[foreground_mask]
        )

        # CIoU 계산
        ciou = calculate_ciou(
            foreground_pred_bboxes,
            foreground_target_bboxes,
            eps=self.eps
        )

        # Loss 정규화 값 계산
        normalizer = (
            target_scores.sum()
            .clamp(min=1.0)
        )

        # CIoU Loss 계산
        bbox_loss = (
            (
                (1.0 - ciou)
                * bbox_weights
            ).sum()
            / normalizer
        )

        return bbox_loss


# Complete IoU 계산
def calculate_ciou(
    box1,
    box2,
    eps=1e-7
):

    # Box 좌표 분리
    box1_x1 = box1[:, 0]
    box1_y1 = box1[:, 1]
    box1_x2 = box1[:, 2]
    box1_y2 = box1[:, 3]

    box2_x1 = box2[:, 0]
    box2_y1 = box2[:, 1]
    box2_x2 = box2[:, 2]
    box2_y2 = box2[:, 3]

    # 각 Box의 너비와 높이 계산
    box1_width = (
        box1_x2 - box1_x1
    ).clamp(min=eps)

    box1_height = (
        box1_y2 - box1_y1
    ).clamp(min=eps)

    box2_width = (
        box2_x2 - box2_x1
    ).clamp(min=eps)

    box2_height = (
        box2_y2 - box2_y1
    ).clamp(min=eps)

    # Intersection 좌표 계산
    intersection_x1 = torch.maximum(
        box1_x1,
        box2_x1
    )

    intersection_y1 = torch.maximum(
        box1_y1,
        box2_y1
    )

    intersection_x2 = torch.minimum(
        box1_x2,
        box2_x2
    )

    intersection_y2 = torch.minimum(
        box1_y2,
        box2_y2
    )

    # Intersection 너비와 높이 계산
    intersection_width = (
        intersection_x2
        - intersection_x1
    ).clamp(min=0.0)

    intersection_height = (
        intersection_y2
        - intersection_y1
    ).clamp(min=0.0)

    # Intersection 영역 계산
    intersection_area = (
        intersection_width
        * intersection_height
    )

    # 각 Box의 영역 계산
    box1_area = (
        box1_width
        * box1_height
    )

    box2_area = (
        box2_width
        * box2_height
    )

    # Union 영역 계산
    union_area = (
        box1_area
        + box2_area
        - intersection_area
    )

    # IoU 계산
    iou = (
        intersection_area
        / (union_area + eps)
    )

    # 각 Box의 중심 좌표 계산
    box1_center_x = (
        box1_x1 + box1_x2
    ) / 2.0

    box1_center_y = (
        box1_y1 + box1_y2
    ) / 2.0

    box2_center_x = (
        box2_x1 + box2_x2
    ) / 2.0

    box2_center_y = (
        box2_y1 + box2_y2
    ) / 2.0

    # 중심점 사이의 거리 제곱 계산
    center_distance_squared = (
        (
            box1_center_x
            - box2_center_x
        ) ** 2
        +
        (
            box1_center_y
            - box2_center_y
        ) ** 2
    )

    # 두 Box를 포함하는 최소 Box 좌표 계산
    enclosing_x1 = torch.minimum(
        box1_x1,
        box2_x1
    )

    enclosing_y1 = torch.minimum(
        box1_y1,
        box2_y1
    )

    enclosing_x2 = torch.maximum(
        box1_x2,
        box2_x2
    )

    enclosing_y2 = torch.maximum(
        box1_y2,
        box2_y2
    )

    # 최소 포함 Box의 너비와 높이 계산
    enclosing_width = (
        enclosing_x2
        - enclosing_x1
    )

    enclosing_height = (
        enclosing_y2
        - enclosing_y1
    )

    # 최소 포함 Box의 대각선 길이 제곱 계산
    enclosing_diagonal_squared = (
        enclosing_width ** 2
        + enclosing_height ** 2
        + eps
    )

    # 두 Box의 종횡비 차이 계산
    aspect_ratio_difference = (
        4.0
        / (math.pi ** 2)
        * (
            torch.atan(
                box2_width
                / box2_height
            )
            -
            torch.atan(
                box1_width
                / box1_height
            )
        ) ** 2
    )

    # 종횡비 가중치 계산
    alpha = (
        aspect_ratio_difference
        /
        (
            1.0
            - iou
            + aspect_ratio_difference
            + eps
        )
    )

    # CIoU 계산
    ciou = (
        iou
        -
        (
            center_distance_squared
            / enclosing_diagonal_squared
        )
        -
        (
            alpha
            * aspect_ratio_difference
        )
    )

    return ciou