# 라이브러리
import torch
import torch.nn as nn

from .bbox_loss import BoundingBoxLoss
from .dfl_loss import DistributionFocalLoss
from .assigner import TaskAlignedAssigner


# YOLO11 Detection Loss
class YOLO11DetectionLoss(nn.Module):

    def __init__(
        self,
        num_classes=80,
        reg_max=16,
        strides=(8, 16, 32),
        box_gain=7.5,
        cls_gain=0.5,
        dfl_gain=1.5,
        tal_topk=10
    ):

        # nn.Module 초기화
        super().__init__()

        # Detection 설정 저장
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides

        # Detect Head 출력 채널 수 계산
        self.num_outputs = (
            self.num_classes
            + 4 * self.reg_max
        )

        # Loss Gain 저장
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain

        # Classification Loss 생성
        self.cls_loss_fn = (
            nn.BCEWithLogitsLoss(
                reduction="none"
            )
        )

        # Bounding Box Loss 생성
        self.bbox_loss_fn = (
            BoundingBoxLoss()
        )

        # Distribution Focal Loss 생성
        self.dfl_loss_fn = (
            DistributionFocalLoss(
                reg_max=self.reg_max
            )
        )

        # TaskAlignedAssigner 생성
        self.assigner = (
            TaskAlignedAssigner(
                topk=tal_topk,
                num_classes=self.num_classes,
                alpha=0.5,
                beta=6.0
            )
        )

        # DFL Bin 값 생성
        self.register_buffer(
            "projection",
            torch.arange(
                self.reg_max,
                dtype=torch.float32,
            ),
            persistent=False
        )


    def forward(
        self,
        predictions,
        batch
    ):

        # Prediction 형태 정리
        pred_dist, pred_scores = (
            self._flatten_predictions(
                predictions
            )
        )

        # Anchor Point 생성
        anchor_points, stride_tensor = (
            self._make_anchors(
                predictions
            )
        )

        # Distribution을 Bbox로 Decode
        pred_bboxes_grid = (
            self._decode_bboxes(
                pred_dist,
                anchor_points
            )
        )

        # Bbox를 실제 이미지 좌표로 변환
        pred_bboxes = (
            pred_bboxes_grid
            * stride_tensor
        )

        # Batch 크기 가져오기
        batch_size = (
            pred_scores.shape[0]
        )

        # GT 데이터 정리
        (
            gt_labels,
            gt_bboxes,
            mask_gt
        ) = self._preprocess_targets(
            batch=batch,
            batch_size=batch_size,
            device=pred_scores.device,
            dtype=pred_bboxes.dtype
        )

        # Prediction과 GT Matching
        with torch.no_grad():

            (
                _target_labels,
                target_bboxes,
                target_scores,
                foreground_mask,
                _target_gt_indices
            ) = self.assigner(
                pred_scores=(
                    pred_scores
                    .detach()
                    .sigmoid()
                ),
                pred_bboxes=(
                    pred_bboxes.detach()
                ),
                anchor_points=(
                    anchor_points
                    * stride_tensor
                ),
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                mask_gt=mask_gt
            )

        # Target score 데이터 타입 변환
        target_scores = (
            target_scores.to(
                dtype=pred_scores.dtype
            )
        )

        # Loss 정규화 값 계산
        normalizer = (
            target_scores.sum()
            .clamp(min=1.0)
        )

        # Classification Loss 계산
        cls_loss = (
            self.cls_loss_fn(
                pred_scores,
                target_scores,
            ).sum()
            / normalizer
        )

        # Bounding Box Loss 계산
        bbox_loss = (
            self.bbox_loss_fn(
                pred_bboxes=pred_bboxes,
                target_bboxes=target_bboxes,
                target_scores=target_scores,
                foreground_mask=foreground_mask
            )
        )

        # Target Bbox를 Grid 좌표로 변환
        target_bboxes_grid = (
            target_bboxes
            / stride_tensor
        )

        # Target Bbox를 ltrb 거리로 변환
        target_dist = (
            self._bbox_to_distance(
                anchor_points=anchor_points,
                target_bboxes=target_bboxes_grid
            )
        )

        # Distribution Focal Loss 계산
        dfl_loss = (
            self.dfl_loss_fn(
                pred_dist=pred_dist,
                target_dist=target_dist,
                target_scores=target_scores,
                foreground_mask=foreground_mask
            )
        )

        # 각 Loss에 Gain 적용
        weighted_box_loss = (
            bbox_loss
            * self.box_gain
        )

        weighted_cls_loss = (
            cls_loss
            * self.cls_gain
        )

        weighted_dfl_loss = (
            dfl_loss
            * self.dfl_gain
        )

        # 전체 Detection Loss 계산
        total_loss = (
            weighted_box_loss
            + weighted_cls_loss
            + weighted_dfl_loss
        )

        # Loss 정보 구성
        loss_items = {
            "box_loss": (
                weighted_box_loss.detach()
            ),
            "cls_loss": (
                weighted_cls_loss.detach()
            ),
            "dfl_loss": (
                weighted_dfl_loss.detach()
            )
        }

        return total_loss, loss_items


    def _flatten_predictions(
        self,
        predictions
    ):

        # Feature Map Prediction 저장
        flattened_predictions = []

        for prediction in predictions:

            # Batch 크기 가져오기
            batch_size = (
                prediction.shape[0]
            )

            # Prediction 형태 변환
            prediction = prediction.view(
                batch_size,
                self.num_outputs,
                -1
            )

            flattened_predictions.append(
                prediction
            )

        # P3, P4, P5 Prediction 결합
        predictions = torch.cat(
            flattened_predictions,
            dim=2
        )

        # Bbox Distribution과 Class 분리
        pred_dist, pred_scores = (
            predictions.split(
                [
                    4 * self.reg_max,
                    self.num_classes,
                ],
                dim=1
            )
        )

        # Tensor 차원 순서 변경
        pred_dist = (
            pred_dist
            .permute(0, 2, 1)
            .contiguous()
        )

        pred_scores = (
            pred_scores
            .permute(0, 2, 1)
            .contiguous()
        )

        return pred_dist, pred_scores


    def _make_anchors(
        self,
        predictions
    ):

        # Anchor Point 저장 공간 생성
        anchor_points = []

        # Stride 저장 공간 생성
        stride_values = []

        for prediction, stride in zip(
            predictions,
            self.strides
        ):

            # Feature Map 크기 가져오기
            height = prediction.shape[2]
            width = prediction.shape[3]

            # X 좌표 생성
            x_coordinates = (
                torch.arange(
                    width,
                    device=prediction.device,
                    dtype=prediction.dtype
                )
                + 0.5
            )

            # Y 좌표 생성
            y_coordinates = (
                torch.arange(
                    height,
                    device=prediction.device,
                    dtype=prediction.dtype
                )
                + 0.5
            )

            # Grid 좌표 생성
            grid_y, grid_x = (
                torch.meshgrid(
                    y_coordinates,
                    x_coordinates,
                    indexing="ij"
                )
            )

            # Anchor Point 생성
            points = torch.stack(
                (
                    grid_x,
                    grid_y
                ),
                dim=-1
            ).reshape(
                -1,
                2
            )

            # 각 Anchor의 Stride 생성
            strides = torch.full(
                (
                    height * width,
                    1
                ),
                stride,
                device=prediction.device,
                dtype=prediction.dtype
            )

            anchor_points.append(
                points
            )

            stride_values.append(
                strides
            )

        # 모든 Feature Map Anchor 결합
        anchor_points = torch.cat(
            anchor_points,
            dim=0
        )

        # 모든 Stride 결합
        stride_tensor = torch.cat(
            stride_values,
            dim=0
        )

        return anchor_points, stride_tensor


    def _decode_bboxes(
        self,
        pred_dist,
        anchor_points
    ):

        # Distribution 형태 변환
        pred_dist = pred_dist.view(
            pred_dist.shape[0],
            pred_dist.shape[1],
            4,
            self.reg_max
        )

        # 각 Bin을 확률로 변환
        pred_prob = torch.softmax(
            pred_dist,
            dim=-1
        )

        # Projection 데이터 타입 변환
        projection = (
            self.projection.to(
                dtype=pred_prob.dtype
            )
        )

        # Distribution의 기대값 계산
        distances = (
            pred_prob
            * projection
        ).sum(dim=-1)

        # Left-Top 거리 가져오기
        left_top = distances[..., 0:2]

        # Right-Bottom 거리 가져오기
        right_bottom = distances[..., 2:4]

        # Bbox 왼쪽 위 좌표 계산
        x1y1 = anchor_points - left_top

        # Bbox 오른쪽 아래 좌표 계산
        x2y2 = anchor_points + right_bottom

        # xyxy Bbox 생성
        pred_bboxes = torch.cat(
            (x1y1, x2y2),
            dim=-1
        )

        return pred_bboxes


    def _preprocess_targets(
        self,
        batch,
        batch_size,
        device,
        dtype
    ):

        # Batch Index 가져오기
        batch_indices = (
            batch["batch_idx"]
            .to(
                device=device,
                dtype=torch.long
            )
        )

        # Class 정보 가져오기
        classes = (
            batch["cls"]
            .to(
                device=device,
                dtype=torch.long
            )
        )

        # Bbox 정보 가져오기
        bboxes = (
            batch["bboxes"]
            .to(
                device=device,
                dtype=dtype
            )
        )

        # 이미지별 객체 개수 계산
        object_counts = torch.bincount(
            batch_indices,
            minlength=batch_size
        )

        # 최대 객체 개수 계산
        max_objects = (
            int(object_counts.max().item())
            if object_counts.numel() > 0
            else 0
        )

        # GT Class 저장 공간 생성
        gt_labels = torch.zeros(
            (
                batch_size,
                max_objects,
                1
            ),
            device=device,
            dtype=torch.long
        )

        # GT Bbox 저장 공간 생성
        gt_bboxes = torch.zeros(
            (
                batch_size,
                max_objects,
                4
            ),
            device=device,
            dtype=dtype
        )

        # 유효한 GT 위치 저장
        mask_gt = torch.zeros(
            (
                batch_size,
                max_objects,
                1
            ),
            device=device,
            dtype=torch.bool
        )

        for batch_index in range(
            batch_size
        ):

            # 현재 이미지 객체 선택
            current_mask = (
                batch_indices
                == batch_index
            )

            # 현재 이미지 객체 개수
            num_objects = int(current_mask.sum().item())

            # 객체가 없는 경우
            if num_objects == 0:
                continue

            # GT Class 저장
            gt_labels[
                batch_index,
                :num_objects,
                0
            ] = classes[
                current_mask
            ]

            # GT Bbox 저장
            gt_bboxes[
                batch_index,
                :num_objects
            ] = bboxes[
                current_mask
            ]

            # 유효한 GT 위치 표시
            mask_gt[
                batch_index,
                :num_objects,
                0
            ] = True

        return gt_labels, gt_bboxes, mask_gt


    def _bbox_to_distance(
        self,
        anchor_points,
        target_bboxes
    ):

        # Target Bbox 좌표 분리
        x1y1 = target_bboxes[..., 0:2]

        x2y2 = target_bboxes[..., 2:4]

        # Anchor에서 Left-Top까지 거리 계산
        left_top = anchor_points - x1y1

        # Anchor에서 Right-Bottom까지 거리 계산
        right_bottom = x2y2 - anchor_points

        # ltrb 거리 생성
        distances = torch.cat(
            (
                left_top,
                right_bottom
            ),
            dim=-1
        )

        # DFL 표현 범위로 제한
        distances = distances.clamp(
            min=0.0,
            max=(
                self.reg_max
                - 1
                - 0.01
            )
        )

        return distances