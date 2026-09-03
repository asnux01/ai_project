# 라이브러리
import math

import torch
import torch.nn as nn


# Task Aligned Assigner
class TaskAlignedAssigner(nn.Module):

    def __init__(
        self,
        topk=10,
        num_classes=80,
        alpha=0.5,
        beta=6.0,
        eps=1e-9
    ):

        # nn.Module 초기화
        super().__init__()

        # Assigner 설정 저장
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps


    @torch.no_grad()
    def forward(
        self,
        pred_scores,
        pred_bboxes,
        anchor_points,
        gt_labels,
        gt_bboxes,
        mask_gt
    ):

        # Batch 크기 가져오기
        batch_size = pred_scores.shape[0]

        # 전체 Anchor 개수 가져오기
        num_anchors = pred_scores.shape[1]

        # 이미지당 최대 GT 개수 가져오기
        max_gt = gt_bboxes.shape[1]

        # GT가 없는 경우
        if max_gt == 0:

            target_labels = torch.full(
                (batch_size, num_anchors),
                self.num_classes,
                device=pred_scores.device,
                dtype=torch.long
            )

            target_bboxes = torch.zeros_like(
                pred_bboxes
            )

            target_scores = torch.zeros_like(
                pred_scores
            )

            foreground_mask = torch.zeros(
                (batch_size, num_anchors),
                device=pred_scores.device,
                dtype=torch.bool
            )

            target_gt_indices = torch.zeros(
                (batch_size, num_anchors),
                device=pred_scores.device,
                dtype=torch.long
            )

            return (
                target_labels,
                target_bboxes,
                target_scores,
                foreground_mask,
                target_gt_indices
            )

        # GT 내부에 위치한 Anchor 선택
        candidate_mask = (
            self._select_candidates_in_gts(
                anchor_points=anchor_points,
                gt_bboxes=gt_bboxes,
                mask_gt=mask_gt
            )
        )

        # Alignment Metric과 CIoU 계산
        (
            alignment_metric,
            overlaps
        ) = self._get_alignment_metrics(
            pred_scores=pred_scores,
            pred_bboxes=pred_bboxes,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            candidate_mask=candidate_mask
        )

        # GT별 Top-K Anchor 선택
        topk_mask = (
            self._select_topk_candidates(
                alignment_metric=(
                    alignment_metric
                ),
                candidate_mask=(
                    candidate_mask
                ),
                mask_gt=mask_gt
            )
        )

        # 최종 Positive 후보 생성
        positive_mask = (
            topk_mask
            & candidate_mask
            & mask_gt.bool()
        )

        # 여러 GT에 할당된 Anchor 정리
        (
            target_gt_indices,
            foreground_mask,
            positive_mask
        ) = self._resolve_multiple_assignments(
            positive_mask=positive_mask,
            overlaps=overlaps
        )

        # Target 생성
        (
            target_labels,
            target_bboxes,
            target_scores
        ) = self._build_targets(
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            target_gt_indices=(
                target_gt_indices
            ),
            foreground_mask=(
                foreground_mask
            ),
            pred_scores=pred_scores
        )

        # Positive Alignment Metric만 유지
        positive_alignment = (
            alignment_metric
            * positive_mask.to(
                alignment_metric.dtype
            )
        )

        # Positive CIoU만 유지
        positive_overlaps = (
            overlaps
            * positive_mask.to(
                overlaps.dtype
            )
        )

        # GT별 최대 Alignment Metric 계산
        max_alignment = (
            positive_alignment.amax(
                dim=-1,
                keepdim=True
            )
        )

        # GT별 최대 CIoU 계산
        max_overlap = (
            positive_overlaps.amax(
                dim=-1,
                keepdim=True
            )
        )

        # Alignment Metric 정규화
        normalized_alignment = (
            positive_alignment
            * max_overlap
            / (max_alignment + self.eps)
        )

        # Anchor별 최종 품질 점수 계산
        anchor_quality = (
            normalized_alignment.amax(
                dim=1
            )
            .unsqueeze(-1)
        )

        # Target score에 품질 점수 적용
        target_scores = target_scores * anchor_quality

        return (
            target_labels,
            target_bboxes,
            target_scores,
            foreground_mask,
            target_gt_indices
        )


    def _select_candidates_in_gts(
        self,
        anchor_points,
        gt_bboxes,
        mask_gt
    ):

        # Anchor X 좌표 가져오기
        anchor_x = anchor_points[:, 0].view(1, 1, -1)

        # Anchor Y 좌표 가져오기
        anchor_y = anchor_points[:, 1].view(1, 1, -1)

        # GT Bbox 좌표 가져오기
        gt_x1 = gt_bboxes[..., 0:1]
        gt_y1 = gt_bboxes[..., 1:2]
        gt_x2 = gt_bboxes[..., 2:3]
        gt_y2 = gt_bboxes[..., 3:4]

        # Anchor가 GT 내부에 있는지 확인
        candidate_mask = (
            (anchor_x - gt_x1 > self.eps)
            & (anchor_y - gt_y1 > self.eps)
            & (gt_x2 - anchor_x > self.eps)
            & (gt_y2 - anchor_y > self.eps)
        )

        # 유효한 GT만 사용
        candidate_mask = candidate_mask & mask_gt.bool()

        return candidate_mask


    def _get_alignment_metrics(
        self,
        pred_scores,
        pred_bboxes,
        gt_labels,
        gt_bboxes,
        candidate_mask
    ):

        # Tensor 크기 가져오기
        batch_size = pred_scores.shape[0]
        num_anchors = pred_scores.shape[1]
        max_gt = gt_bboxes.shape[1]

        # GT Class Index 가져오기
        gt_class_indices = (
            gt_labels[..., 0].long()
            .clamp(
                min=0,
                max=self.num_classes - 1
            )
        )

        # Class Index를 Anchor 개수만큼 확장
        class_indices = (
            gt_class_indices
            .unsqueeze(-1)
            .expand(batch_size, max_gt, num_anchors)
        )

        # 각 GT Class에 대한 Prediction Score 선택
        class_scores = torch.gather(
            pred_scores.permute(0, 2, 1),
            dim=1,
            index=class_indices
        )

        # GT와 Prediction Bbox의 CIoU 계산
        overlaps = self._pairwise_ciou(
            gt_bboxes=gt_bboxes,
            pred_bboxes=pred_bboxes
        )

        # 음수 CIoU 제거
        overlaps = overlaps.clamp(min=0.0)

        # Task Alignment Metric 계산
        alignment_metric = (
            class_scores.pow(self.alpha)
            * overlaps.pow(self.beta)
        )

        # 후보가 아닌 Anchor Metric 제거
        alignment_metric = (
            alignment_metric
            * candidate_mask.to(alignment_metric.dtype)
        )

        overlaps = (
            overlaps
            * candidate_mask.to(overlaps.dtype)
        )

        return alignment_metric, overlaps


    def _select_topk_candidates(
        self,
        alignment_metric,
        candidate_mask,
        mask_gt
    ):

        # 실제 사용할 Top-K 값 계산
        topk = min(
            self.topk,
            alignment_metric.shape[-1],
        )

        # 후보가 아닌 Anchor의 Metric 제거
        masked_metric = (
            alignment_metric.masked_fill(
                ~candidate_mask,
                -1.0
            )
        )

        # GT별 높은 Metric의 Anchor Index 선택
        _, topk_indices = torch.topk(
            masked_metric,
            k=topk,
            dim=-1,
            largest=True,
        )

        # Top-K Mask 저장 공간 생성
        topk_mask = torch.zeros_like(
            candidate_mask
        )

        # 선택된 Anchor 위치 표시
        topk_mask.scatter_(
            dim=-1,
            index=topk_indices,
            value=True,
        )

        # 실제 후보 Anchor만 유지
        topk_mask = topk_mask & candidate_mask

        # 유효한 GT만 유지
        topk_mask = topk_mask & mask_gt.bool()

        return topk_mask


    def _resolve_multiple_assignments(
        self,
        positive_mask,
        overlaps
    ):

        # 각 Anchor가 할당된 GT 개수 계산
        assignment_count = positive_mask.sum(dim=1)

        # 여러 GT에 동시에 할당된 Anchor 확인
        multiple_mask = assignment_count > 1

        # 중복 할당이 있는 경우
        if multiple_mask.any():

            # Anchor별 가장 높은 CIoU의 GT 선택
            best_gt_indices = overlaps.argmax(dim=1)

            # 가장 좋은 GT만 표시할 Mask 생성
            best_gt_mask = torch.zeros_like(positive_mask)

            best_gt_mask.scatter_(
                dim=1,
                index=best_gt_indices.unsqueeze(1),
                value=True
            )

            # 중복 Anchor는 가장 높은 CIoU GT만 유지
            positive_mask = torch.where(
                multiple_mask.unsqueeze(1),
                best_gt_mask,
                positive_mask
            )

        # 최종 Foreground Anchor 확인
        foreground_mask = positive_mask.any(dim=1)

        # 각 Anchor가 할당된 GT Index 계산
        target_gt_indices = (
            positive_mask.to(
                torch.int64
            )
            .argmax(dim=1)
        )

        return (
            target_gt_indices,
            foreground_mask,
            positive_mask
        )


    def _build_targets(
        self,
        gt_labels,
        gt_bboxes,
        target_gt_indices,
        foreground_mask,
        pred_scores
    ):

        # Batch 크기 가져오기
        batch_size = target_gt_indices.shape[0]

        # Anchor 개수 가져오기
        num_anchors = target_gt_indices.shape[1]

        # Batch Index 생성
        batch_indices = (
            torch.arange(
                batch_size,
                device=gt_labels.device
            )
            .unsqueeze(1)
            .expand(batch_size, num_anchors)
        )

        # Anchor에 할당된 GT Class 가져오기
        target_labels = (
            gt_labels[
                batch_indices,
                target_gt_indices,
                0
            ].long()
        )

        # Anchor에 할당된 GT Bbox 가져오기
        target_bboxes = (
            gt_bboxes[
                batch_indices,
                target_gt_indices
            ]
        )

        # Target Score 저장 공간 생성
        target_scores = torch.zeros(
            (
                batch_size,
                num_anchors,
                self.num_classes
            ),
            device=pred_scores.device,
            dtype=pred_scores.dtype
        )

        # Class Index 범위 제한
        target_class_indices = (
            target_labels.clamp(
                min=0,
                max=self.num_classes - 1
            )
        )

        # One-Hot Class Target 생성
        target_scores.scatter_(
            dim=2,
            index=(
                target_class_indices
                .unsqueeze(-1)
            ),
            value=1.0
        )

        # Background Anchor의 Class Target 제거
        target_scores = (
            target_scores
            * foreground_mask
            .unsqueeze(-1)
            .to(target_scores.dtype)
        )

        # Background Label 설정
        target_labels = torch.where(
            foreground_mask,
            target_labels,
            torch.full_like(
                target_labels,
                self.num_classes
            )
        )

        return (
            target_labels,
            target_bboxes,
            target_scores
        )


    def _pairwise_ciou(
        self,
        gt_bboxes,
        pred_bboxes
    ):

        # GT Bbox 차원 확장
        gt_boxes = gt_bboxes.unsqueeze(2)

        # Prediction Bbox 차원 확장
        pred_boxes = pred_bboxes.unsqueeze(1)

        # GT Bbox 좌표 분리
        gt_x1 = gt_boxes[..., 0]
        gt_y1 = gt_boxes[..., 1]
        gt_x2 = gt_boxes[..., 2]
        gt_y2 = gt_boxes[..., 3]

        # Prediction Bbox 좌표 분리
        pred_x1 = pred_boxes[..., 0]
        pred_y1 = pred_boxes[..., 1]
        pred_x2 = pred_boxes[..., 2]
        pred_y2 = pred_boxes[..., 3]

        # GT Bbox 너비와 높이 계산
        gt_width = (gt_x2 - gt_x1).clamp(min=self.eps)
        gt_height = (gt_y2 - gt_y1).clamp(min=self.eps)

        # Prediction Bbox 너비와 높이 계산
        pred_width = (pred_x2 - pred_x1).clamp( min=self.eps)
        pred_height = (pred_y2 - pred_y1).clamp(min=self.eps)

        # Intersection 좌표 계산
        intersection_x1 = torch.maximum(
            gt_x1,
            pred_x1
        )

        intersection_y1 = torch.maximum(
            gt_y1,
            pred_y1
        )

        intersection_x2 = torch.minimum(
            gt_x2,
            pred_x2
        )

        intersection_y2 = torch.minimum(
            gt_y2,
            pred_y2
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

        # GT와 Prediction 영역 계산
        gt_area = gt_width * gt_height
        pred_area = pred_width * pred_height

        # Union 영역 계산
        union_area = gt_area + pred_area - intersection_area

        # IoU 계산
        iou = intersection_area / (union_area + self.eps)

        # GT 중심 좌표 계산
        gt_center_x = (gt_x1 + gt_x2) / 2.0

        gt_center_y = (gt_y1 + gt_y2) / 2.0

        # Prediction 중심 좌표 계산
        pred_center_x = (pred_x1 + pred_x2) / 2.0

        pred_center_y = (pred_y1 + pred_y2) / 2.0

        # 중심점 거리 제곱 계산
        center_distance_squared = (
            (gt_center_x - pred_center_x) ** 2
            + (gt_center_y - pred_center_y) ** 2
        )

        # 두 Bbox를 포함하는 최소 Bbox 계산
        enclosing_x1 = torch.minimum(gt_x1, pred_x1)
        enclosing_y1 = torch.minimum(gt_y1, pred_y1)
        enclosing_x2 = torch.maximum(gt_x2, pred_x2)
        enclosing_y2 = torch.maximum(gt_y2, pred_y2)

        # 최소 포함 Bbox 크기 계산
        enclosing_width = enclosing_x2 - enclosing_x1
        enclosing_height = enclosing_y2 - enclosing_y1

        # 최소 포함 Bbox 대각선 제곱 계산
        enclosing_diagonal_squared = (
            enclosing_width ** 2
            + enclosing_height ** 2
            + self.eps
        )

        # Bbox 종횡비 차이 계산
        aspect_ratio_difference = (
            4.0 / (math.pi ** 2)
            * (
                torch.atan(pred_width / pred_height)
                - torch.atan(gt_width / gt_height)
            ) ** 2
        )

        # 종횡비 가중치 계산
        alpha_ciou = (
            aspect_ratio_difference
            / (1.0 - iou + aspect_ratio_difference + self.eps)
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
                alpha_ciou
                * aspect_ratio_difference
            )
        )

        return ciou