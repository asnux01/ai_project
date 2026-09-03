# 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F


# Distribution Focal Loss
class DistributionFocalLoss(nn.Module):

    def __init__(
        self,
        reg_max=16
    ):

        # nn.Module 초기화
        super().__init__()

        # Distribution Bin 개수 저장
        self.reg_max = reg_max


    def forward(
        self,
        pred_dist,
        target_dist,
        target_scores,
        foreground_mask
    ):

        # Positive sample이 없는 경우
        if not foreground_mask.any():
            return pred_dist.sum() * 0.0

        # Positive prediction Distribution 선택
        foreground_pred_dist = pred_dist[foreground_mask]

        # Positive target Distance 선택
        foreground_target_dist = target_dist[foreground_mask]

        # Prediction 형태 변환
        foreground_pred_dist = (
            foreground_pred_dist.view(
                -1,
                4,
                self.reg_max
            )
        )

        # Target 왼쪽 Bin 계산
        left_bin = foreground_target_dist.floor().long()

        # Target 오른쪽 Bin 계산
        right_bin = left_bin + 1

        # 왼쪽 Bin 가중치 계산
        left_weight = (
            right_bin.to(dtype=foreground_target_dist.dtype)
            - foreground_target_dist
        )

        # 오른쪽 Bin 가중치 계산
        right_weight = 1.0 - left_weight

        # Cross Entropy 계산을 위해 Prediction 형태 변환
        pred_logits = (
            foreground_pred_dist
            .reshape(
                -1,
                self.reg_max
            )
        )

        # 왼쪽 Bin Cross Entropy 계산
        left_loss = F.cross_entropy(
            pred_logits,
            left_bin.reshape(-1),
            reduction="none"
        )

        # 오른쪽 Bin Cross Entropy 계산
        right_loss = F.cross_entropy(
            pred_logits,
            right_bin.reshape(-1),
            reduction="none"
        )

        # 두 Bin의 가중 Loss 계산
        dfl_loss = (
            left_loss * left_weight.reshape(-1)
            +
            right_loss * right_weight.reshape(-1)
        )

        # Bbox의 네 방향 Loss로 형태 변환
        dfl_loss = dfl_loss.view(-1, 4)

        # 네 방향 Loss 평균 계산
        dfl_loss = dfl_loss.mean(dim=-1)

        # Positive sample의 가중치 계산
        bbox_weights = (
            target_scores.sum(
                dim=-1
            )[foreground_mask]
        )

        # Loss 정규화 값 계산
        normalizer = (
            target_scores.sum()
            .clamp(min=1.0)
        )

        # 최종 DFL Loss 계산
        dfl_loss = (
            (
                dfl_loss
                * bbox_weights
            ).sum()
            / normalizer
        )

        return dfl_loss