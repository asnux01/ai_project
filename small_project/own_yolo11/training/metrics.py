"""YOLO11의 decoded output을 후처리하고 COCO 방식 mAP를 계산한다."""

# 예측 bbox와 metric 입력을 Tensor로 다룬다.
import torch

# 클래스가 다른 bbox는 서로 억제하지 않는
# batched NMS를 사용한다.
from torchvision.ops import (
    batched_nms,
)


def xywh_to_xyxy(
    boxes,
):
    """
    중심점 xywh bbox를
    좌상단/우하단 xyxy bbox로 변환한다.
    """

    if not isinstance(
        boxes,
        torch.Tensor,
    ):
        raise TypeError(
            "boxes는 Tensor여야 합니다."
        )

    if (
        boxes.ndim != 2
        or boxes.shape[-1] != 4
    ):
        raise ValueError(
            "boxes shape은 [N, 4]여야 합니다."
        )

    # 원본 Tensor를 직접 변경하지 않도록 복사한다.
    converted = (
        boxes.clone()
    )

    # 기존 값을 덮어쓰기 전에
    # 중심점과 너비·높이를 분리한다.
    center_x = boxes[:, 0]
    center_y = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    # xywh를 xyxy로 변환한다.
    converted[:, 0] = (
        center_x
        - width / 2.0
    )

    converted[:, 1] = (
        center_y
        - height / 2.0
    )

    converted[:, 2] = (
        center_x
        + width / 2.0
    )

    converted[:, 3] = (
        center_y
        + height / 2.0
    )

    return converted


def postprocess_predictions(
    decoded_output,
    num_classes,
    image_size,
    confidence_threshold=0.001,
    nms_iou_threshold=0.7,
    max_detections=300,
):
    """
    모델의 decoded output에
    confidence filtering과 NMS를 적용한다.

    Args:
        decoded_output:
            shape:
                [B, 4 + num_classes, N]

            앞의 4개 채널:
                xywh bbox

            나머지 채널:
                sigmoid가 적용된 class 확률

        num_classes:
            객체 탐지 클래스 수

        image_size:
            모델 입력 이미지 크기

        confidence_threshold:
            이 값보다 낮은 class 확률은 제거

        nms_iou_threshold:
            NMS에서 사용할 IoU 기준

        max_detections:
            이미지마다 최종적으로 남길 최대 예측 수

    Returns:
        predictions:
            이미지별 탐지 결과

            [
                {
                    "boxes": [N, 4],
                    "scores": [N],
                    "labels": [N],
                },
                ...
            ]
    """

    # --------------------------------------------------
    # 1. 입력 형식 검사
    # --------------------------------------------------

    if not isinstance(
        decoded_output,
        torch.Tensor,
    ):
        raise TypeError(
            "decoded_output은 Tensor여야 합니다."
        )

    if decoded_output.ndim != 3:
        raise ValueError(
            "decoded_output shape은 "
            "[B, 4 + C, N]이어야 합니다."
        )

    # bbox 4채널과 class 채널 수가 맞는지 확인한다.
    if (
        decoded_output.shape[1]
        != 4 + num_classes
    ):
        raise ValueError(
            "decoded_output의 채널 수가 "
            "num_classes와 맞지 않습니다."
        )

    if image_size <= 0:
        raise ValueError(
            "image_size는 0보다 커야 합니다."
        )

    if not (
        0.0
        <= confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "confidence_threshold는 "
            "0과 1 사이여야 합니다."
        )

    if not (
        0.0
        < nms_iou_threshold
        <= 1.0
    ):
        raise ValueError(
            "nms_iou_threshold는 "
            "0보다 크고 1 이하여야 합니다."
        )

    if max_detections <= 0:
        raise ValueError(
            "max_detections는 1 이상이어야 합니다."
        )

    # --------------------------------------------------
    # 2. 이미지별 후처리
    # --------------------------------------------------

    batch_size = (
        decoded_output.shape[0]
    )

    predictions = []

    for batch_index in range(
        batch_size
    ):
        # [4, N]을 [N, 4]로 변환한다.
        boxes = (
            decoded_output[
                batch_index,
                :4,
            ]
            .transpose(
                0,
                1,
            )
        )

        # Head가 반환한 xywh를
        # NMS가 사용하는 xyxy로 변환한다.
        boxes = xywh_to_xyxy(
            boxes
        )

        # [C, N]을 [N, C]로 변환한다.
        class_probabilities = (
            decoded_output[
                batch_index,
                4:,
            ]
            .transpose(
                0,
                1,
            )
        )

        # 각 위치에서 확률이 가장 높은
        # class와 그 확률을 선택한다.
        scores, labels = (
            class_probabilities.max(
                dim=1
            )
        )

        # --------------------------------------------------
        # 3. Confidence filtering
        # --------------------------------------------------

        confidence_mask = (
            scores
            >= confidence_threshold
        )

        boxes = boxes[
            confidence_mask
        ]

        scores = scores[
            confidence_mask
        ]

        labels = labels[
            confidence_mask
        ]

        # bbox가 모델 입력 범위를 벗어나지 않게 제한한다.
        boxes.clamp_(
            min=0.0,
            max=float(
                image_size
            ),
        )

        # 너비 또는 높이가 0 이하인 bbox를 제거한다.
        if boxes.numel() > 0:

            valid_mask = (
                (
                    boxes[:, 2]
                    > boxes[:, 0]
                )
                & (
                    boxes[:, 3]
                    > boxes[:, 1]
                )
            )

            boxes = boxes[
                valid_mask
            ]

            scores = scores[
                valid_mask
            ]

            labels = labels[
                valid_mask
            ]

        # --------------------------------------------------
        # 4. 클래스별 NMS
        # --------------------------------------------------

        if boxes.numel() > 0:

            # batched_nms는 label이 다른 bbox를
            # 서로 독립적으로 처리한다.
            keep_indices = (
                batched_nms(
                    boxes=boxes,
                    scores=scores,
                    idxs=labels,
                    iou_threshold=(
                        nms_iou_threshold
                    ),
                )
            )

            # 점수가 높은 결과 중
            # 최대 max_detections개만 유지한다.
            keep_indices = (
                keep_indices[
                    :max_detections
                ]
            )

            boxes = boxes[
                keep_indices
            ]

            scores = scores[
                keep_indices
            ]

            labels = labels[
                keep_indices
            ]

        # TorchMetrics가 요구하는 형식으로 저장한다.
        predictions.append(
            {
                "boxes": boxes,
                "scores": scores,
                "labels": labels.to(
                    dtype=torch.int64
                ),
            }
        )

    return predictions


class DetectionMAP:
    """
    TorchMetrics의 MeanAveragePrecision을
    YOLO11 출력 형식에 연결한다.
    """

    def __init__(
        self,
        num_classes,
        image_size,
        confidence_threshold=0.001,
        nms_iou_threshold=0.7,
        max_detections=300,
    ):
        """mAP 계산 객체를 생성한다."""

        # --------------------------------------------------
        # 1. 기본 설정 검사
        # --------------------------------------------------

        if num_classes <= 0:
            raise ValueError(
                "num_classes는 1 이상이어야 합니다."
            )

        if image_size <= 0:
            raise ValueError(
                "image_size는 0보다 커야 합니다."
            )

        # --------------------------------------------------
        # 2. 선택 패키지 import
        # --------------------------------------------------

        try:
            # mAP를 활성화했을 때만
            # TorchMetrics를 불러온다.
            from torchmetrics.detection.mean_ap import (
                MeanAveragePrecision,
            )

        except ImportError as error:
            raise ImportError(
                "mAP를 사용하려면 "
                "torchmetrics와 pycocotools가 "
                "설치되어야 합니다. "
                "calculate_map=False이면 "
                "mAP 없이 학습할 수 있습니다."
            ) from error

        # 후처리에 사용할 설정 저장
        self.num_classes = int(
            num_classes
        )

        self.image_size = int(
            image_size
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.nms_iou_threshold = float(
            nms_iou_threshold
        )

        self.max_detections = int(
            max_detections
        )

        # Dataset target과 후처리 결과가
        # 모두 xyxy 형식이므로 box_format도 xyxy로 지정한다.
        self.metric = (
            MeanAveragePrecision(
                box_format="xyxy",
                iou_type="bbox",
                class_metrics=False,
            )
        )

    def reset(self):
        """
        이전 epoch에서 누적된
        예측과 정답을 초기화한다.
        """

        self.metric.reset()

    def update(
        self,
        decoded_output,
        targets,
    ):
        """
        한 validation batch의
        예측과 정답을 metric에 추가한다.
        """

        # YOLO11 decoded output에
        # confidence filtering과 NMS를 적용한다.
        predictions = (
            postprocess_predictions(
                decoded_output=(
                    decoded_output
                ),
                num_classes=(
                    self.num_classes
                ),
                image_size=(
                    self.image_size
                ),
                confidence_threshold=(
                    self.confidence_threshold
                ),
                nms_iou_threshold=(
                    self.nms_iou_threshold
                ),
                max_detections=(
                    self.max_detections
                ),
            )
        )

        # 예측과 정답의 이미지 수가 같아야 한다.
        if (
            len(predictions)
            != len(targets)
        ):
            raise ValueError(
                "예측 batch 크기와 "
                "target 개수가 다릅니다."
            )

        # pycocotools 기반 평가는 CPU에서 동작하므로
        # 예측 Tensor를 계산 그래프에서 분리해 CPU로 옮긴다.
        cpu_predictions = [
            {
                "boxes": (
                    prediction["boxes"]
                    .detach()
                    .cpu()
                ),
                "scores": (
                    prediction["scores"]
                    .detach()
                    .cpu()
                ),
                "labels": (
                    prediction["labels"]
                    .detach()
                    .cpu()
                ),
            }
            for prediction in predictions
        ]

        # target도 동일하게 CPU Tensor로 정리한다.
        cpu_targets = [
            {
                "boxes": (
                    target["boxes"]
                    .detach()
                    .cpu()
                ),
                "labels": (
                    target["labels"]
                    .detach()
                    .cpu()
                ),
            }
            for target in targets
        ]

        # 현재 batch 예측과 정답을 누적한다.
        self.metric.update(
            cpu_predictions,
            cpu_targets,
        )

    def compute(self):
        """
        현재 epoch에 누적된
        COCO mAP와 recall을 계산한다.
        """

        result = (
            self.metric.compute()
        )

        # 로그 및 checkpoint에 저장하기 쉽도록
        # scalar Tensor를 Python float으로 변환한다.
        return {
            "map": float(
                result["map"].item()
            ),
            "map_50": float(
                result["map_50"].item()
            ),
            "map_75": float(
                result["map_75"].item()
            ),
            "mar_100": float(
                result["mar_100"].item()
            ),
        }