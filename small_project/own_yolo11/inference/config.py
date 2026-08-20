"""추론 실행에 필요한 설정과 유효성 검사를 정의한다."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """YOLO11 이미지 추론 설정."""

    # 학습 코드가 저장한 best.pt 또는 last.pt 경로
    weights: Path

    # None이면 checkpoint의 model_config/train_config에 저장된 크기를 사용한다.
    image_size: int | None = None

    # Ultralytics NMS로 전달할 confidence/IoU와 최대 탐지 개수
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.7
    max_detections: int = 300

    # auto는 CUDA → MPS → CPU 순서로 사용 가능한 장치를 고른다.
    device: str = "auto"
    batch_size: int = 1

    # 현재 Blackwell 환경의 FP16 오류 때문에 AMP는 안전하게 False가 기본이다.
    use_amp: bool = False

    # 학습 검증이 EMA 모델로 수행됐으므로 추론도 EMA를 우선 사용한다.
    prefer_ema: bool = True

    # None은 모든 클래스, tuple은 지정한 클래스만 남긴다.
    classes: tuple[int, ...] | None = None
    agnostic_nms: bool = False

    def __post_init__(self):
        """경로를 정규화하고 잘못된 값을 추론 전에 차단한다."""

        # 상대 경로와 ~를 실제 절대 경로로 바꿔 이후 모듈이 같은 파일을 참조하게 한다.
        weights = Path(self.weights).expanduser().resolve()
        object.__setattr__(self, "weights", weights)

        if weights.suffix.lower() != ".pt":
            raise ValueError("weights는 .pt 체크포인트여야 합니다.")

        if not weights.is_file():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {weights}")

        # 입력 크기를 생략한 경우에만 checkpoint 값으로 결정하므로 None을 허용한다.
        if self.image_size is not None:
            if (
                isinstance(self.image_size, bool)
                or not isinstance(self.image_size, int)
                or self.image_size <= 0
            ):
                raise ValueError("image_size는 양의 정수 또는 None이어야 합니다.")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold는 0과 1 사이여야 합니다.")

        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold는 0과 1 사이여야 합니다.")

        if self.max_detections <= 0:
            raise ValueError("max_detections는 1 이상이어야 합니다.")

        if self.batch_size <= 0:
            raise ValueError("batch_size는 1 이상이어야 합니다.")

        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device는 'auto', 'cpu', 'cuda:0' 같은 문자열이어야 합니다.")

        # frozen dataclass이므로 검증된 정규화 값은 object.__setattr__으로 저장한다.
        object.__setattr__(self, "device", self.device.strip().lower())

        if self.classes is not None:
            # 입력 순서를 유지하면서 중복 클래스 번호를 제거한다.
            classes = tuple(dict.fromkeys(int(class_id) for class_id in self.classes))

            if any(class_id < 0 for class_id in classes):
                raise ValueError("classes에는 0 이상의 클래스 번호만 사용할 수 있습니다.")

            object.__setattr__(self, "classes", classes or None)
