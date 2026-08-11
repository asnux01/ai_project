"""YOLO11 학습에 사용하는 설정을 한곳에서 관리한다."""

# dataclass는 여러 학습 설정을 하나의 객체로 묶을 때 사용한다.
from dataclasses import asdict, dataclass, field

# pathlib.Path는 Windows와 Linux 경로를 같은 방식으로 다루게 해준다.
from pathlib import Path

# Any는 설정 딕셔너리 안에 여러 자료형이 들어갈 수 있음을 나타낸다.
from typing import Any


def _serialize_config_value(value):
    """Path와 tuple을 checkpoint에 저장하기 쉬운 기본 자료형으로 변환한다."""

    # torch.save가 운영체제별 Path 객체에 의존하지 않도록
    # 모든 Path를 일반 문자열로 바꾼다.
    if isinstance(value, Path):
        return str(value)

    # 중첩된 딕셔너리 안의 값도 재귀적으로 변환한다.
    if isinstance(value, dict):
        return {
            key: _serialize_config_value(item)
            for key, item in value.items()
        }

    # JSON이나 checkpoint에서 다루기 편하도록 tuple도 list로 바꾼다.
    if isinstance(value, (list, tuple)):
        return [
            _serialize_config_value(item)
            for item in value
        ]

    # 숫자, 문자열, bool, None은 그대로 반환한다.
    return value


@dataclass
class TrainConfig:
    """학습 실행에 필요한 경로와 하이퍼파라미터를 보관한다."""

    # --------------------------------------------------
    # 프로젝트와 데이터셋 경로
    # --------------------------------------------------

    # config.py가 있는 own_yolo11 폴더를 프로젝트 기준 경로로 사용한다.
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )

    # COCO 데이터가 실제로 저장되는 서버의 물리 경로다.
    dataset_storage_dir: Path = Path(
        "/home/jblee/datasets/coco"
    )

    # 프로젝트 내부에서 접근할 심볼릭 링크 경로다.
    # None이면 <project_root>/datasets/coco를 자동으로 사용한다.
    dataset_link_dir: Path | None = None

    # 데이터가 없을 때 COCO2017을 자동으로 다운로드한다.
    auto_download: bool = True

    # 인증서 hostname 오류가 발생한 경우에만 COCO 공식 호스트에 한해
    # 인증서 검증 없이 다시 시도한다.
    # 서버의 인증서 문제가 해결되면 False로 바꾸는 것이 더 안전하다.
    allow_insecure_ssl_fallback: bool = True

    # --------------------------------------------------
    # 모델 설정
    # --------------------------------------------------

    image_size: int = 640
    num_classes: int = 80
    scale: str = "n"
    reg_max: int = 16
    strides: tuple[int, int, int] = (8, 16, 32)

    # --------------------------------------------------
    # 기본 학습 설정
    # --------------------------------------------------

    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0005

    # 전체 epoch 중 앞부분에서 학습률을 서서히 증가시킨다.
    warmup_epochs: float = 3.0

    # Cosine schedule 마지막 학습률은
    # learning_rate * min_lr_ratio로 계산한다.
    min_lr_ratio: float = 0.01

    # --------------------------------------------------
    # Loss 설정
    # --------------------------------------------------

    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    tal_topk: int = 10

    # --------------------------------------------------
    # 학습 효율과 안정성 설정
    # --------------------------------------------------

    # CUDA에서 float16 자동 혼합 정밀도를 사용한다.
    use_amp: bool = True

    # 학습 모델의 이동평균 가중치를 별도로 관리한다.
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_tau: float = 2000.0

    # Gradient가 지나치게 커졌을 때 최대 norm을 제한한다.
    # 사용하지 않으려면 None으로 설정한다.
    max_grad_norm: float | None = 10.0

    # mAP는 후처리와 추가 연산이 필요하므로 기본값은 끈다.
    # 기본 학습이 정상 동작한 다음 True로 변경한다.
    calculate_map: bool = False
    confidence_threshold: float = 0.001
    nms_iou_threshold: float = 0.7
    max_detections: int = 300

    # --------------------------------------------------
    # DataLoader 설정
    # --------------------------------------------------

    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # --------------------------------------------------
    # 데이터 증강 설정
    # --------------------------------------------------

    horizontal_flip_probability: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.02

    # --------------------------------------------------
    # 재현성과 기록 설정
    # --------------------------------------------------

    seed: int = 42

    # True는 재현성을 높이지만 일부 CUDA 연산이 느려질 수 있다.
    deterministic: bool = False
    log_interval: int = 20

    # --------------------------------------------------
    # 저장 및 학습 재개 설정
    # --------------------------------------------------

    checkpoint_dir: Path | None = None
    log_dir: Path | None = None

    # 이어서 학습할 checkpoint 경로다.
    # None이면 새로운 학습을 시작한다.
    resume_checkpoint: Path | None = None

    def __post_init__(self):
        """문자열 경로를 정규화하고 잘못된 설정을 일찍 검사한다."""

        # ~ 문자를 사용자 홈 디렉터리로 확장하고
        # 프로젝트 기준 경로는 절대 경로로 정규화한다.
        self.project_root = Path(
            self.project_root
        ).expanduser().resolve()

        # 실제 COCO 파일이 저장되는 경로는 심볼릭 링크의 대상이므로
        # 최종 물리 경로까지 해석해도 문제가 없다.
        self.dataset_storage_dir = Path(
            self.dataset_storage_dir
        ).expanduser().resolve()

        # 링크 경로를 입력하지 않으면 프로젝트 아래의
        # datasets/coco를 고정된 링크 위치로 사용한다.
        if self.dataset_link_dir is None:
            self.dataset_link_dir = (
                self.project_root
                / "datasets"
                / "coco"
            )

        else:
            # 심볼릭 링크 경로에는 resolve()를 사용하지 않는다.
            # 이미 링크가 존재하면 resolve()가 링크 자체가 아니라
            # /home/jblee/datasets/coco를 반환할 수 있기 때문이다.
            self.dataset_link_dir = Path(
                self.dataset_link_dir
            ).expanduser().absolute()

        # 별도 경로를 지정하지 않으면 프로젝트 아래에
        # checkpoint와 log 디렉터리를 만든다.
        if self.checkpoint_dir is None:
            self.checkpoint_dir = (
                self.project_root
                / "checkpoints"
            )

        else:
            self.checkpoint_dir = Path(
                self.checkpoint_dir
            ).expanduser().resolve()

        if self.log_dir is None:
            self.log_dir = (
                self.project_root
                / "logs"
            )

        else:
            self.log_dir = Path(
                self.log_dir
            ).expanduser().resolve()

        # resume_checkpoint가 지정된 경우에만 절대 경로로 변환한다.
        if self.resume_checkpoint is not None:
            self.resume_checkpoint = Path(
                self.resume_checkpoint
            ).expanduser().resolve()

        # 모델이나 DataLoader를 만들기 전에 모든 설정을 검사한다.
        self._validate()

    def _validate(self):
        """학습 중간이 아니라 실행 직후 설정 오류를 알려준다."""

        # bool은 Python에서 int의 하위 자료형이므로
        # True가 image_size=1로 통과하지 않게 별도로 제외한다.
        if (
            isinstance(self.image_size, bool)
            or not isinstance(
                self.image_size,
                int,
            )
        ):
            raise TypeError(
                "image_size는 정수여야 합니다."
            )

        # 현재 모델 Head는 P3, P4, P5 세 단계 출력을 사용하므로
        # strides에도 양수 3개가 필요하다.
        if (
            not isinstance(
                self.strides,
                (tuple, list),
            )
            or len(self.strides) != 3
            or any(
                isinstance(stride, bool)
                or not isinstance(
                    stride,
                    (int, float),
                )
                or stride <= 0
                for stride in self.strides
            )
        ):
            raise ValueError(
                "strides에는 양수 3개가 필요합니다."
            )

        # Backbone과 Neck의 downsampling/upsampling 크기가 맞도록
        # 입력 크기는 가장 큰 stride의 배수로 제한한다.
        if (
            self.image_size <= 0
            or self.image_size
            % max(self.strides)
            != 0
        ):
            raise ValueError(
                "image_size는 0보다 크고 "
                "가장 큰 stride의 배수여야 합니다."
            )

        # 모델 구조에 직접 사용되는 설정 검사
        if self.num_classes <= 0:
            raise ValueError(
                "num_classes는 1 이상이어야 합니다."
            )

        if self.scale not in {
            "n",
            "s",
            "m",
            "l",
            "x",
        }:
            raise ValueError(
                "scale은 n, s, m, l, x 중 "
                "하나여야 합니다."
            )

        if self.reg_max <= 0:
            raise ValueError(
                "reg_max는 1 이상이어야 합니다."
            )

        # 기본 학습 하이퍼파라미터 검사
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size는 1 이상이어야 합니다."
            )

        if self.epochs <= 0:
            raise ValueError(
                "epochs는 1 이상이어야 합니다."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate는 0보다 커야 합니다."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay는 0 이상이어야 합니다."
            )

        if not (
            0.0
            <= self.warmup_epochs
            < self.epochs
        ):
            raise ValueError(
                "warmup_epochs는 0 이상이고 "
                "epochs보다 작아야 합니다."
            )

        if not (
            0.0
            < self.min_lr_ratio
            <= 1.0
        ):
            raise ValueError(
                "min_lr_ratio는 0보다 크고 "
                "1 이하여야 합니다."
            )

        # Loss와 gradient 안정성 관련 설정 검사
        if self.tal_topk <= 0:
            raise ValueError(
                "tal_topk는 1 이상이어야 합니다."
            )

        if (
            self.max_grad_norm is not None
            and self.max_grad_norm <= 0
        ):
            raise ValueError(
                "max_grad_norm은 0보다 크거나 "
                "None이어야 합니다."
            )

        # DataLoader 관련 설정 검사
        if self.num_workers < 0:
            raise ValueError(
                "num_workers는 0 이상이어야 합니다."
            )

        if self.prefetch_factor <= 0:
            raise ValueError(
                "prefetch_factor는 1 이상이어야 합니다."
            )

        # 증강 확률과 재현성 관련 설정 검사
        if not (
            0.0
            <= self.horizontal_flip_probability
            <= 1.0
        ):
            raise ValueError(
                "horizontal_flip_probability는 "
                "0과 1 사이여야 합니다."
            )

        if self.seed < 0:
            raise ValueError(
                "seed는 0 이상이어야 합니다."
            )

        if self.log_interval <= 0:
            raise ValueError(
                "log_interval은 1 이상이어야 합니다."
            )

        # EMA와 mAP 후처리 관련 설정 검사
        if not (
            0.0
            < self.ema_decay
            < 1.0
        ):
            raise ValueError(
                "ema_decay는 0과 1 사이여야 합니다."
            )

        if self.ema_tau <= 0:
            raise ValueError(
                "ema_tau는 0보다 커야 합니다."
            )

        if not (
            0.0
            <= self.confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "confidence_threshold는 "
                "0과 1 사이여야 합니다."
            )

        if not (
            0.0
            < self.nms_iou_threshold
            <= 1.0
        ):
            raise ValueError(
                "nms_iou_threshold는 0보다 크고 "
                "1 이하여야 합니다."
            )

        if self.max_detections <= 0:
            raise ValueError(
                "max_detections는 1 이상이어야 합니다."
            )

    @property
    def train_image_dir(self):
        """심볼릭 링크 기준의 COCO train 이미지 경로를 반환한다."""

        return (
            self.dataset_link_dir
            / "images"
            / "train2017"
        )

    @property
    def val_image_dir(self):
        """심볼릭 링크 기준의 COCO validation 이미지 경로를 반환한다."""

        return (
            self.dataset_link_dir
            / "images"
            / "val2017"
        )

    @property
    def train_annotation_file(self):
        """COCO train annotation 파일 경로를 반환한다."""

        return (
            self.dataset_link_dir
            / "annotations"
            / "instances_train2017.json"
        )

    @property
    def val_annotation_file(self):
        """COCO validation annotation 파일 경로를 반환한다."""

        return (
            self.dataset_link_dir
            / "annotations"
            / "instances_val2017.json"
        )

    def to_dict(self) -> dict[str, Any]:
        """설정을 checkpoint에 기록할 수 있는 딕셔너리로 반환한다."""

        return _serialize_config_value(
            asdict(self)
        )


def get_config():
    """기본 학습 설정 객체를 생성한다."""

    # train.py는 이 객체 하나를 전달받아 사용한다.
    return TrainConfig()