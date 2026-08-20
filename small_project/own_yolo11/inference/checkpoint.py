"""학습 체크포인트에서 추론 모델을 안전하게 복원한다."""

# 복원 결과와 경로를 명확한 자료형으로 관리한다.
from dataclasses import dataclass
from pathlib import Path

# Tensor, device, state_dict 로딩에 사용한다.
import torch

# 학습 때 사용한 것과 정확히 같은 자체 YOLO11 구조다.
from model import Yolov11


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """복원된 모델과 추론에 필요한 체크포인트 메타데이터."""

    # eval()이 적용되고 선택한 장치로 이동된 실제 추론 모델
    model: Yolov11
    device: torch.device

    # 전처리와 결과 해석에 필요한 모델 설정
    image_size: int
    num_classes: int
    strides: tuple[int, ...]

    # EMA와 기본 모델 중 무엇을 불러왔는지 로그에 표시한다.
    weight_source: str
    completed_epoch: int


def select_device(device_spec="auto"):
    """Ultralytics식 auto 선택을 단일 PyTorch device로 정규화한다."""

    # 대소문자와 앞뒤 공백에 관계없이 동일하게 처리한다.
    device_spec = str(device_spec).strip().lower()

    if device_spec == "auto":
        # 서버 추론에서는 CUDA를 가장 먼저 선택한다.
        if torch.cuda.is_available():
            return torch.device("cuda:0")

        # Apple Silicon 환경에서는 CUDA 대신 MPS를 사용할 수 있다.
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        # GPU backend가 하나도 없으면 항상 동작 가능한 CPU로 돌아간다.
        return torch.device("cpu")

    # Ultralytics CLI에서 흔히 사용하는 --device 0 형식도 허용한다.
    if device_spec.isdigit():
        device_spec = f"cuda:{device_spec}"

    # 'cpu', 'cuda', 'cuda:0', 'mps' 문자열을 PyTorch device로 변환한다.
    device = torch.device(device_spec)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device를 요청했지만 torch.cuda.is_available()이 False입니다.")

    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"CUDA device index {device.index}를 사용할 수 없습니다. "
            f"감지된 CUDA 장치 수: {torch.cuda.device_count()}"
        )

    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS device를 요청했지만 사용할 수 없습니다.")

    return device


def _safe_torch_load(checkpoint_path):
    """PyTorch 버전에 맞춰 CPU에서 체크포인트를 읽는다."""

    try:
        # weights_only=True는 신뢰하지 않는 임의 Python 객체의 역직렬화를 제한한다.
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # 구형 PyTorch는 weights_only 인자를 받지 않으므로 자체 생성 파일에 한해 호환한다.
        return torch.load(
            checkpoint_path,
            map_location="cpu",
        )


def _extract_ema_weights(checkpoint):
    """ModelEMA가 저장한 중첩 EMA state_dict를 꺼낸다."""

    # training/checkpoint.py의 ema_state_dict 안에는
    # training/ema.py가 만든 또 하나의 ema_state_dict가 들어 있다.
    ema_container = checkpoint.get("ema_state_dict")

    if not isinstance(ema_container, dict):
        return None

    ema_weights = ema_container.get("ema_state_dict")

    if not isinstance(ema_weights, dict):
        return None

    return ema_weights


def load_inference_model(
    checkpoint_path,
    device="auto",
    prefer_ema=True,
):
    """체크포인트 구조로 모델을 만들고 EMA 또는 기본 가중치를 복원한다."""

    # --------------------------------------------------
    # 1. 체크포인트 파일 읽기
    # --------------------------------------------------
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    checkpoint = _safe_torch_load(checkpoint_path)

    if not isinstance(checkpoint, dict):
        raise TypeError("체크포인트 최상위 값은 dict여야 합니다.")

    # --------------------------------------------------
    # 2. 학습 당시 모델 구조 복원
    # --------------------------------------------------
    model_config = checkpoint.get("model_config")

    if not isinstance(model_config, dict):
        raise KeyError("체크포인트에 model_config가 없습니다.")

    required_model_keys = {
        "num_classes",
        "scale",
        "reg_max",
        "strides",
    }
    missing_keys = required_model_keys - model_config.keys()

    if missing_keys:
        raise KeyError(f"model_config에 필요한 값이 없습니다: {sorted(missing_keys)}")

    # JSON/log 저장 과정에서 list가 된 stride도 모델이 기대하는 tuple로 되돌린다.
    num_classes = int(model_config["num_classes"])
    strides = tuple(int(value) for value in model_config["strides"])

    # 가중치를 넣기 전에 동일한 채널 수와 scale의 빈 모델을 생성한다.
    model = Yolov11(
        num_classes=num_classes,
        scale=str(model_config["scale"]),
        reg_max=int(model_config["reg_max"]),
        strides=strides,
    )

    # --------------------------------------------------
    # 3. EMA 또는 기본 모델 가중치 선택
    # --------------------------------------------------
    ema_weights = _extract_ema_weights(checkpoint)

    if prefer_ema and ema_weights is not None:
        # validation loss와 mAP가 EMA 모델로 계산됐으므로 기본 추론 선택이다.
        weights = ema_weights
        weight_source = "ema_state_dict"
    else:
        # --no-ema 또는 EMA가 없는 checkpoint에서는 실제 학습 모델을 사용한다.
        weights = checkpoint.get("model_state_dict")
        weight_source = "model_state_dict"

    if not isinstance(weights, dict):
        raise KeyError(f"체크포인트에 사용할 {weight_source}가 없습니다.")

    # strict=True로 구조가 다른 checkpoint를 조용히 일부만 로드하는 일을 막는다.
    model.load_state_dict(weights, strict=True)

    # --------------------------------------------------
    # 4. 추론 장치 이동 및 eval 모드
    # --------------------------------------------------
    selected_device = select_device(device)
    model = model.to(selected_device)

    # BatchNorm running statistics를 사용하고 학습 전용 동작을 끈다.
    model.eval()

    # --------------------------------------------------
    # 5. 전처리용 image_size 복원
    # --------------------------------------------------
    train_config = checkpoint.get("train_config")
    train_config = train_config if isinstance(train_config, dict) else {}
    image_size = int(
        model_config.get(
            "image_size",
            train_config.get("image_size", 640),
        )
    )

    if image_size <= 0 or image_size % max(strides) != 0:
        raise ValueError("체크포인트 image_size는 양수이며 최대 stride의 배수여야 합니다.")

    # Predictor가 원본 checkpoint dict를 다시 알 필요가 없도록 필요한 값만 반환한다.
    return LoadedModel(
        model=model,
        device=selected_device,
        image_size=image_size,
        num_classes=num_classes,
        strides=strides,
        weight_source=weight_source,
        completed_epoch=int(checkpoint.get("epoch", 0)),
    )
