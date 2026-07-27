# yolo11_coco_qat_int8_full.py
# ------------------------------------------------------------
# 목적:
#   1. COCO128 또는 COCO 전체 데이터셋을 다운로드한다.
#   2. FP32 pretrained yolo11n.pt를 불러온다.
#   3. Ultralytics YOLO 학습 루프 내부에서 PyTorch QAT를 적용한다.
#   4. fake quant 상태로 QAT fine-tuning을 진행한다.
#   5. QAT 적용 모델을 state_dict 형태로 별도 저장한다.
#   6. QAT 모델을 실제 INT8 모델로 변환한다.                    [추가]
#   7. INT8 모델을 TorchScript(.pt)와 ONNX(.onnx)로 저장한다.   [추가]
#   8. QAT/INT8 파일 크기와 FP32/INT8 CPU latency를 비교한다.   [추가]
# ------------------------------------------------------------
#
# 실행 예시:
#   python yolo11_coco_qat_int8_full.py \
#       --dataset coco128 \
#       --epochs 20 \
#       --qat-backend x86 \
#       --onnx-opset 13
#
# INT8 변환을 생략하고 기존 동작만 수행:
#   python yolo11_coco_qat_int8_full.py \
#       --dataset coco128 \
#       --epochs 20 \
#       --skip-int8-convert
#
# 변환 후 FP32 원본과 INT8 모델의 mAP 비교까지 수행:
#   python yolo11_coco_qat_int8_full.py \
#       --dataset coco128 \
#       --epochs 20 \
#       --eval-after-convert
# ------------------------------------------------------------

import argparse
import copy
import inspect
from pathlib import Path
import time
import warnings

import torch
import torch.nn as nn
import torch.ao.quantization as tq

from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL, SETTINGS
from ultralytics.utils.downloads import download


# ------------------------------------------------------------
# 1. COCO 클래스 이름
# ------------------------------------------------------------

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


# ------------------------------------------------------------
# 2. 데이터셋 다운로드 관련 함수
# ------------------------------------------------------------

def has_jpgs(directory: Path) -> bool:
    """
    특정 폴더 안에 jpg/jpeg/png 이미지가 있는지 확인한다.
    """

    if not directory.exists():
        return False

    return (
        any(directory.glob("*.jpg"))
        or any(directory.glob("*.jpeg"))
        or any(directory.glob("*.png"))
    )


def get_default_datasets_dir() -> Path:
    """
    Ultralytics 기본 datasets_dir를 가져온다.
    실패하면 현재 폴더 아래 datasets를 사용한다.
    """

    try:
        return Path(SETTINGS["datasets_dir"]).expanduser().resolve()
    except Exception:
        return Path("./datasets").resolve()


def ensure_coco128(datasets_dir: Path) -> Path:
    """
    COCO128 다운로드 함수.
    """

    coco128_dir = datasets_dir / "coco128"
    image_dir = coco128_dir / "images" / "train2017"

    if has_jpgs(image_dir):
        print(f"[DATA] COCO128 이미 존재: {coco128_dir}")
        return coco128_dir

    print("[DATA] COCO128 다운로드 시작")
    print(f"[DATA] 저장 위치: {datasets_dir}")

    datasets_dir.mkdir(parents=True, exist_ok=True)

    url = ASSETS_URL + "/coco128.zip"
    download([url], dir=datasets_dir)

    if not has_jpgs(image_dir):
        raise RuntimeError(
            f"COCO128 다운로드 후 이미지를 찾지 못했습니다: {image_dir}"
        )

    print(f"[DATA] COCO128 다운로드 완료: {coco128_dir}")
    return coco128_dir


def ensure_coco_full(datasets_dir: Path, download_test: bool = False) -> Path:
    """
    COCO 2017 전체 데이터셋 다운로드 함수.
    """

    coco_dir = datasets_dir / "coco"
    train_img_dir = coco_dir / "images" / "train2017"
    val_img_dir = coco_dir / "images" / "val2017"
    train_label_dir = coco_dir / "labels" / "train2017"
    val_label_dir = coco_dir / "labels" / "val2017"

    train_ok = has_jpgs(train_img_dir)
    val_ok = has_jpgs(val_img_dir)
    label_ok = train_label_dir.exists() and val_label_dir.exists()

    if train_ok and val_ok and label_ok:
        print(f"[DATA] COCO 전체 데이터셋 이미 존재: {coco_dir}")
        return coco_dir

    print("[DATA] COCO 전체 데이터셋 다운로드 시작")
    print("[DATA] 용량이 크므로 시간이 오래 걸릴 수 있습니다.")
    print(f"[DATA] 저장 위치: {coco_dir}")

    datasets_dir.mkdir(parents=True, exist_ok=True)

    if not label_ok:
        print("[DATA] COCO YOLO labels 다운로드")
        label_url = ASSETS_URL + "/coco2017labels.zip"
        download([label_url], dir=datasets_dir)

    image_urls = []

    if not train_ok:
        image_urls.append("http://images.cocodataset.org/zips/train2017.zip")

    if not val_ok:
        image_urls.append("http://images.cocodataset.org/zips/val2017.zip")

    if download_test:
        test_img_dir = coco_dir / "images" / "test2017"
        if not has_jpgs(test_img_dir):
            image_urls.append("http://images.cocodataset.org/zips/test2017.zip")

    if image_urls:
        print("[DATA] COCO images 다운로드")
        download(image_urls, dir=coco_dir / "images", threads=3)

    if not has_jpgs(train_img_dir):
        raise RuntimeError(f"train2017 이미지를 찾지 못했습니다: {train_img_dir}")

    if not has_jpgs(val_img_dir):
        raise RuntimeError(f"val2017 이미지를 찾지 못했습니다: {val_img_dir}")

    if not train_label_dir.exists():
        raise RuntimeError(f"train2017 라벨을 찾지 못했습니다: {train_label_dir}")

    if not val_label_dir.exists():
        raise RuntimeError(f"val2017 라벨을 찾지 못했습니다: {val_label_dir}")

    print(f"[DATA] COCO 전체 데이터셋 다운로드 완료: {coco_dir}")
    return coco_dir


def write_local_coco_yaml(
    dataset_name: str,
    dataset_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Ultralytics 학습에 사용할 local yaml 파일을 직접 생성한다.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = output_dir / f"{dataset_name}_local.yaml"

    if dataset_name == "coco128":
        train_path = "images/train2017"
        val_path = "images/train2017"

    elif dataset_name == "coco":
        train_path = "images/train2017"
        val_path = "images/val2017"

    else:
        raise ValueError(
            "dataset_name은 coco128 또는 coco여야 합니다."
        )

    lines = []

    lines.append(f"path: {dataset_dir.as_posix()}")
    lines.append(f"train: {train_path}")
    lines.append(f"val: {val_path}")
    lines.append("")
    lines.append("nc: 80")
    lines.append("")
    lines.append("names:")

    for index, name in enumerate(COCO_NAMES):
        lines.append(f"  {index}: {name}")

    yaml_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"[DATA] local yaml 생성 완료: {yaml_path}")

    return yaml_path


def prepare_dataset(args) -> Path:
    """
    args.dataset에 따라 COCO128 또는 COCO 전체를 준비하고,
    local yaml 파일 경로를 반환한다.
    """

    datasets_dir = (
        Path(args.datasets_dir)
        .expanduser()
        .resolve()
    )

    if args.dataset == "coco128":
        dataset_dir = ensure_coco128(
            datasets_dir
        )

    elif args.dataset == "coco":
        dataset_dir = ensure_coco_full(
            datasets_dir=datasets_dir,
            download_test=args.download_test,
        )

    else:
        raise ValueError(
            "--dataset은 coco128 또는 coco만 지원합니다."
        )

    yaml_path = write_local_coco_yaml(
        dataset_name=args.dataset,
        dataset_dir=dataset_dir,
        output_dir=Path(
            args.yaml_dir
        ).resolve(),
    )

    return yaml_path


# ------------------------------------------------------------
# 3. QAT 관련 함수
# ------------------------------------------------------------

class QATNoOpEMA:
    """
    Ultralytics Trainer는 내부에서 self.ema.update(),
    self.ema.update_attr(), self.ema.ema를 사용한다.

    QAT에서는 Conv+BN fuse와 fake quant/observer 삽입 이후
    기존 EMA 업데이트가 key mismatch 또는 shape mismatch를 일으킬 수 있다.

    따라서 실제 EMA 평균 업데이트는 하지 않고,
    Ultralytics Trainer가 기대하는 최소 인터페이스만 제공한다.
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        self.ema = model
        self.updates = 0

    def update(
        self,
        model: nn.Module,
    ):
        return

    def update_attr(
        self,
        model: nn.Module,
        include=(),
        exclude=(),
    ):
        for key in include:
            if hasattr(model, key):
                try:
                    setattr(
                        self.ema,
                        key,
                        getattr(model, key),
                    )

                except Exception:
                    pass

        return


class QuantizedConvBoundary(nn.Module):
    """
    하나의 합성곱 연산 앞뒤에 양자화 경계를 추가한다.

    학습 중에는:

        QuantStub
        → fake quant observer
        → Conv
        → DeQuantStub

    convert() 이후에는:

        FP32 입력
        → Quantize
        → 실제 INT8 Conv
        → DeQuantize
        → FP32 출력

    YOLO의 SiLU, Concat, Upsample, Detect 후처리 등은 FP32로
    유지하고 합성곱 연산만 INT8 커널로 실행하기 위한 경계다.
    """

    def __init__(
        self,
        conv: nn.Module,
    ):
        super().__init__()

        self.quant = tq.QuantStub()
        self.conv = conv
        self.dequant = tq.DeQuantStub()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)

        return x


class TensorOnlyExportWrapper(nn.Module):
    """
    YOLO의 tuple/list/dict 출력을 export 가능한
    대표 Tensor 하나로 정리한다.

    eval 모드의 YOLO detection 모델은 보통 예측 Tensor와
    중간 feature를 tuple로 반환한다.

    TorchScript와 ONNX 출력에는 첫 번째 예측 Tensor만 사용한다.
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(x)

        return extract_first_tensor(
            output
        )


def choose_qat_qconfig(
    backend: str,
):
    """
    QAT backend에 맞는 qconfig를 고른다.
    """

    try:
        torch.backends.quantized.engine = backend

    except Exception:
        pass

    try:
        qconfig = tq.get_default_qat_qconfig(
            backend
        )

        print(
            f"[QAT] backend={backend} qconfig 사용"
        )

        return qconfig

    except Exception as error:
        print(
            f"[QAT] backend={backend} "
            f"qconfig 생성 실패: {error}"
        )

        fallback = (
            "fbgemm"
            if backend in ["x86", "fbgemm"]
            else "qnnpack"
        )

        try:
            torch.backends.quantized.engine = fallback

        except Exception:
            pass

        qconfig = tq.get_default_qat_qconfig(
            fallback
        )

        print(
            f"[QAT] fallback backend={fallback} "
            "qconfig 사용"
        )

        return qconfig


def fuse_yolo_conv_bn_for_qat(
    model: nn.Module,
):
    """
    Ultralytics YOLO의 Conv 블록에서
    conv + bn fuse를 시도한다.
    """

    fused_count = 0
    failed_count = 0

    for name, module in model.named_modules():
        has_conv_bn = (
            hasattr(module, "conv")
            and hasattr(module, "bn")
            and isinstance(
                module.conv,
                nn.Conv2d,
            )
            and isinstance(
                module.bn,
                nn.BatchNorm2d,
            )
        )

        if not has_conv_bn:
            continue

        try:
            tq.fuse_modules_qat(
                module,
                ["conv", "bn"],
                inplace=True,
            )

            module._qat_conv_bn_fused = True
            fused_count += 1

        except Exception:
            failed_count += 1

    print(
        f"[QAT] Conv+BN fuse 성공: "
        f"{fused_count}"
    )

    print(
        f"[QAT] Conv+BN fuse 실패 또는 skip: "
        f"{failed_count}"
    )


def insert_quant_dequant_boundaries_for_yolo_convs(
    model: nn.Module,
):
    """
    YOLO 내부 합성곱을 QuantizedConvBoundary로 감싼다.

    eager-mode 정적 양자화에서 quantized Conv는
    quantized Tensor를 입력받는다.

    YOLO 내부의 SiLU, residual add, concat, detect 연산 전체를
    quantized Tensor로 유지하면 backend가 지원하지 않는
    연산이 발생할 수 있다.

    따라서 각 Conv 앞에서 quantize하고,
    Conv 직후 dequantize한다.

    이 방식은 quant/dequant 비용이 추가되지만
    실제 INT8 Conv 커널을 사용하면서 모델 전체 실행의
    호환성을 높인다.
    """

    wrapped_fused_count = 0
    wrapped_plain_count = 0

    # --------------------------------------------------------
    # 1. Conv+BN이 fuse된 블록 처리
    # --------------------------------------------------------

    for module in model.modules():
        if not getattr(
            module,
            "_qat_conv_bn_fused",
            False,
        ):
            continue

        conv = getattr(
            module,
            "conv",
            None,
        )

        if (
            conv is None
            or isinstance(
                conv,
                QuantizedConvBoundary,
            )
        ):
            continue

        module.conv = QuantizedConvBoundary(
            conv
        )

        wrapped_fused_count += 1

    # --------------------------------------------------------
    # 2. fuse되지 않은 일반 Conv2d 처리
    # --------------------------------------------------------

    protected_module_ids = set()

    for boundary in model.modules():
        if isinstance(
            boundary,
            QuantizedConvBoundary,
        ):
            protected_module_ids.update(
                id(descendant)
                for descendant
                in boundary.modules()
            )

    for parent in list(
        model.modules()
    ):
        if id(parent) in protected_module_ids:
            continue

        for child_name, child in list(
            parent.named_children()
        ):
            if isinstance(
                child,
                QuantizedConvBoundary,
            ):
                continue

            if isinstance(
                child,
                nn.Conv2d,
            ):
                setattr(
                    parent,
                    child_name,
                    QuantizedConvBoundary(
                        child
                    ),
                )

                wrapped_plain_count += 1

    print(
        "[QAT] INT8 Conv 경계 삽입: "
        f"fused={wrapped_fused_count}, "
        f"plain={wrapped_plain_count}"
    )


def disable_qat_for_unsupported_or_sensitive_layers(
    model: nn.Module,
):
    """
    일부 layer는 QAT 대상에서 제외한다.
    """

    disabled = []

    for name, module in model.named_modules():
        lower_name = name.lower()

        should_disable = False

        if "dfl" in lower_name:
            should_disable = True

        if isinstance(
            module,
            nn.Upsample,
        ):
            should_disable = True

        if should_disable:
            module.qconfig = None
            disabled.append(name)

    if disabled:
        print("[QAT] QAT 제외 모듈:")

        for name in disabled:
            print(f"  - {name}")


def enable_qat_callback_factory(
    args,
):
    """
    Ultralytics callback은 함수만 받을 수 있으므로,
    args를 내부에 들고 있는 callback 함수를 만들어 반환한다.
    """

    def enable_qat_callback(
        trainer,
    ):
        """
        trainer.model이 준비된 뒤 실행되는
        QAT 준비 callback.
        """

        model = trainer.model

        if model is None:
            raise RuntimeError(
                "[QAT] trainer.model이 "
                "아직 준비되지 않았습니다."
            )

        if getattr(
            model,
            "_qat_prepared",
            False,
        ):
            print(
                "[QAT] 이미 prepare_qat가 "
                "적용되어 있습니다."
            )

            return

        print("[QAT] prepare_qat 시작")

        model.train()

        if args.fuse:
            fuse_yolo_conv_bn_for_qat(
                model
            )

        else:
            print(
                "[QAT] --no-fuse 옵션으로 "
                "Conv+BN fuse 생략"
            )

        qconfig = choose_qat_qconfig(
            args.qat_backend
        )

        model.qconfig = qconfig

        insert_quant_dequant_boundaries_for_yolo_convs(
            model
        )

        disable_qat_for_unsupported_or_sensitive_layers(
            model
        )

        tq.prepare_qat(
            model,
            inplace=True,
        )

        model._qat_prepared = True
        model._qat_quantized_engine = (
            torch.backends.quantized.engine
        )

        trainer.ema = QATNoOpEMA(
            model
        )

        print(
            "[QAT] EMA 업데이트를 "
            "no-op으로 대체했습니다."
        )

        trainer.args.save = False

        def _skip_ultralytics_save_model(
            *save_args,
            **save_kwargs,
        ):
            print(
                "[QAT] Ultralytics 기본 "
                "checkpoint 저장을 건너뜁니다."
            )

            return False

        trainer.save_model = (
            _skip_ultralytics_save_model
        )

        print(
            "[QAT] Ultralytics 기본 "
            "checkpoint 저장을 비활성화했습니다."
        )

        print("[QAT] prepare_qat 완료")

        print(
            "[QAT] 이제부터 fake quant 상태로 "
            "fine-tuning 됩니다."
        )

    return enable_qat_callback


def qat_epoch_control_callback_factory(
    args,
):
    """
    epoch별로 observer와 BatchNorm을 제어하는
    callback을 만든다.
    """

    def qat_epoch_control_callback(
        trainer,
    ):
        model = trainer.model
        epoch = int(trainer.epoch)

        if epoch >= args.freeze_observer_epoch:
            model.apply(
                tq.disable_observer
            )

            if (
                epoch
                == args.freeze_observer_epoch
            ):
                print(
                    f"[QAT] epoch {epoch}: "
                    "observer 비활성화"
                )

        if epoch >= args.freeze_bn_epoch:
            for module in model.modules():
                if isinstance(
                    module,
                    nn.BatchNorm2d,
                ):
                    module.eval()

                if hasattr(
                    module,
                    "freeze_bn_stats",
                ):
                    try:
                        module.freeze_bn_stats()

                    except Exception:
                        pass

            if epoch == args.freeze_bn_epoch:
                print(
                    f"[QAT] epoch {epoch}: "
                    "BatchNorm 통계 고정"
                )

    return qat_epoch_control_callback


def save_qat_model_callback_factory(
    args,
):
    """
    학습 종료 후 QAT 모델을 따로 저장하는
    callback을 만든다.
    """

    def save_qat_model_callback(
        trainer,
    ):
        save_dir = Path(
            trainer.save_dir
        )

        weights_dir = (
            save_dir
            / "weights"
        )

        weights_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model = unwrap_parallel_model(
            trainer.model
        )

        model.eval()

        state_dict_path = (
            weights_dir
            / "qat_state_dict.pt"
        )

        torch.save(
            {
                "model_state_dict": (
                    model.state_dict()
                ),
                "qat_backend": (
                    args.qat_backend
                ),
                "quantized_engine": getattr(
                    model,
                    "_qat_quantized_engine",
                    torch.backends.quantized.engine,
                ),
                "dataset": args.dataset,
                "imgsz": args.imgsz,
                "fp32_start_weight": args.fp32,
                "note": (
                    "QAT fine-tuned fake-quant "
                    "model state_dict. "
                    "Use the same model structure "
                    "and QAT preparation steps "
                    "before loading this state_dict."
                ),
            },
            state_dict_path,
        )

        trainer._qat_state_dict_path = (
            state_dict_path
        )

        print(
            "[QAT] state_dict 저장 완료: "
            f"{state_dict_path}"
        )

    return save_qat_model_callback


# ------------------------------------------------------------
# 4. INT8 변환, export, 비교 관련 함수
# ------------------------------------------------------------

def unwrap_parallel_model(
    model: nn.Module,
) -> nn.Module:
    """
    DDP/DataParallel wrapper가 있으면
    실제 내부 모델을 반환한다.
    """

    if (
        hasattr(model, "module")
        and isinstance(
            model.module,
            nn.Module,
        )
    ):
        return model.module

    return model


def extract_first_tensor(
    output,
) -> torch.Tensor:
    """
    Tensor, tuple, list, dict 형태의 출력에서
    첫 번째 Tensor를 재귀적으로 찾는다.
    """

    if isinstance(
        output,
        torch.Tensor,
    ):
        return output

    if isinstance(
        output,
        (tuple, list),
    ):
        for item in output:
            try:
                return extract_first_tensor(
                    item
                )

            except TypeError:
                continue

    if isinstance(
        output,
        dict,
    ):
        for item in output.values():
            try:
                return extract_first_tensor(
                    item
                )

            except TypeError:
                continue

    raise TypeError(
        "모델 출력에서 export 가능한 "
        "Tensor를 찾지 못했습니다. "
        f"출력 타입: {type(output)}"
    )


def set_quantized_engine(
    preferred_engine: str,
) -> str:
    """
    현재 PyTorch 빌드에서 사용할 수 있는
    quantized engine을 선택한다.
    """

    supported = list(
        torch.backends.quantized.supported_engines
    )

    candidates = [
        preferred_engine
    ]

    if preferred_engine == "x86":
        candidates.extend(
            [
                "fbgemm",
                "onednn",
                "qnnpack",
            ]
        )

    elif preferred_engine == "fbgemm":
        candidates.extend(
            [
                "x86",
                "onednn",
                "qnnpack",
            ]
        )

    elif preferred_engine == "qnnpack":
        candidates.extend(
            [
                "x86",
                "fbgemm",
                "onednn",
            ]
        )

    for candidate in candidates:
        if candidate in supported:
            torch.backends.quantized.engine = (
                candidate
            )

            return candidate

    raise RuntimeError(
        "사용 가능한 quantized engine을 "
        "찾지 못했습니다. "
        f"요청={preferred_engine}, "
        f"지원={supported}"
    )


def make_dummy_input(
    imgsz: int,
) -> torch.Tensor:
    """
    변환 검증, export, latency 측정에 사용할
    CPU 더미 입력을 만든다.
    """

    return torch.randn(
        1,
        3,
        imgsz,
        imgsz,
        dtype=torch.float32,
        device="cpu",
    )


def clone_qat_model_for_fp32_latency(
    qat_model: nn.Module,
) -> nn.Module:
    """
    QAT 학습된 가중치는 유지하되
    fake quant를 끈 FP32 비교 모델을 만든다.
    """

    fp32_model = (
        copy.deepcopy(qat_model)
        .cpu()
        .eval()
    )

    fp32_model.apply(
        tq.disable_observer
    )

    fp32_model.apply(
        tq.disable_fake_quant
    )

    return fp32_model


def convert_qat_model_to_int8(
    qat_model: nn.Module,
    backend: str,
) -> nn.Module:
    """
    QAT 완료 모델을 CPU 실제 INT8 모델로 변환한다.
    """

    int8_model = (
        copy.deepcopy(qat_model)
        .cpu()
        .eval()
    )

    int8_model.apply(
        tq.disable_observer
    )

    actual_engine = set_quantized_engine(
        backend
    )

    print(
        "[CONVERT] quantized engine 설정: "
        f"{actual_engine}"
    )

    tq.convert(
        int8_model,
        inplace=True,
    )

    int8_model.eval()

    return int8_model


def measure_cpu_latency_ms(
    model: nn.Module,
    dummy_input: torch.Tensor,
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> float:
    """
    CPU에서 단일 입력의 평균 inference latency(ms)를
    측정한다.
    """

    model.eval()

    with torch.inference_mode():
        for _ in range(
            warmup_runs
        ):
            model(
                dummy_input
            )

        start_time = (
            time.perf_counter()
        )

        for _ in range(
            benchmark_runs
        ):
            model(
                dummy_input
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

    return (
        elapsed
        * 1000.0
        / benchmark_runs
    )


def export_int8_torchscript(
    int8_model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
) -> Path:
    """
    INT8 모델을 trace 기반 TorchScript로 저장하고
    재로딩 검증한다.
    """

    export_model = (
        TensorOnlyExportWrapper(
            int8_model
        )
        .cpu()
        .eval()
    )

    with torch.inference_mode():
        traced_model = torch.jit.trace(
            export_model,
            dummy_input,
            strict=False,
            check_trace=False,
        )

        try:
            traced_model = torch.jit.freeze(
                traced_model.eval()
            )

        except Exception as error:
            print(
                "[CONVERT] TorchScript freeze 생략: "
                f"{error}"
            )

        torch.jit.save(
            traced_model,
            str(output_path),
        )

        loaded_model = torch.jit.load(
            str(output_path),
            map_location="cpu",
        )

        loaded_model.eval()

        loaded_model(
            dummy_input
        )

    print(
        "[CONVERT] INT8 TorchScript 저장 완료: "
        f"{output_path}"
    )

    return output_path


def export_int8_onnx(
    int8_model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
) -> Path:
    """
    변환된 INT8 모델을 ONNX로 export한다.

    PyTorch 버전에 따라 torch.onnx.export()의
    dynamo 인자 유무가 다르므로 함수 signature를 확인한 뒤
    가능한 경우 legacy exporter를 사용한다.
    """

    export_model = (
        TensorOnlyExportWrapper(
            int8_model
        )
        .cpu()
        .eval()
    )

    export_kwargs = {
        "f": str(output_path),
        "opset_version": opset,
        "input_names": [
            "images"
        ],
        "output_names": [
            "predictions"
        ],
        "do_constant_folding": True,
        "dynamic_axes": {
            "images": {
                0: "batch"
            },
            "predictions": {
                0: "batch"
            },
        },
    }

    try:
        export_signature = inspect.signature(
            torch.onnx.export
        )

        if (
            "dynamo"
            in export_signature.parameters
        ):
            export_kwargs["dynamo"] = False

    except Exception:
        pass

    with torch.inference_mode():
        torch.onnx.export(
            export_model,
            dummy_input,
            **export_kwargs,
        )

    print(
        "[CONVERT] INT8 ONNX 저장 완료: "
        f"{output_path} "
        f"(opset={opset})"
    )

    return output_path


def get_file_size_bytes(
    path: Path,
) -> int:
    """
    파일 크기를 byte 단위로 반환한다.
    """

    if not path.exists():
        return 0

    return path.stat().st_size


def format_size_mb(
    size_bytes: int,
) -> str:
    """
    byte 크기를 MiB 문자열로 변환한다.
    """

    return (
        f"{size_bytes / (1024 ** 2):.2f} MiB"
    )


def print_size_comparison(
    qat_state_dict_path: Path,
    int8_torchscript_path: Path | None,
    int8_onnx_path: Path | None,
):
    """
    QAT state_dict와 INT8 산출물 파일 크기를
    비교해 출력한다.
    """

    qat_size = get_file_size_bytes(
        qat_state_dict_path
    )

    print(
        "[CONVERT] 파일 크기 비교"
    )

    print(
        "[CONVERT] QAT fake-quant state_dict: "
        f"{format_size_mb(qat_size)}"
    )

    if (
        int8_torchscript_path is not None
        and int8_torchscript_path.exists()
    ):
        int8_torchscript_size = (
            get_file_size_bytes(
                int8_torchscript_path
            )
        )

        print(
            "[CONVERT] INT8 TorchScript: "
            f"{format_size_mb(int8_torchscript_size)}"
        )

        if qat_size > 0:
            reduction = (
                1.0
                - int8_torchscript_size
                / qat_size
            ) * 100.0

            print(
                "[CONVERT] TorchScript 기준 "
                f"크기 변화: {reduction:+.2f}% 감소"
            )

    if (
        int8_onnx_path is not None
        and int8_onnx_path.exists()
    ):
        int8_onnx_size = get_file_size_bytes(
            int8_onnx_path
        )

        print(
            "[CONVERT] INT8 ONNX: "
            f"{format_size_mb(int8_onnx_size)}"
        )


def print_latency_comparison(
    fp32_latency_ms: float,
    int8_latency_ms: float,
):
    """
    FP32와 INT8 평균 latency 및 speedup을 출력한다.
    """

    print(
        "[CONVERT] FP32 CPU latency: "
        f"{fp32_latency_ms:.3f} ms/image"
    )

    print(
        "[CONVERT] INT8 CPU latency: "
        f"{int8_latency_ms:.3f} ms/image"
    )

    if int8_latency_ms > 0:
        speedup = (
            fp32_latency_ms
            / int8_latency_ms
        )

        print(
            f"[CONVERT] INT8 speedup: "
            f"{speedup:.3f}x"
        )


def extract_map_values(
    metrics,
) -> tuple[float, float]:
    """
    Ultralytics validation 결과에서
    mAP50과 mAP50-95를 꺼낸다.
    """

    box_metrics = getattr(
        metrics,
        "box",
        None,
    )

    if box_metrics is None:
        raise RuntimeError(
            "validation 결과에서 "
            "box metric을 찾지 못했습니다."
        )

    map50 = float(
        getattr(
            box_metrics,
            "map50",
        )
    )

    map50_95 = float(
        getattr(
            box_metrics,
            "map",
        )
    )

    return map50, map50_95


def evaluate_fp32_and_int8_map(
    args,
    data_yaml: Path,
    int8_model: nn.Module,
):
    """
    선택적으로 FP32 원본과 변환된 INT8 모델의
    mAP를 비교한다.

    FP32는 args.fp32의 원본 pretrained 모델을 사용한다.
    INT8은 현재 QAT 학습 후 convert된 nn.Module을
    CPU에서 검증한다.
    """

    print(
        "[CONVERT] FP32/INT8 mAP 비교 시작"
    )

    val_kwargs = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "plots": False,
        "save_json": False,
        "verbose": False,
    }

    fp32_result = None
    int8_result = None

    try:
        fp32_yolo = YOLO(
            args.fp32
        )

        fp32_metrics = fp32_yolo.val(
            device=args.device,
            **val_kwargs,
        )

        fp32_result = extract_map_values(
            fp32_metrics
        )

        print(
            "[CONVERT] FP32 원본 mAP: "
            f"mAP50={fp32_result[0]:.4f}, "
            f"mAP50-95={fp32_result[1]:.4f}"
        )

    except Exception as error:
        print(
            "[CONVERT] FP32 원본 mAP 평가 실패: "
            f"{type(error).__name__}: {error}"
        )

    try:
        int8_yolo = YOLO(
            args.fp32
        )

        int8_yolo.model = int8_model

        int8_metrics = int8_yolo.val(
            device="cpu",
            **val_kwargs,
        )

        int8_result = extract_map_values(
            int8_metrics
        )

        print(
            "[CONVERT] INT8 변환 모델 mAP: "
            f"mAP50={int8_result[0]:.4f}, "
            f"mAP50-95={int8_result[1]:.4f}"
        )

    except Exception as error:
        print(
            "[CONVERT] INT8 mAP 평가 실패: "
            f"{type(error).__name__}: {error}"
        )

    if (
        fp32_result is not None
        and int8_result is not None
    ):
        print(
            "[CONVERT] INT8 - FP32 mAP 차이: "
            f"mAP50="
            f"{int8_result[0] - fp32_result[0]:+.4f}, "
            f"mAP50-95="
            f"{int8_result[1] - fp32_result[1]:+.4f}"
        )


def convert_qat_to_int8_callback_factory(
    args,
):
    """
    학습 종료 후 QAT 모델을 실제 INT8로 변환하고
    export하는 callback을 만든다.
    """

    def convert_qat_to_int8_callback(
        trainer,
    ):
        if args.skip_int8_convert:
            print(
                "[CONVERT] --skip-int8-convert 지정: "
                "INT8 변환 생략"
            )

            return

        save_dir = Path(
            trainer.save_dir
        )

        weights_dir = (
            save_dir
            / "weights"
        )

        weights_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        qat_state_dict_path = Path(
            getattr(
                trainer,
                "_qat_state_dict_path",
                weights_dir
                / "qat_state_dict.pt",
            )
        )

        int8_torchscript_path = (
            weights_dir
            / "int8_model.pt"
        )

        int8_onnx_path = (
            weights_dir
            / "int8_model.onnx"
        )

        print(
            "[CONVERT] QAT -> INT8 변환 시작"
        )

        qat_model = unwrap_parallel_model(
            trainer.model
        )

        if not getattr(
            qat_model,
            "_qat_prepared",
            False,
        ):
            print(
                "[CONVERT] QAT prepare 상태를 찾지 못해 "
                "INT8 변환을 건너뜁니다."
            )

            return

        qat_model.eval()

        preferred_engine = getattr(
            qat_model,
            "_qat_quantized_engine",
            args.qat_backend,
        )

        try:
            fp32_latency_model = (
                clone_qat_model_for_fp32_latency(
                    qat_model
                )
            )

        except Exception as error:
            fp32_latency_model = None

            print(
                "[CONVERT] FP32 latency 비교 모델 "
                "복제 실패: "
                f"{type(error).__name__}: {error}"
            )

        try:
            int8_model = (
                convert_qat_model_to_int8(
                    qat_model=qat_model,
                    backend=preferred_engine,
                )
            )

        except Exception as error:
            print(
                "[CONVERT] "
                "torch.ao.quantization.convert() 실패: "
                f"{type(error).__name__}: {error}"
            )

            print(
                "[CONVERT] qat_state_dict.pt는 유지되며, "
                "INT8 산출물은 생성되지 않았습니다."
            )

            return

        dummy_input = make_dummy_input(
            args.imgsz
        )

        try:
            with torch.inference_mode():
                int8_output = int8_model(
                    dummy_input
                )

                primary_output = (
                    extract_first_tensor(
                        int8_output
                    )
                )

            print(
                "[CONVERT] INT8 forward 검증 완료: "
                f"output_shape="
                f"{tuple(primary_output.shape)}"
            )

        except Exception as error:
            print(
                "[CONVERT] INT8 forward 검증 실패: "
                f"{type(error).__name__}: {error}"
            )

            print(
                "[CONVERT] 모델 구조에 quantized backend가 "
                "지원하지 않는 연산이 남아 있을 수 있습니다."
            )

            return

        if fp32_latency_model is not None:
            try:
                fp32_latency_ms = (
                    measure_cpu_latency_ms(
                        fp32_latency_model,
                        dummy_input,
                    )
                )

                int8_latency_ms = (
                    measure_cpu_latency_ms(
                        int8_model,
                        dummy_input,
                    )
                )

                print_latency_comparison(
                    fp32_latency_ms=(
                        fp32_latency_ms
                    ),
                    int8_latency_ms=(
                        int8_latency_ms
                    ),
                )

            except Exception as error:
                print(
                    "[CONVERT] latency 비교 실패: "
                    f"{type(error).__name__}: {error}"
                )

        saved_torchscript_path = None
        saved_onnx_path = None

        # ----------------------------------------------------
        # TorchScript export
        # ----------------------------------------------------

        try:
            saved_torchscript_path = (
                export_int8_torchscript(
                    int8_model=int8_model,
                    dummy_input=dummy_input,
                    output_path=(
                        int8_torchscript_path
                    ),
                )
            )

        except Exception as error:
            if int8_torchscript_path.exists():
                int8_torchscript_path.unlink()

            print(
                "[CONVERT] INT8 TorchScript "
                "export 실패: "
                f"{type(error).__name__}: {error}"
            )

        # ----------------------------------------------------
        # ONNX export
        # ----------------------------------------------------

        try:
            saved_onnx_path = export_int8_onnx(
                int8_model=int8_model,
                dummy_input=dummy_input,
                output_path=int8_onnx_path,
                opset=args.onnx_opset,
            )

        except Exception as error:
            if int8_onnx_path.exists():
                int8_onnx_path.unlink()

            print(
                "[CONVERT] INT8 ONNX export 실패: "
                f"{type(error).__name__}: {error}"
            )

            print(
                "[CONVERT] PyTorch eager quantized 연산의 "
                "ONNX 지원 여부는 PyTorch/ONNX 버전과 "
                "연산 종류에 따라 달라질 수 있습니다."
            )

        print_size_comparison(
            qat_state_dict_path=(
                qat_state_dict_path
            ),
            int8_torchscript_path=(
                saved_torchscript_path
            ),
            int8_onnx_path=(
                saved_onnx_path
            ),
        )

        if args.eval_after_convert:
            data_yaml = Path(
                args._data_yaml
            )

            evaluate_fp32_and_int8_map(
                args=args,
                data_yaml=data_yaml,
                int8_model=int8_model,
            )

        print(
            "[CONVERT] QAT -> INT8 변환 단계 완료"
        )

    return convert_qat_to_int8_callback


# ------------------------------------------------------------
# 5. 메인 학습 함수
# ------------------------------------------------------------

def run_qat_training(
    args,
):
    """
    전체 QAT 학습 실행 함수.
    """

    data_yaml = prepare_dataset(
        args
    )

    args._data_yaml = str(
        data_yaml
    )

    if args.download_only:
        print(
            "[DONE] 다운로드만 수행하고 종료합니다."
        )

        print(
            f"[DONE] data yaml: {data_yaml}"
        )

        return

    print(
        f"[MODEL] FP32 pretrained 모델 로드: "
        f"{args.fp32}"
    )

    yolo = YOLO(
        args.fp32
    )

    yolo.add_callback(
        "on_pretrain_routine_end",
        enable_qat_callback_factory(
            args
        ),
    )

    yolo.add_callback(
        "on_train_epoch_start",
        qat_epoch_control_callback_factory(
            args
        ),
    )

    yolo.add_callback(
        "on_train_end",
        save_qat_model_callback_factory(
            args
        ),
    )

    yolo.add_callback(
        "on_train_end",
        convert_qat_to_int8_callback_factory(
            args
        ),
    )

    print(
        "[TRAIN] QAT fine-tuning 시작"
    )

    try:
        yolo.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,

            pretrained=True,

            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            weight_decay=args.weight_decay,
            cos_lr=args.cos_lr,

            amp=False,

            mosaic=args.mosaic,
            mixup=args.mixup,
            close_mosaic=args.close_mosaic,

            workers=args.workers,

            save=False,

            plots=True,
            val=True,
        )

    except FileNotFoundError as error:
        message = str(
            error
        )

        qat_paths = []

        trainer = getattr(
            yolo,
            "trainer",
            None,
        )

        if (
            trainer is not None
            and getattr(
                trainer,
                "save_dir",
                None,
            ) is not None
        ):
            direct_path = (
                Path(trainer.save_dir)
                / "weights"
                / "qat_state_dict.pt"
            )

            if direct_path.exists():
                qat_paths.append(
                    direct_path
                )

        qat_paths.extend(
            Path(".").glob(
                f"runs/**/{args.name}/weights/"
                "qat_state_dict.pt"
            )
        )

        qat_paths = sorted(
            set(qat_paths),
            key=lambda path: (
                path.stat().st_mtime
            ),
            reverse=True,
        )

        if (
            "Training completed but no checkpoint was saved"
            in message
            and qat_paths
        ):
            print(
                "[QAT] Ultralytics 기본 best.pt/last.pt가 "
                "없어 발생한 종료 후 확인 에러를 "
                "무시합니다."
            )

            print(
                "[QAT] QAT state_dict는 정상 저장되어 "
                f"있습니다: {qat_paths[0]}"
            )

        else:
            raise

    print(
        "[DONE] QAT fine-tuning 완료"
    )


# ------------------------------------------------------------
# 6. 옵션 파서
# ------------------------------------------------------------

def parse_args():
    """
    명령줄 옵션을 읽어 반환한다.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="coco128",
        choices=[
            "coco128",
            "coco",
        ],
        help=(
            "사용할 데이터셋. "
            "처음에는 coco128로 테스트 추천."
        ),
    )

    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=str(
            get_default_datasets_dir()
        ),
        help=(
            "COCO/COCO128을 저장하거나 "
            "불러올 상위 폴더."
        ),
    )

    parser.add_argument(
        "--yaml-dir",
        type=str,
        default="./generated_yamls",
        help=(
            "local data yaml을 저장할 폴더."
        ),
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
        help=(
            "데이터셋 다운로드와 yaml 생성만 하고 종료."
        ),
    )

    parser.add_argument(
        "--download-test",
        action="store_true",
        help=(
            "전체 COCO 사용 시 test2017도 다운로드."
        ),
    )

    parser.add_argument(
        "--fp32",
        type=str,
        default="yolo11n.pt",
        help=(
            "시작할 FP32 pretrained "
            "YOLO11 pt 파일."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs_qat",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolo11_qat_coco",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
    )

    parser.add_argument(
        "--lr0",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lrf",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--cos-lr",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--mixup",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--qat-backend",
        type=str,
        default="x86",
        choices=[
            "x86",
            "fbgemm",
            "qnnpack",
        ],
        help=(
            "QAT backend. "
            "PC/서버는 x86 또는 fbgemm, "
            "ARM은 qnnpack 권장."
        ),
    )

    parser.add_argument(
        "--freeze-observer-epoch",
        type=int,
        default=10,
        help=(
            "이 epoch부터 observer 업데이트를 멈춤."
        ),
    )

    parser.add_argument(
        "--freeze-bn-epoch",
        type=int,
        default=10,
        help=(
            "이 epoch부터 BatchNorm 통계를 고정."
        ),
    )

    parser.add_argument(
        "--no-fuse",
        dest="fuse",
        action="store_false",
        help=(
            "Conv+BN fuse를 하지 않음."
        ),
    )

    # --------------------------------------------------------
    # INT8 변환 및 저장 옵션
    # --------------------------------------------------------

    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=13,
        help=(
            "INT8 ONNX export에 사용할 "
            "opset 버전."
        ),
    )

    parser.add_argument(
        "--skip-int8-convert",
        action="store_true",
        help=(
            "QAT 학습과 qat_state_dict.pt 저장만 수행하고 "
            "INT8 convert/TorchScript/ONNX export는 생략."
        ),
    )

    parser.add_argument(
        "--eval-after-convert",
        action="store_true",
        help=(
            "변환 후 FP32 원본과 INT8 모델의 "
            "mAP를 비교."
        ),
    )

    parser.set_defaults(
        fuse=True
    )

    args = parser.parse_args()

    if args.onnx_opset < 11:
        parser.error(
            "--onnx-opset은 최소 11이어야 합니다."
        )

    if args.imgsz <= 0:
        parser.error(
            "--imgsz는 1 이상이어야 합니다."
        )

    return args


# ------------------------------------------------------------
# 7. 실행
# ------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    args = parse_args()

    run_qat_training(
        args
    )