# yolo11_coco_qat_full.py
# ------------------------------------------------------------
# 목적:
#   1. COCO128 또는 COCO 전체 데이터셋을 다운로드한다.
#   2. FP32 pretrained yolo11n.pt를 불러온다.
#   3. Ultralytics YOLO 학습 루프 내부에서 PyTorch QAT를 적용한다.
#   4. fake quant 상태로 QAT fine-tuning을 진행한다.
#   5. QAT 적용 모델을 state_dict 형태로 별도 저장한다.
# ------------------------------------------------------------

import argparse
from pathlib import Path
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


def write_local_coco_yaml(dataset_name: str, dataset_dir: Path, output_dir: Path) -> Path:
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
        raise ValueError("dataset_name은 coco128 또는 coco여야 합니다.")

    lines = []
    lines.append(f"path: {dataset_dir.as_posix()}")
    lines.append(f"train: {train_path}")
    lines.append(f"val: {val_path}")
    lines.append("")
    lines.append("nc: 80")
    lines.append("")
    lines.append("names:")

    for i, name in enumerate(COCO_NAMES):
        lines.append(f"  {i}: {name}")

    yaml_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[DATA] local yaml 생성 완료: {yaml_path}")
    return yaml_path


def prepare_dataset(args) -> Path:
    """
    args.dataset에 따라 COCO128 또는 COCO 전체를 준비하고,
    local yaml 파일 경로를 반환한다.
    """

    datasets_dir = Path(args.datasets_dir).expanduser().resolve()

    if args.dataset == "coco128":
        dataset_dir = ensure_coco128(datasets_dir)

    elif args.dataset == "coco":
        dataset_dir = ensure_coco_full(
            datasets_dir=datasets_dir,
            download_test=args.download_test,
        )

    else:
        raise ValueError("--dataset은 coco128 또는 coco만 지원합니다.")

    yaml_path = write_local_coco_yaml(
        dataset_name=args.dataset,
        dataset_dir=dataset_dir,
        output_dir=Path(args.yaml_dir).resolve(),
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

    def __init__(self, model: nn.Module):
        self.ema = model
        self.updates = 0

    def update(self, model: nn.Module):
        return

    def update_attr(self, model: nn.Module, include=(), exclude=()):
        for k in include:
            if hasattr(model, k):
                try:
                    setattr(self.ema, k, getattr(model, k))
                except Exception:
                    pass
        return


def choose_qat_qconfig(backend: str):
    """
    QAT backend에 맞는 qconfig를 고른다.
    """

    try:
        torch.backends.quantized.engine = backend
    except Exception:
        pass

    try:
        qconfig = tq.get_default_qat_qconfig(backend)
        print(f"[QAT] backend={backend} qconfig 사용")
        return qconfig

    except Exception as e:
        print(f"[QAT] backend={backend} qconfig 생성 실패: {e}")

        fallback = "fbgemm" if backend in ["x86", "fbgemm"] else "qnnpack"

        try:
            torch.backends.quantized.engine = fallback
        except Exception:
            pass

        qconfig = tq.get_default_qat_qconfig(fallback)
        print(f"[QAT] fallback backend={fallback} qconfig 사용")
        return qconfig


def fuse_yolo_conv_bn_for_qat(model: nn.Module):
    """
    Ultralytics YOLO의 Conv 블록에서 conv + bn fuse를 시도한다.
    """

    fused_count = 0
    failed_count = 0

    for name, module in model.named_modules():
        has_conv_bn = (
            hasattr(module, "conv")
            and hasattr(module, "bn")
            and isinstance(module.conv, nn.Conv2d)
            and isinstance(module.bn, nn.BatchNorm2d)
        )

        if not has_conv_bn:
            continue

        try:
            tq.fuse_modules_qat(
                module,
                ["conv", "bn"],
                inplace=True,
            )
            fused_count += 1

        except Exception:
            failed_count += 1

    print(f"[QAT] Conv+BN fuse 성공: {fused_count}")
    print(f"[QAT] Conv+BN fuse 실패 또는 skip: {failed_count}")


def disable_qat_for_unsupported_or_sensitive_layers(model: nn.Module):
    """
    일부 layer는 QAT 대상에서 제외한다.
    """

    disabled = []

    for name, module in model.named_modules():
        lname = name.lower()

        should_disable = False

        if "dfl" in lname:
            should_disable = True

        if isinstance(module, nn.Upsample):
            should_disable = True

        if should_disable:
            module.qconfig = None
            disabled.append(name)

    if disabled:
        print("[QAT] QAT 제외 모듈:")
        for name in disabled:
            print(f"  - {name}")


def enable_qat_callback_factory(args):
    """
    Ultralytics callback은 함수만 받을 수 있으므로,
    args를 내부에 들고 있는 callback 함수를 만들어 반환한다.
    """

    def enable_qat_callback(trainer):
        """
        trainer.model이 준비된 뒤 실행되는 QAT 준비 callback.
        """

        model = trainer.model

        if model is None:
            raise RuntimeError("[QAT] trainer.model이 아직 준비되지 않았습니다.")

        if getattr(model, "_qat_prepared", False):
            print("[QAT] 이미 prepare_qat가 적용되어 있습니다.")
            return

        print("[QAT] prepare_qat 시작")

        model.train()

        if args.fuse:
            fuse_yolo_conv_bn_for_qat(model)
        else:
            print("[QAT] --no-fuse 옵션으로 Conv+BN fuse 생략")

        qconfig = choose_qat_qconfig(args.qat_backend)
        model.qconfig = qconfig

        disable_qat_for_unsupported_or_sensitive_layers(model)

        tq.prepare_qat(model, inplace=True)

        model._qat_prepared = True

        trainer.ema = QATNoOpEMA(model)
        print("[QAT] EMA 업데이트를 no-op으로 대체했습니다.")

        trainer.args.save = False

        def _skip_ultralytics_save_model(*save_args, **save_kwargs):
            print("[QAT] Ultralytics 기본 checkpoint 저장을 건너뜁니다.")
            return False

        trainer.save_model = _skip_ultralytics_save_model
        print("[QAT] Ultralytics 기본 checkpoint 저장을 비활성화했습니다.")

        print("[QAT] prepare_qat 완료")
        print("[QAT] 이제부터 fake quant 상태로 fine-tuning 됩니다.")

    return enable_qat_callback


def qat_epoch_control_callback_factory(args):
    """
    epoch별로 observer와 BatchNorm을 제어하는 callback을 만든다.
    """

    def qat_epoch_control_callback(trainer):
        model = trainer.model
        epoch = int(trainer.epoch)

        if epoch >= args.freeze_observer_epoch:
            model.apply(tq.disable_observer)

            if epoch == args.freeze_observer_epoch:
                print(f"[QAT] epoch {epoch}: observer 비활성화")

        if epoch >= args.freeze_bn_epoch:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

                if hasattr(module, "freeze_bn_stats"):
                    try:
                        module.freeze_bn_stats()
                    except Exception:
                        pass

            if epoch == args.freeze_bn_epoch:
                print(f"[QAT] epoch {epoch}: BatchNorm 통계 고정")

    return qat_epoch_control_callback


def save_qat_model_callback_factory(args):
    """
    학습 종료 후 QAT 모델을 따로 저장하는 callback을 만든다.
    """

    def save_qat_model_callback(trainer):
        save_dir = Path(trainer.save_dir)
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        model = trainer.model
        model.eval()

        state_dict_path = weights_dir / "qat_state_dict.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "qat_backend": args.qat_backend,
                "dataset": args.dataset,
                "imgsz": args.imgsz,
                "fp32_start_weight": args.fp32,
                "note": (
                    "QAT fine-tuned fake-quant model state_dict. "
                    "This is not a fully converted deployment INT8 model. "
                    "For deployment, convert/export with the target backend."
                ),
            },
            state_dict_path,
        )

        print(f"[QAT] state_dict 저장 완료: {state_dict_path}")

    return save_qat_model_callback


# ------------------------------------------------------------
# 4. 메인 학습 함수
# ------------------------------------------------------------

def run_qat_training(args):
    """
    전체 QAT 학습 실행 함수.
    """

    data_yaml = prepare_dataset(args)

    if args.download_only:
        print("[DONE] 다운로드만 수행하고 종료합니다.")
        print(f"[DONE] data yaml: {data_yaml}")
        return

    print(f"[MODEL] FP32 pretrained 모델 로드: {args.fp32}")
    yolo = YOLO(args.fp32)

    yolo.add_callback(
        "on_pretrain_routine_end",
        enable_qat_callback_factory(args),
    )

    yolo.add_callback(
        "on_train_epoch_start",
        qat_epoch_control_callback_factory(args),
    )

    yolo.add_callback(
        "on_train_end",
        save_qat_model_callback_factory(args),
    )

    print("[TRAIN] QAT fine-tuning 시작")

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

            # QAT 모델은 Ultralytics 기본 checkpoint 저장에서
            # pickle 불가능한 quantization 객체 때문에 실패할 수 있으므로 끈다.
            # QAT 모델은 on_train_end callback에서 state_dict로 따로 저장한다.
            save=False,

            plots=True,
            val=True,
        )

    except FileNotFoundError as e:
        msg = str(e)

        qat_paths = sorted(
            Path(".").glob(f"runs/**/{args.name}/weights/qat_state_dict.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if "Training completed but no checkpoint was saved" in msg and qat_paths:
            print(
                "[QAT] Ultralytics 기본 best.pt/last.pt가 없어 발생한 "
                "종료 후 확인 에러를 무시합니다."
            )
            print(f"[QAT] QAT state_dict는 정상 저장되어 있습니다: {qat_paths[0]}")
        else:
            raise

    print("[DONE] QAT fine-tuning 완료")


# ------------------------------------------------------------
# 5. 옵션 파서
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="coco128",
        choices=["coco128", "coco"],
        help="사용할 데이터셋. 처음에는 coco128로 테스트 추천.",
    )

    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=str(get_default_datasets_dir()),
        help="COCO/COCO128을 저장할 폴더.",
    )

    parser.add_argument(
        "--yaml-dir",
        type=str,
        default="./generated_yamls",
        help="local data yaml을 저장할 폴더.",
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
        help="데이터셋 다운로드와 yaml 생성만 하고 종료.",
    )

    parser.add_argument(
        "--download-test",
        action="store_true",
        help="전체 COCO 사용 시 test2017도 다운로드.",
    )

    parser.add_argument(
        "--fp32",
        type=str,
        default="yolo11n.pt",
        help="시작할 FP32 pretrained YOLO11 pt 파일.",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--project", type=str, default="runs_qat")
    parser.add_argument("--name", type=str, default="yolo11_qat_coco")

    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--cos-lr", action="store_true", default=True)

    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=3)

    parser.add_argument(
        "--qat-backend",
        type=str,
        default="x86",
        choices=["x86", "fbgemm", "qnnpack"],
        help="QAT backend. PC/서버는 x86 또는 fbgemm, ARM은 qnnpack 권장.",
    )

    parser.add_argument(
        "--freeze-observer-epoch",
        type=int,
        default=10,
        help="이 epoch부터 observer 업데이트를 멈춤.",
    )

    parser.add_argument(
        "--freeze-bn-epoch",
        type=int,
        default=10,
        help="이 epoch부터 BatchNorm 통계를 고정.",
    )

    parser.add_argument(
        "--no-fuse",
        dest="fuse",
        action="store_false",
        help="Conv+BN fuse를 하지 않음.",
    )

    parser.set_defaults(fuse=True)

    return parser.parse_args()


# ------------------------------------------------------------
# 6. 실행
# ------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()
    run_qat_training(args)