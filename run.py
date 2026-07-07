# yolo11_coco_qat_full.py
# ------------------------------------------------------------
# 목적:
#   1. COCO128 또는 COCO 전체 데이터셋을 다운로드한다.
#   2. FP32 pretrained yolo11n.pt를 불러온다.
#   3. Ultralytics YOLO 학습 루프 내부에서 PyTorch QAT를 적용한다.
#   4. fake quant 상태로 QAT fine-tuning을 진행한다.
#   5. QAT 적용 모델을 별도로 저장한다.
#
# 설치:
#   pip install ultralytics torch torchvision
#
# 빠른 테스트:
#   python yolo11_coco_qat_full.py --dataset coco128 --epochs 5 --batch 4 --device 0
#
# 전체 COCO 사용:
#   python yolo11_coco_qat_full.py --dataset coco --epochs 10 --batch 16 --device 0
#
# 다운로드만:
#   python yolo11_coco_qat_full.py --dataset coco128 --download-only
#
# 주의:
#   이 코드는 "QAT fine-tuning" 코드다.
#   학습 중에는 fake quant로 INT8 양자화 오차를 흉내 낸다.
#   최종 배포용 완전한 INT8 engine은 TensorRT, Vitis AI, ONNX Runtime 등
#   target backend에서 별도 변환 과정이 필요할 수 있다.
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
    특정 폴더 안에 jpg 이미지가 있는지 확인한다.
    COCO가 이미 받아져 있는지 판단할 때 사용한다.
    """

    if not directory.exists():
        return False

    return any(directory.glob("*.jpg")) or any(directory.glob("*.jpeg")) or any(directory.glob("*.png"))


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

    COCO128은 COCO train2017 중 첫 128장만 있는 작은 테스트용 데이터셋이다.
    QAT 코드가 정상 작동하는지 확인할 때 먼저 쓰는 것을 추천한다.
    """

    coco128_dir = datasets_dir / "coco128"
    image_dir = coco128_dir / "images" / "train2017"

    if has_jpgs(image_dir):
        print(f"[DATA] COCO128 이미 존재: {coco128_dir}")
        return coco128_dir

    print("[DATA] COCO128 다운로드 시작")
    print(f"[DATA] 저장 위치: {datasets_dir}")

    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics assets에 있는 coco128.zip 사용
    url = ASSETS_URL + "/coco128.zip"

    # ultralytics download 함수는 zip 다운로드 후 압축해제를 처리한다.
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

    다운로드 구성:
        labels:
            coco2017labels.zip

        images:
            train2017.zip
            val2017.zip
            선택적으로 test2017.zip

    주의:
        전체 COCO는 용량이 크다.
        train2017 약 19GB, val2017 약 1GB, test2017 약 7GB 수준이다.
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

    # 1. YOLO 형식 COCO labels 다운로드
    if not label_ok:
        print("[DATA] COCO YOLO labels 다운로드")
        label_url = ASSETS_URL + "/coco2017labels.zip"
        download([label_url], dir=datasets_dir)

    # 2. COCO 이미지 다운로드
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

    # 최종 확인
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

    이렇게 하는 이유:
        그냥 data=coco.yaml을 쓰면 Ultralytics 기본 datasets_dir를 사용한다.
        여기서는 우리가 다운로드한 정확한 경로를 path로 박아 넣기 위해
        local yaml을 생성한다.
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

def choose_qat_qconfig(backend: str):
    """
    QAT backend에 맞는 qconfig를 고른다.

    backend 예시:
        x86
        fbgemm
        qnnpack

    일반 PC/서버:
        x86 또는 fbgemm

    ARM 계열:
        qnnpack
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
    Ultralytics YOLO의 기본 Conv 블록은 대체로 아래 형태다.

        Conv(
            conv = nn.Conv2d(...)
            bn   = nn.BatchNorm2d(...)
            act  = SiLU(...)
        )

    QAT에서는 Conv와 BN을 fuse하는 것이 일반적이다.
    여기서는 conv + bn만 fuse를 시도한다.

    SiLU는 ReLU처럼 단순 fuse 대상이 아니므로 그대로 둔다.
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

    제외 예시:
        - DFL layer
        - Upsample
        - 특수 출력 처리 모듈

    이유:
        YOLO detect head에는 DFL, decode 관련 처리가 섞여 있다.
        이런 부분까지 무리하게 quantization을 걸면 convert/export에서 문제가 생길 수 있다.
    """

    disabled = []

    for name, module in model.named_modules():
        lname = name.lower()

        should_disable = False

        # DFL은 bbox distribution을 거리값으로 바꾸는 특수 layer
        if "dfl" in lname:
            should_disable = True

        # Upsample은 quantized module로 바꿀 필요가 거의 없음
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

        핵심:
            trainer.model에 직접 prepare_qat를 적용한다.

        밖에서 YOLO(...).model에 적용하지 않는 이유:
            Ultralytics train() 내부에서 trainer.model을 따로 구성하거나
            checkpoint를 로드하는 과정이 있을 수 있기 때문이다.
        """

        model = trainer.model

        if model is None:
            raise RuntimeError("[QAT] trainer.model이 아직 준비되지 않았습니다.")

        if getattr(model, "_qat_prepared", False):
            print("[QAT] 이미 prepare_qat가 적용되어 있습니다.")
            return

        print("[QAT] prepare_qat 시작")

        model.train()

        # 1. Conv + BN fuse
        if args.fuse:
            fuse_yolo_conv_bn_for_qat(model)
        else:
            print("[QAT] --no-fuse 옵션으로 Conv+BN fuse 생략")

        # 2. qconfig 설정
        qconfig = choose_qat_qconfig(args.qat_backend)
        model.qconfig = qconfig

        # 3. 민감 layer 제외
        disable_qat_for_unsupported_or_sensitive_layers(model)

        # 4. 실제 QAT 준비
        #    이 과정에서 fake quant module / observer가 삽입된다.
        tq.prepare_qat(model, inplace=True)

        model._qat_prepared = True

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

        # observer 비활성화
        # observer는 activation/weight 범위를 관찰해서 scale, zero_point를 갱신한다.
        # 후반에는 고정시키는 것이 안정적일 때가 많다.
        if epoch >= args.freeze_observer_epoch:
            model.apply(tq.disable_observer)

            if epoch == args.freeze_observer_epoch:
                print(f"[QAT] epoch {epoch}: observer 비활성화")

        # BatchNorm 통계 고정
        if epoch >= args.freeze_bn_epoch:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

                # ConvBn QAT fused module은 freeze_bn_stats를 가질 수 있다.
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

        # 1. state_dict 저장
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
                    "This is not necessarily a fully converted deployment INT8 model."
                ),
            },
            state_dict_path,
        )

        # 2. 전체 모델 pickle 저장
        # 같은 ultralytics/torch 버전에서 다시 불러와 실험하기 편하다.
        # 단, 버전이 바뀌면 호환성이 떨어질 수 있다.
        full_model_path = weights_dir / "qat_full_model_pickle.pt"

        try:
            torch.save(model, full_model_path)
            print(f"[QAT] 전체 모델 저장 완료: {full_model_path}")
        except Exception as e:
            print(f"[QAT] 전체 모델 저장 실패: {e}")

        print(f"[QAT] state_dict 저장 완료: {state_dict_path}")

    return save_qat_model_callback


# ------------------------------------------------------------
# 4. 메인 학습 함수
# ------------------------------------------------------------

def run_qat_training(args):
    """
    전체 QAT 학습 실행 함수.
    """

    # 1. 데이터셋 준비
    data_yaml = prepare_dataset(args)

    if args.download_only:
        print("[DONE] 다운로드만 수행하고 종료합니다.")
        print(f"[DONE] data yaml: {data_yaml}")
        return

    # 2. FP32 pretrained YOLO11 로드
    #    yolo11n.pt가 없으면 Ultralytics가 자동 다운로드한다.
    print(f"[MODEL] FP32 pretrained 모델 로드: {args.fp32}")
    yolo = YOLO(args.fp32)

    # 3. QAT callback 등록
    #
    # on_pretrain_routine_end:
    #   데이터/모델 준비가 끝난 뒤 실행되는 지점이다.
    #
    # on_train_epoch_start:
    #   epoch마다 observer, BN 제어
    #
    # on_train_end:
    #   학습 종료 후 QAT 모델 따로 저장
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

    # 4. QAT fine-tuning 실행
    print("[TRAIN] QAT fine-tuning 시작")

    yolo.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,

        # pretrained weight에서 시작
        pretrained=True,

        # QAT는 보통 작은 learning rate로 fine-tuning
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        cos_lr=args.cos_lr,

        # QAT + AMP는 충돌하거나 불안정할 수 있으므로 기본적으로 끔
        amp=False,

        # augmentation
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,

        # 기타
        workers=args.workers,
        save=True,
        plots=True,
        val=True,
    )

    print("[DONE] QAT fine-tuning 완료")


# ------------------------------------------------------------
# 5. 옵션 파서
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # 데이터셋 관련
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

    # 모델 관련
    parser.add_argument(
        "--fp32",
        type=str,
        default="yolo11n.pt",
        help="시작할 FP32 pretrained YOLO11 pt 파일.",
    )

    # 학습 관련
    parser.add_argument("--epochs", type=int, default=5)
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

    # augmentation
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=3)

    # QAT 관련
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
        default=3,
        help="이 epoch부터 observer 업데이트를 멈춤.",
    )

    parser.add_argument(
        "--freeze-bn-epoch",
        type=int,
        default=3,
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