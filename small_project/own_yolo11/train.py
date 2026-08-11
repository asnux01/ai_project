# 라이브러리
import os

import torch
from torch.utils.data import DataLoader

from data import Coco2017Dataset, detection_collate_fn
from loss import YOLO11DetectionLoss
from model import Yolov11
from training import build_optimizer, train_epoch, validate_epoch


def build_model_config(
    num_classes,
    scale,
    reg_max,
    strides,
    image_size,
):
    """학습과 동일한 모델을 추론에서 재생성하는 데 필요한 설정을 만든다."""

    return {
        "num_classes": int(num_classes),
        "scale": str(scale),
        "reg_max": int(reg_max),
        "strides": tuple(strides),
        "image_size": int(image_size),
    }


def build_checkpoint(
    epoch,
    model,
    optimizer,
    model_config,
    train_loss=None,
    val_loss=None,
):
    """모델 설정과 상태를 하나의 checkpoint 딕셔너리로 묶는다."""

    checkpoint = {
        # 학습 재개 시 다음 epoch를 결정하기 위한 현재 epoch
        "epoch": int(epoch),

        # 추론 시 같은 scale과 Head 설정으로
        # 모델을 다시 만들기 위한 값
        "model_config": dict(model_config),

        # 실제 학습된 모델 파라미터
        "model_state_dict": model.state_dict(),

        # 학습 재개 시 momentum 등을 복원하기 위한
        # optimizer 상태
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # best checkpoint에는 비교에 사용한 loss도 함께 기록
    if train_loss is not None:
        checkpoint["train_loss"] = train_loss

    if val_loss is not None:
        checkpoint["val_loss"] = val_loss

    return checkpoint


def main():
    """데이터, 모델, loss, optimizer를 만들고 전체 학습을 실행한다."""

    # CUDA가 사용 가능하면 GPU를,
    # 그렇지 않으면 CPU를 선택한다.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # --------------------------------------------------
    # 데이터 및 학습 하이퍼파라미터
    # --------------------------------------------------

    image_size = 640
    batch_size = 8
    epochs = 100
    learning_rate = 0.001
    weight_decay = 0.0005

    # --------------------------------------------------
    # 모델 구조 관련 설정
    # --------------------------------------------------

    num_classes = 80
    reg_max = 16
    strides = (8, 16, 32)

    # depth와 width 값을 직접 입력하지 않고
    # n, s, m, l, x 중 하나의 스케일만 선택한다.
    scale = "n"

    # best.pt와 last.pt를 저장할 폴더를 준비한다.
    save_dir = "checkpoints"
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    # 학습용 COCO2017 이미지와 annotation을 연결한다.
    train_dataset = Coco2017Dataset(
        image_dir=(
            "datasets/coco/images/train2017"
        ),
        annotation_file=(
            "datasets/coco/annotations/"
            "instances_train2017.json"
        ),
        image_size=image_size,
    )

    # 검증용 COCO2017 이미지와 annotation을 연결한다.
    val_dataset = Coco2017Dataset(
        image_dir=(
            "datasets/coco/images/val2017"
        ),
        annotation_file=(
            "datasets/coco/annotations/"
            "instances_val2017.json"
        ),
        image_size=image_size,
    )

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    # 이미지마다 annotation 개수가 다르므로
    # 전용 collate 함수를 사용한다.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,

        # 먼저 동작을 확인하기 쉽도록 worker를 0으로 둔다.
        num_workers=0,

        # CUDA 사용 시 page-locked memory를 사용해
        # 전송 비용을 줄일 수 있다.
        pin_memory=torch.cuda.is_available(),

        collate_fn=detection_collate_fn,
        drop_last=False,
    )

    # 검증 데이터의 순서는 학습에 영향을 주지 않으므로
    # 데이터를 섞지 않는다.
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
        drop_last=False,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    # 선택한 scale 이름을 모델에 전달하면
    # 모델 내부에서 해당 스케일 계수를 조회한다.
    model = Yolov11(
        num_classes=num_classes,
        scale=scale,
        reg_max=reg_max,
        strides=strides,
    )

    model = model.to(device)

    # 학습과 추론이 동일한 구조를 사용하도록
    # checkpoint에 함께 저장할 설정을 만든다.
    model_config = build_model_config(
        num_classes=num_classes,
        scale=model.scale,
        reg_max=reg_max,
        strides=strides,
        image_size=image_size,
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    # YOLO11의 box, class, DFL loss를
    # 계산하는 객체를 생성한다.
    criterion = YOLO11DetectionLoss(
        num_classes=num_classes,
        reg_max=reg_max,
        strides=strides,
        box_gain=7.5,
        cls_gain=0.5,
        dfl_gain=1.5,
        tal_topk=10,
    )

    criterion = criterion.to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    # 학습 가능한 모델 파라미터를
    # AdamW optimizer에 등록한다.
    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # 검증 total loss가 가장 작은 checkpoint를 선택한다.
    best_val_loss = float("inf")

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(epochs):
        print()
        print(
            f"========== Epoch "
            f"{epoch + 1}/{epochs} =========="
        )

        # 한 번의 학습 epoch에서 forward, backward,
        # parameter update를 수행한다.
        train_loss = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # 같은 epoch의 검증 loss를
        # gradient 갱신 없이 계산한다.
        val_loss = validate_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # 네 종류의 학습 및 검증 loss를
        # 사람이 읽기 쉬운 형태로 출력한다.
        print()

        print(
            "Train | "
            f"Box: {train_loss['box_loss']:.4f} | "
            f"Cls: {train_loss['cls_loss']:.4f} | "
            f"DFL: {train_loss['dfl_loss']:.4f} | "
            f"Total: {train_loss['total_loss']:.4f}"
        )

        print(
            "Val   | "
            f"Box: {val_loss['box_loss']:.4f} | "
            f"Cls: {val_loss['cls_loss']:.4f} | "
            f"DFL: {val_loss['dfl_loss']:.4f} | "
            f"Total: {val_loss['total_loss']:.4f}"
        )

        # 이전 best보다 검증 total loss가 작아졌을 때만
        # best.pt를 갱신한다.
        if val_loss["total_loss"] < best_val_loss:
            best_val_loss = val_loss["total_loss"]

            checkpoint = build_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            torch.save(
                checkpoint,
                os.path.join(
                    save_dir,
                    "best.pt",
                ),
            )

            print("Best model saved.")

    # 최종 epoch의 상태는 성능과 관계없이
    # last.pt로 저장한다.
    last_checkpoint = build_checkpoint(
        epoch=epochs,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
    )

    torch.save(
        last_checkpoint,
        os.path.join(
            save_dir,
            "last.pt",
        ),
    )

    print()
    print("Training completed.")


# 이 파일을 직접 실행했을 때만 학습을 시작한다.
if __name__ == "__main__":
    main()