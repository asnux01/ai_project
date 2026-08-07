# --------------------------------------------------
# Import library
# --------------------------------------------------

import os

import torch

from torch.utils.data import DataLoader


# --------------------------------------------------
# Import Dataset
# --------------------------------------------------

from .data import (
    Coco2017Dataset,
    detection_collate_fn,
)


# --------------------------------------------------
# Import Model
# --------------------------------------------------

# 실제 네 프로젝트 구조에 맞게 수정
from .model import Yolov11


# --------------------------------------------------
# Import Loss
# --------------------------------------------------

from .loss import YOLO11DetectionLoss


# --------------------------------------------------
# Import Training modules
# --------------------------------------------------

from .training import *


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )


    # --------------------------------------------------
    # Training Parameters
    # --------------------------------------------------

    image_size = 640

    batch_size = 8

    epochs = 100

    learning_rate = 0.001

    weight_decay = 0.0005

    num_classes = 80

    reg_max = 16

    strides = (
        8,
        16,
        32,
    )


    # --------------------------------------------------
    # Checkpoint Directory
    # --------------------------------------------------

    save_dir = "checkpoints"

    os.makedirs(
        save_dir,
        exist_ok=True,
    )


    # --------------------------------------------------
    # Train Dataset
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Validation Dataset
    # --------------------------------------------------

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
    # Train DataLoader
    # --------------------------------------------------

    train_loader = DataLoader(
        dataset=train_dataset,

        # 한 번에 학습할 이미지 개수
        batch_size=batch_size,

        # 학습 데이터 순서를 섞는다.
        shuffle=True,

        # 처음에는 오류 확인이 쉬운 0 사용
        num_workers=0,

        # GPU 전송을 도울 수 있다.
        pin_memory=torch.cuda.is_available(),

        # 객체 탐지용 collate 함수
        collate_fn=detection_collate_fn,

        # 마지막 batch도 사용한다.
        drop_last=False,
    )


    # --------------------------------------------------
    # Validation DataLoader
    # --------------------------------------------------

    val_loader = DataLoader(
        dataset=val_dataset,

        batch_size=batch_size,

        # Validation 데이터는
        # 순서를 섞을 필요가 없다.
        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

        collate_fn=detection_collate_fn,

        drop_last=False,
    )


    # --------------------------------------------------
    # Model 생성
    # --------------------------------------------------

    model = Yolov11(
        num_classes=num_classes,
        reg_max=reg_max,
    )

    # 모델을 GPU 또는 CPU로 이동한다.
    model = model.to(device)


    # --------------------------------------------------
    # Loss 생성
    # --------------------------------------------------

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
    # Optimizer 생성
    # --------------------------------------------------

    optimizer = build_optimizer(
        model=model,

        learning_rate=learning_rate,

        weight_decay=weight_decay,
    )


    # --------------------------------------------------
    # Best Validation Loss
    # --------------------------------------------------

    best_val_loss = float("inf")


    # --------------------------------------------------
    # Epoch 반복
    # --------------------------------------------------

    for epoch in range(epochs):

        print()
        print(
            f"========== "
            f"Epoch {epoch + 1}/{epochs} "
            f"=========="
        )


        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        train_loss = train_epoch(
            model=model,

            data_loader=train_loader,

            criterion=criterion,

            optimizer=optimizer,

            device=device,
        )


        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_loss = validate_epoch(
            model=model,

            data_loader=val_loader,

            criterion=criterion,

            device=device,
        )


        # --------------------------------------------------
        # Epoch 결과 출력
        # --------------------------------------------------

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


        # --------------------------------------------------
        # Best Model 저장
        # --------------------------------------------------

        if (
            val_loss["total_loss"]
            < best_val_loss
        ):

            best_val_loss = (
                val_loss["total_loss"]
            )

            checkpoint = {
                # 현재 Epoch
                "epoch":
                    epoch + 1,

                # Model Parameter
                "model_state_dict":
                    model.state_dict(),

                # Optimizer 상태
                "optimizer_state_dict":
                    optimizer.state_dict(),

                # Training Loss
                "train_loss":
                    train_loss,

                # Validation Loss
                "val_loss":
                    val_loss,
            }

            torch.save(
                checkpoint,

                os.path.join(
                    save_dir,
                    "best.pt",
                ),
            )

            print(
                "Best model saved."
            )


    # --------------------------------------------------
    # Last Model 저장
    # --------------------------------------------------

    last_checkpoint = {
        "epoch":
            epochs,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),
    }

    torch.save(
        last_checkpoint,

        os.path.join(
            save_dir,
            "last.pt",
        ),
    )


    # --------------------------------------------------
    # Training 종료
    # --------------------------------------------------

    print()
    print("Training completed.")