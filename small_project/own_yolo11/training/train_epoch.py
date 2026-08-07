# --------------------------------------------------
# Train one epoch
# --------------------------------------------------

def train_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    """
    학습 데이터 전체를 한 번 순회한다.

    한 batch의 동작 순서:

        images, targets
            ↓
        model(images)
            ↓
        predictions
            ↓
        criterion(predictions, targets)
            ↓
        total_loss
            ↓
        backward()
            ↓
        optimizer.step()

    Args:
        model:
            학습할 YOLO 모델

        data_loader:
            학습용 DataLoader

        criterion:
            YOLO11DetectionLoss

        optimizer:
            모델 Parameter를 업데이트할 Optimizer

        device:
            cpu 또는 cuda

    Returns:
        epoch_loss:
            한 epoch 동안의 평균 Loss

            {
                "box_loss": ...,
                "cls_loss": ...,
                "dfl_loss": ...,
                "total_loss": ...
            }
    """

    # --------------------------------------------------
    # Model을 Training mode로 변경
    # --------------------------------------------------

    model.train()


    # --------------------------------------------------
    # Loss 누적 변수
    # --------------------------------------------------

    running_box_loss = 0.0
    running_cls_loss = 0.0
    running_dfl_loss = 0.0
    running_total_loss = 0.0

    batch_count = 0


    # --------------------------------------------------
    # DataLoader 순회
    # --------------------------------------------------

    for images, targets in data_loader:

        # --------------------------------------------------
        # 이미지를 device로 이동
        # --------------------------------------------------

        # images:
        #
        # [B, 3, 640, 640]
        images = images.to(
            device=device,
            non_blocking=True,
        )


        # --------------------------------------------------
        # 이전 Gradient 초기화
        # --------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True,
        )


        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        predictions = model(images)

        # predictions:
        #
        # {
        #     "box_logits":
        #         [B, 4 * reg_max, N],
        #
        #     "class_logits":
        #         [B, num_classes, N],
        #
        #     "features":
        #         [P3, P4, P5]
        # }


        # --------------------------------------------------
        # Loss 계산
        # --------------------------------------------------

        total_loss, loss_items = criterion(
            predictions,
            targets,
        )


        # --------------------------------------------------
        # Backward
        # --------------------------------------------------

        # Loss를 기준으로
        # 각 Parameter의 Gradient를 계산한다.
        total_loss.backward()


        # --------------------------------------------------
        # Parameter Update
        # --------------------------------------------------

        # 계산된 Gradient를 사용하여
        # 모델 Parameter를 업데이트한다.
        optimizer.step()


        # --------------------------------------------------
        # Loss 누적
        # --------------------------------------------------

        running_box_loss += (
            loss_items["box_loss"].item()
        )

        running_cls_loss += (
            loss_items["cls_loss"].item()
        )

        running_dfl_loss += (
            loss_items["dfl_loss"].item()
        )

        running_total_loss += (
            loss_items["total_loss"].item()
        )

        batch_count += 1


    # --------------------------------------------------
    # Epoch 평균 Loss
    # --------------------------------------------------

    epoch_loss = {
        "box_loss": (
            running_box_loss
            / batch_count
        ),

        "cls_loss": (
            running_cls_loss
            / batch_count
        ),

        "dfl_loss": (
            running_dfl_loss
            / batch_count
        ),

        "total_loss": (
            running_total_loss
            / batch_count
        ),
    }


    # --------------------------------------------------
    # Epoch Loss 반환
    # --------------------------------------------------

    return epoch_loss