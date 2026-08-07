# --------------------------------------------------
# Import library
# --------------------------------------------------

import torch


# --------------------------------------------------
# Validate one epoch
# --------------------------------------------------

def validate_epoch(
    model,
    data_loader,
    criterion,
    device,
):
    """
    Validation 데이터 전체를 한 번 순회하여
    Loss를 계산한다.

    Training과 달리:

        backward() 하지 않음
        optimizer.step() 하지 않음

    따라서 모델 Parameter는 변경되지 않는다.

    Args:
        model:
            검증할 YOLO 모델

        data_loader:
            Validation DataLoader

        criterion:
            YOLO11DetectionLoss

        device:
            cpu 또는 cuda

    Returns:
        epoch_loss:
            한 epoch 동안의 평균 Validation Loss

            {
                "box_loss": ...,
                "cls_loss": ...,
                "dfl_loss": ...,
                "total_loss": ...
            }
    """

    # --------------------------------------------------
    # Model을 Evaluation mode로 변경
    # --------------------------------------------------

    model.eval()


    # --------------------------------------------------
    # Loss 누적 변수
    # --------------------------------------------------

    running_box_loss = 0.0
    running_cls_loss = 0.0
    running_dfl_loss = 0.0
    running_total_loss = 0.0

    batch_count = 0


    # --------------------------------------------------
    # Gradient 계산 비활성화
    # --------------------------------------------------

    with torch.no_grad():

        # --------------------------------------------------
        # DataLoader 순회
        # --------------------------------------------------

        for images, targets in data_loader:

            # --------------------------------------------------
            # 이미지를 device로 이동
            # --------------------------------------------------

            images = images.to(
                device=device,
                non_blocking=True,
            )


            # --------------------------------------------------
            # Forward
            # --------------------------------------------------

            predictions = model(images)


            # --------------------------------------------------
            # Loss 계산
            # --------------------------------------------------

            total_loss, loss_items = criterion(
                predictions,
                targets,
            )


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