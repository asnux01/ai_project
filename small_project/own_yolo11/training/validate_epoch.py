"""Gradient 갱신 없이 validation loss와 선택적인 mAP를 계산한다."""

# inference_mode와 AMP autocast를 사용한다.
import torch


def validate_epoch(
    model,
    data_loader,
    criterion,
    device,
    use_amp=True,
    metric=None,
    logger=None,
):
    """
    Validation 데이터 전체를 한 번 순회한다.

    Training과 달리:

        backward() 하지 않음
        optimizer.step() 하지 않음

    따라서 모델 Parameter는 변경되지 않는다.

    Args:
        model:
            검증할 YOLO 또는 EMA 모델

        data_loader:
            Validation DataLoader

        criterion:
            YOLO11DetectionLoss

        device:
            cpu 또는 cuda 장치

        use_amp:
            CUDA에서 AMP 검증을 사용할지 결정

        metric:
            선택적인 DetectionMAP 객체

            None이면 validation loss만 계산

        logger:
            검증 결과를 기록할 logger

    Returns:
        epoch_result:
            평균 validation loss와
            선택적인 mAP 결과
    """

    # 문자열 device도 처리할 수 있도록
    # torch.device로 통일한다.
    device = torch.device(
        device
    )

    # CUDA가 아닌 환경에서는
    # use_amp=True여도 자동으로 AMP를 끈다.
    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    # BatchNorm과 Dropout 등을
    # 검증 모드로 전환한다.
    model.eval()

    # 이전 epoch에서 누적된 예측과 정답이
    # 이번 epoch에 섞이지 않도록 초기화한다.
    if metric is not None:
        metric.reset()

    # --------------------------------------------------
    # Loss 누적 변수
    # --------------------------------------------------

    running_box_loss = 0.0
    running_cls_loss = 0.0
    running_dfl_loss = 0.0
    running_total_loss = 0.0

    batch_count = 0

    # no_grad보다 더 강하게
    # autograd 관련 상태를 비활성화한다.
    with torch.inference_mode():

        # Validation DataLoader 순회
        for (
            images,
            targets,
        ) in data_loader:

            # 이미지를 GPU 또는 CPU로 이동한다.
            images = images.to(
                device=device,
                non_blocking=True,
            )

            # --------------------------------------------------
            # Forward 및 Loss 계산
            # --------------------------------------------------

            # 검증 forward에도 CUDA AMP를 사용할 수 있다.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                # eval mode의 YOLO11 Head는
                # 다음 tuple을 반환한다.
                #
                # (
                #     decoded_output,
                #     raw_output,
                # )
                model_output = model(
                    images
                )

                # 잘못된 출력 형식을
                # Loss에 전달하기 전에 확인한다.
                if (
                    not isinstance(
                        model_output,
                        tuple,
                    )
                    or len(
                        model_output
                    )
                    != 2
                ):
                    raise TypeError(
                        "eval mode의 모델 출력은 "
                        "(decoded_output, raw_output)"
                        "이어야 합니다."
                    )

                # decoded_output:
                #     bbox와 class 확률이 decode된 결과
                #
                # predictions:
                #     Loss 계산에 필요한 raw output
                (
                    decoded_output,
                    predictions,
                ) = model_output

                # validation loss 계산
                total_loss, loss_items = (
                    criterion(
                        predictions,
                        targets,
                    )
                )

            # NaN 또는 inf validation loss를
            # 정상 값처럼 평균내지 않도록 중단한다.
            if not torch.isfinite(
                total_loss
            ).all():
                raise FloatingPointError(
                    "유한하지 않은 "
                    "validation loss가 발생했습니다."
                )

            # --------------------------------------------------
            # 선택적인 mAP 누적
            # --------------------------------------------------

            if metric is not None:
                metric.update(
                    decoded_output=(
                        decoded_output
                    ),
                    targets=targets,
                )

            # --------------------------------------------------
            # Loss 누적
            # --------------------------------------------------

            running_box_loss += (
                loss_items[
                    "box_loss"
                ].item()
            )

            running_cls_loss += (
                loss_items[
                    "cls_loss"
                ].item()
            )

            running_dfl_loss += (
                loss_items[
                    "dfl_loss"
                ].item()
            )

            running_total_loss += (
                loss_items[
                    "total_loss"
                ].item()
            )

            batch_count += 1

    # Validation DataLoader가 비어 있으면
    # 평균 Loss를 계산할 수 없다.
    if batch_count == 0:
        raise ValueError(
            "검증 DataLoader가 비어 있습니다. "
            "검증 데이터셋 경로와 "
            "DataLoader 설정을 확인해주세요."
        )

    # --------------------------------------------------
    # Epoch 평균 Validation Loss
    # --------------------------------------------------

    epoch_result = {
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

    # mAP 기능이 활성화됐다면
    # 계산된 평가 결과를 같은 딕셔너리에 추가한다.
    if metric is not None:
        epoch_result.update(
            metric.compute()
        )

    # Validation 결과를
    # 콘솔과 로그 파일에 기록한다.
    if logger is not None:
        logger.info(
            "Validation | "
            "Box %.4f | "
            "Cls %.4f | "
            "DFL %.4f | "
            "Total %.4f",
            epoch_result[
                "box_loss"
            ],
            epoch_result[
                "cls_loss"
            ],
            epoch_result[
                "dfl_loss"
            ],
            epoch_result[
                "total_loss"
            ],
        )

        # mAP를 계산한 경우에는
        # mAP 결과도 별도로 기록한다.
        if metric is not None:
            logger.info(
                "Metrics | "
                "mAP50-95 %.4f | "
                "mAP50 %.4f | "
                "mAP75 %.4f | "
                "mAR100 %.4f",
                epoch_result[
                    "map"
                ],
                epoch_result[
                    "map_50"
                ],
                epoch_result[
                    "map_75"
                ],
                epoch_result[
                    "mar_100"
                ],
            )

    return epoch_result