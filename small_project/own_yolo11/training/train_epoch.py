"""AMP, Scheduler, EMA를 적용해 학습 데이터 한 epoch를 처리한다."""

# Forward, backward, AMP,
# gradient clipping 등에 사용한다.
import torch
from tqdm.auto import tqdm


def train_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    scheduler=None,
    ema=None,
    use_amp=True,
    max_grad_norm=None,
    epoch_index=0,
    total_epochs=1,
    global_step=0,
    logger=None,
    log_interval=20,
):
    """
    학습 데이터 전체를 한 번 순회한다.

    한 batch의 처리 순서:

        images, targets
            ↓
        AMP autocast
            ↓
        model(images)
            ↓
        criterion(predictions, targets)
            ↓
        GradScaler 또는 일반 backward
            ↓
        Gradient clipping
            ↓
        optimizer.step()
            ↓
        scheduler.step()
            ↓
        EMA update

    Args:
        model:
            학습할 YOLO 모델

        data_loader:
            학습용 DataLoader

        criterion:
            YOLO11DetectionLoss

        optimizer:
            모델 Parameter를 갱신할 Optimizer

        device:
            cpu 또는 cuda 장치

        scaler:
            CUDA AMP에 사용할 GradScaler

        scheduler:
            batch 단위로 갱신할 Scheduler

        ema:
            Optimizer 갱신 이후 업데이트할 ModelEMA

        use_amp:
            CUDA에서 자동 혼합 정밀도를 사용할지 결정

        max_grad_norm:
            Gradient 최대 norm

            None이면 clipping하지 않음

        epoch_index:
            0부터 시작하는 현재 epoch 번호

        total_epochs:
            전체 epoch 수

        global_step:
            지금까지 실제로 실행된 Optimizer 갱신 수

        logger:
            학습 진행 상황을 기록할 logger

        log_interval:
            몇 batch마다 진행 상황을 기록할지 결정

    Returns:
        epoch_loss:
            한 epoch의 평균 Loss

        global_step:
            이번 epoch 이후 누적 Optimizer 갱신 수
    """

    # 문자열 device도 처리할 수 있도록
    # torch.device로 통일한다.
    device = torch.device(
        device
    )

    if log_interval <= 0:
        raise ValueError(
            "log_interval은 1 이상이어야 합니다."
        )

    if (
        max_grad_norm is not None
        and max_grad_norm <= 0
    ):
        raise ValueError(
            "max_grad_norm은 0보다 크거나 "
            "None이어야 합니다."
        )

    # AMP float16은 CUDA에서만 활성화한다.
    #
    # CPU에서 use_amp=True여도 자동으로 비활성화된다.
    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    # CUDA AMP를 사용하는데 GradScaler가 없으면
    # float16 gradient가 0이 될 가능성이 있다.
    if (
        amp_enabled
        and scaler is None
    ):
        raise ValueError(
            "CUDA AMP를 사용하려면 "
            "GradScaler가 필요합니다."
        )

    # BatchNorm과 Dropout 등이
    # 학습 모드로 동작하도록 설정한다.
    model.train()

    # --------------------------------------------------
    # Loss 누적 변수
    # --------------------------------------------------

    running_box_loss = 0.0
    running_cls_loss = 0.0
    running_dfl_loss = 0.0
    running_total_loss = 0.0

    batch_count = 0

    # 진행 로그에 전체 batch 수를 표시한다.
    total_batches = len(
        data_loader
    )

    # Ultralytics 형식의 진행도 헤더를 출력한다.
    progress_header = (
        f"{'Epoch':>11}"
        f"{'GPU_mem':>11}"
        f"{'box_loss':>11}"
        f"{'cls_loss':>11}"
        f"{'dfl_loss':>11}"
        f"{'Instances':>11}"
        f"{'Size':>11}"
    )

    print(progress_header)

    # 현재 epoch의 배치 진행률을 표시한다.
    progress_bar = tqdm(
        data_loader,
        total=total_batches,
        dynamic_ncols=True,
        leave=True,
    )
    
    # --------------------------------------------------
    # DataLoader 순회
    # --------------------------------------------------

    for (
        batch_index,
        (
            images,
            targets,
        ),
    ) in enumerate(
        progress_bar
    ):
        # 이미지를 GPU 또는 CPU로 이동한다.
        #
        # pin_memory=True인 DataLoader와 CUDA를 사용하면
        # non_blocking=True가 비동기 전송에 도움을 줄 수 있다.
        images = images.to(
            device=device,
            non_blocking=True,
        )

        # 이전 batch의 Gradient를 초기화한다.
        #
        # set_to_none=True는 0 Tensor를 만드는 것보다
        # 메모리 사용과 연산량을 줄일 수 있다.
        optimizer.zero_grad(
            set_to_none=True,
        )

        # --------------------------------------------------
        # Forward와 Loss 계산
        # --------------------------------------------------

        # 모델 forward에만 AMP float16을 적용한다.
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            predictions = model(
                images
            )

        # train mode의 YOLO11 Head는
        # raw output 딕셔너리를 반환해야 한다.
        if not isinstance(
            predictions,
            dict,
        ):
            raise TypeError(
                "train mode 모델 출력은 "
                "dict여야 합니다."
            )

        required_prediction_keys = {
            "box_logits",
            "class_logits",
            "features",
        }

        missing_prediction_keys = (
            required_prediction_keys
            - predictions.keys()
        )

        if missing_prediction_keys:
            raise KeyError(
                "모델 출력에 필요한 값이 없습니다: "
                f"{sorted(missing_prediction_keys)}"
            )

        # 모델의 AMP 출력값을 loss 계산용 FP32로 변환한다.
        # float() 연산을 사용해도 gradient 연결은 유지된다.
        loss_predictions = dict(
            predictions
        )

        loss_predictions[
            "box_logits"
        ] = predictions[
            "box_logits"
        ].float()

        loss_predictions[
            "class_logits"
        ] = predictions[
            "class_logits"
        ].float()

        loss_predictions[
            "features"
        ] = [
            feature.float()
            for feature
            in predictions["features"]
        ]

        # 사용자 정의 loss와 TaskAlignedAssigner는
        # AMP를 끄고 FP32로 계산한다.
        with torch.autocast(
            device_type=device.type,
            enabled=False,
        ):
            total_loss, loss_items = (
                criterion(
                    loss_predictions,
                    targets,
                )
            )

        # NaN 또는 inf loss로 backward하면
        # 모델 Parameter 전체가 손상될 수 있으므로 중단한다.
        if not torch.isfinite(
            total_loss
        ).all():
            raise FloatingPointError(
                "유한하지 않은 학습 loss가 "
                "발생했습니다. "
                f"epoch={epoch_index + 1}, "
                f"batch={batch_index + 1}"
            )

        # AMP overflow가 없으면
        # Optimizer가 실행됐다고 판단하기 위한 기본값
        optimizer_was_run = True

        # --------------------------------------------------
        # AMP backward 및 Parameter 갱신
        # --------------------------------------------------

        if amp_enabled:
            # 작은 float16 Gradient가 0이 되지 않도록
            # Loss에 scale을 적용한 뒤 backward한다.
            scaler.scale(
                total_loss
            ).backward()

            if max_grad_norm is not None:
                # Gradient clipping 전에
                # scale된 Gradient를 원래 크기로 되돌린다.
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        max_grad_norm
                    ),
                )

            # overflow 여부를 판단하기 위해
            # update 이전 scale을 저장한다.
            previous_scale = (
                scaler.get_scale()
            )

            # inf 또는 NaN Gradient가 있으면
            # GradScaler가 optimizer.step()을 자동으로 건너뛴다.
            scaler.step(
                optimizer
            )

            # 다음 batch에 사용할 scale을 갱신한다.
            scaler.update()

            # scale이 감소했다면 overflow가 발생해
            # 이번 Optimizer update가 실행되지 않은 것이다.
            optimizer_was_run = (
                scaler.get_scale()
                >= previous_scale
            )

        # --------------------------------------------------
        # 일반 float32 backward 및 Parameter 갱신
        # --------------------------------------------------

        else:
            total_loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        max_grad_norm
                    ),
                )

            optimizer.step()

        # --------------------------------------------------
        # Scheduler와 EMA 갱신
        # --------------------------------------------------

        # Optimizer가 실제로 실행된 경우에만
        # Scheduler와 EMA도 같은 update 횟수로 진행한다.
        if optimizer_was_run:

            if scheduler is not None:
                scheduler.step()

            if ema is not None:
                ema.update(
                    model
                )

            global_step += 1

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

        # 현재 epoch에서 지금까지 계산한 평균 loss
        average_box_loss = (
            running_box_loss
            / batch_count
        )

        average_cls_loss = (
            running_cls_loss
            / batch_count
        )

        average_dfl_loss = (
            running_dfl_loss
            / batch_count
        )

        # 현재 배치에 포함된 전체 객체 수
        instance_count = sum(
            int(
                target["labels"].shape[0]
            )
            for target in targets
        )

        # 입력 이미지 크기
        input_size = int(
            images.shape[-1]
        )

        # 현재 CUDA 예약 메모리
        if device.type == "cuda":
            gpu_memory_gb = (
                torch.cuda.memory_reserved(
                    device=device
                )
                / (1024 ** 3)
            )

            gpu_memory_text = (
                f"{gpu_memory_gb:.1f}G"
            )

        else:
            gpu_memory_text = "0G"

        # tqdm 설명 부분을 Ultralytics 형식으로 갱신한다.
        epoch_text = (
            f"{epoch_index + 1}"
            f"/{total_epochs}"
        )

        progress_description = (
            f"{epoch_text:>11}"
            f"{gpu_memory_text:>11}"
            f"{average_box_loss:>11.3f}"
            f"{average_cls_loss:>11.3f}"
            f"{average_dfl_loss:>11.3f}"
            f"{instance_count:>11}"
            f"{input_size:>11}"
        )

        progress_bar.set_description(
            progress_description,
            refresh=False,
        )

    # 빈 DataLoader이면
    # 아래 평균 계산에서 0으로 나누게 된다.
    if batch_count == 0:
        raise ValueError(
            "학습 DataLoader가 비어 있습니다."
        )

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

    return (
        epoch_loss,
        global_step,
    )