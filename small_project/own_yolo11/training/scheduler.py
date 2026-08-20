"""Warmup과 Cosine decay를 결합한 학습률 Scheduler를 생성한다."""

# Cosine 곡선을 계산할 때
# 원주율과 cos 함수를 사용한다.
import math

# PyTorch의 LambdaLR로
# batch별 학습률 배율을 적용한다.
import torch


def build_scheduler(
    optimizer,
    epochs,
    steps_per_epoch,
    warmup_epochs=3.0,
    min_lr_ratio=0.01,
):
    """
    Batch 단위 Warmup + Cosine Scheduler를 생성한다.

    학습률 변화:

        작은 학습률
            ↓ Warmup
        기본 학습률
            ↓ Cosine decay
        기본 학습률 x min_lr_ratio

    Args:
        optimizer:
            학습률을 변경할 Optimizer

        epochs:
            전체 학습 epoch 수

        steps_per_epoch:
            한 epoch의 학습 batch 수

        warmup_epochs:
            학습률을 서서히 증가시킬 epoch 수

        min_lr_ratio:
            마지막 학습률이 기본 학습률의
            몇 배가 될지 정하는 값

    Returns:
        scheduler:
            optimizer.step() 이후 batch마다
            step()을 호출할 Scheduler
    """

    # --------------------------------------------------
    # 1. 입력값 유효성 검사
    # --------------------------------------------------

    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):
        raise TypeError(
            "optimizer는 "
            "torch.optim.Optimizer여야 합니다."
        )

    if (
        isinstance(
            epochs,
            bool,
        )
        or not isinstance(
            epochs,
            int,
        )
    ):
        raise TypeError(
            "epochs는 정수여야 합니다."
        )

    if epochs <= 0:
        raise ValueError(
            "epochs는 1 이상이어야 합니다."
        )

    if (
        isinstance(
            steps_per_epoch,
            bool,
        )
        or not isinstance(
            steps_per_epoch,
            int,
        )
    ):
        raise TypeError(
            "steps_per_epoch은 정수여야 합니다."
        )

    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch은 1 이상이어야 합니다."
        )

    if (
        isinstance(
            warmup_epochs,
            bool,
        )
        or not isinstance(
            warmup_epochs,
            (int, float),
        )
    ):
        raise TypeError(
            "warmup_epochs는 숫자여야 합니다."
        )

    if not (
        0.0
        <= warmup_epochs
        < epochs
    ):
        raise ValueError(
            "warmup_epochs는 0 이상이고 "
            "epochs보다 작아야 합니다."
        )

    if (
        isinstance(
            min_lr_ratio,
            bool,
        )
        or not isinstance(
            min_lr_ratio,
            (int, float),
        )
    ):
        raise TypeError(
            "min_lr_ratio는 숫자여야 합니다."
        )

    if not (
        0.0
        < min_lr_ratio
        <= 1.0
    ):
        raise ValueError(
            "min_lr_ratio는 0보다 크고 "
            "1 이하여야 합니다."
        )

    # --------------------------------------------------
    # 2. Epoch 값을 batch step 단위로 변환
    # --------------------------------------------------

    # Scheduler는 batch마다 실행되므로
    # 전체 epoch를 전체 batch update 수로 변환한다.
    total_steps = (
        epochs
        * steps_per_epoch
    )

    # 소수 epoch도 사용할 수 있도록
    # 반올림해서 정수 step으로 바꾼다.
    warmup_steps = int(
        round(
            warmup_epochs
            * steps_per_epoch
        )
    )

    # Cosine 구간이 마지막 batch에서
    # progress=1에 도달하도록 간격을 계산한다.
    cosine_denominator = max(
        (
            total_steps
            - warmup_steps
            - 1
        ),
        1,
    )

    # --------------------------------------------------
    # 3. 현재 step의 학습률 배율 계산
    # --------------------------------------------------

    def learning_rate_multiplier(
        current_step,
    ):
        """
        현재 batch step을
        기본 learning rate의 배율로 변환한다.
        """

        # Warmup 구간에서는
        # 1/warmup_steps부터 1까지 선형 증가한다.
        if (
            warmup_steps > 0
            and current_step
            < warmup_steps
        ):
            return (
                current_step + 1
            ) / warmup_steps

        # Warmup 이후 현재 위치를
        # 0~1 범위의 진행률로 변환한다.
        cosine_progress = (
            (
                current_step
                - warmup_steps
            )
            / cosine_denominator
        )

        # checkpoint 복원 또는 마지막 추가 호출로
        # 범위를 벗어나도 비정상 학습률이 되지 않게 제한한다.
        cosine_progress = min(
            max(
                cosine_progress,
                0.0,
            ),
            1.0,
        )

        # progress=0에서는 1,
        # progress=1에서는 0인 Cosine 값을 만든다.
        cosine_value = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * cosine_progress
                )
            )
        )

        # 최종 학습률 배율이 0이 아니라
        # min_lr_ratio에서 끝나도록 조정한다.
        return (
            min_lr_ratio
            + (
                1.0
                - min_lr_ratio
            )
            * cosine_value
        )

    # --------------------------------------------------
    # 4. PyTorch Scheduler 생성
    # --------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=(
                learning_rate_multiplier
            ),
        )
    )

    return scheduler