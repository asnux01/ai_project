"""학습 상태 전체를 안전하게 저장하고 복원한다."""

# checkpoint 경로와 임시 파일을
# 운영체제에 관계없이 처리한다.
from pathlib import Path

# 모델, Optimizer 등의 state_dict를 파일로 저장한다.
import torch


# checkpoint 내부 구조의 버전이다.
#
# 나중에 저장 구조가 바뀌면 이 값을 올려서
# 이전 checkpoint와 구분할 수 있다.
CHECKPOINT_FORMAT_VERSION = 1


def build_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    ema,
    best_val_loss,
    best_map,
    global_step,
    model_config,
    train_config,
    train_loss=None,
    val_loss=None,
):
    """
    현재 학습 상태를
    하나의 checkpoint 딕셔너리로 묶는다.

    epoch에는 지금까지 완료된 epoch 수를 저장한다.

    예:
        첫 번째 epoch를 끝낸 경우:
            epoch=1
    """

    # --------------------------------------------------
    # 1. 필수 값 유효성 검사
    # --------------------------------------------------

    # bool은 int의 하위 자료형이므로
    # 정수 검사에서 별도로 제외한다.
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
    ):
        raise TypeError(
            "epoch는 정수여야 합니다."
        )

    if epoch < 0:
        raise ValueError(
            "epoch는 0 이상이어야 합니다."
        )

    if (
        isinstance(global_step, bool)
        or not isinstance(
            global_step,
            int,
        )
    ):
        raise TypeError(
            "global_step은 정수여야 합니다."
        )

    if global_step < 0:
        raise ValueError(
            "global_step은 0 이상이어야 합니다."
        )

    if not isinstance(
        model_config,
        dict,
    ):
        raise TypeError(
            "model_config는 dict여야 합니다."
        )

    if not isinstance(
        train_config,
        dict,
    ):
        raise TypeError(
            "train_config는 dict여야 합니다."
        )

    # --------------------------------------------------
    # 2. 기본 학습 상태 저장
    # --------------------------------------------------

    checkpoint = {
        # 저장 구조 호환성을 확인할 버전
        "format_version": (
            CHECKPOINT_FORMAT_VERSION
        ),

        # 지금까지 완료된 epoch 수
        "epoch": int(
            epoch
        ),

        # 지금까지 실제로 실행된 Optimizer 갱신 수
        "global_step": int(
            global_step
        ),

        # 같은 구조의 YOLO 모델을
        # 다시 생성하는 데 필요한 설정
        "model_config": dict(
            model_config
        ),

        # 이번 학습에 사용한 전체 설정
        "train_config": dict(
            train_config
        ),

        # 모델의 Parameter와 Buffer
        "model_state_dict": (
            model.state_dict()
        ),

        # AdamW momentum 등의 상태
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        # 지금까지 가장 작은 validation loss
        "best_val_loss": float(
            best_val_loss
        ),
        
        # 지금까지 기록한 최고 mAP50-95
        "best_map": float(
            best_map
        ),
    }

    # --------------------------------------------------
    # 3. 선택적인 학습 상태 저장
    # --------------------------------------------------

    # Scheduler를 사용하지 않는 실험도
    # 저장할 수 있도록 None을 허용한다.
    checkpoint[
        "scheduler_state_dict"
    ] = (
        scheduler.state_dict()
        if scheduler is not None
        else None
    )

    # AMP를 사용하지 않으면
    # scaler가 None일 수 있다.
    checkpoint[
        "scaler_state_dict"
    ] = (
        scaler.state_dict()
        if scaler is not None
        else None
    )

    # EMA를 사용하지 않는 경우에는
    # EMA 상태 대신 None을 저장한다.
    checkpoint[
        "ema_state_dict"
    ] = (
        ema.state_dict()
        if ema is not None
        else None
    )

    # 사람이 checkpoint 결과를 확인할 수 있도록
    # 마지막 train loss도 함께 저장한다.
    checkpoint[
        "train_loss"
    ] = (
        dict(train_loss)
        if train_loss is not None
        else None
    )

    # 마지막 validation loss 및 mAP를 저장한다.
    checkpoint[
        "val_loss"
    ] = (
        dict(val_loss)
        if val_loss is not None
        else None
    )

    return checkpoint


def save_checkpoint(
    checkpoint,
    checkpoint_path,
):
    """
    checkpoint를 임시 파일에 먼저 기록한 뒤
    최종 파일로 교체한다.
    """

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "checkpoint는 dict여야 합니다."
        )

    checkpoint_path = Path(
        checkpoint_path
    )

    # PyTorch checkpoint임을 알 수 있도록
    # .pt 확장자만 허용한다.
    if (
        checkpoint_path.suffix
        != ".pt"
    ):
        raise ValueError(
            "checkpoint 파일 확장자는 "
            ".pt여야 합니다."
        )

    # checkpoints 디렉터리가 없다면
    # 상위 경로까지 함께 생성한다.
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 저장 도중 서버나 프로세스가 중단되더라도
    # 기존 정상 checkpoint가 손상되지 않게 한다.
    #
    # 예:
    # last.pt → last.pt.tmp에 먼저 저장
    temporary_path = (
        checkpoint_path.with_suffix(
            checkpoint_path.suffix
            + ".tmp"
        )
    )

    try:
        # 학습 상태 전체를 임시 파일에 기록한다.
        torch.save(
            checkpoint,
            temporary_path,
        )

        # 저장이 완전히 끝난 임시 파일만
        # 최종 checkpoint 이름으로 교체한다.
        temporary_path.replace(
            checkpoint_path
        )

    except Exception:

        # 저장 실패로 남은 불완전한 임시 파일을 제거한다.
        temporary_path.unlink(
            missing_ok=True
        )

        raise

    # 로그에서 저장 위치를 기록할 수 있도록
    # 최종 경로를 반환한다.
    return checkpoint_path


def _load_checkpoint_file(
    checkpoint_path,
    map_location,
):
    """
    설치된 PyTorch 버전에 맞춰
    checkpoint 파일을 읽는다.
    """

    try:
        # 이 checkpoint는 Tensor와 기본 자료형만 저장한다.
        #
        # weights_only=True를 사용하면
        # 임의 Python 객체 실행 위험을 줄일 수 있다.
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )

    except TypeError:
        # 오래된 PyTorch에는
        # weights_only 인자가 없을 수 있다.
        #
        # 이 경우 직접 생성한
        # 신뢰할 수 있는 checkpoint만 불러와야 한다.
        return torch.load(
            checkpoint_path,
            map_location=map_location,
        )


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    map_location="cpu"
):
    """
    checkpoint에서 모델과
    전체 학습 상태를 복원한다.

    Args:
        checkpoint_path:
            불러올 .pt 파일 경로

        model:
            Parameter를 복원할 YOLO 모델

        optimizer:
            AdamW 상태를 복원할 Optimizer

        scheduler:
            현재 학습률 위치를 복원할 Scheduler

        scaler:
            AMP scale을 복원할 GradScaler

        ema:
            EMA 모델을 복원할 ModelEMA

        map_location:
            checkpoint Tensor를 먼저 불러올 장치

    Returns:
        resume_state:
            train.py에서 학습을 이어가는 데 필요한 상태
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    # 지정한 checkpoint가 실제 파일인지 확인한다.
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "checkpoint 파일을 "
            "찾을 수 없습니다: "
            f"{checkpoint_path}"
        )

    # checkpoint 파일을 지정한 장치로 불러온다.
    checkpoint = (
        _load_checkpoint_file(
            checkpoint_path=(
                checkpoint_path
            ),
            map_location=(
                map_location
            ),
        )
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "checkpoint 파일 내부는 "
            "dict여야 합니다."
        )

    # 모델 및 학습 재개에 반드시 필요한 key
    required_keys = {
        "format_version",
        "epoch",
        "global_step",
        "model_state_dict",
        "best_val_loss",
        "model_config",
    }

    missing_keys = (
        required_keys
        - checkpoint.keys()
    )

    if missing_keys:
        raise KeyError(
            "checkpoint에 필요한 값이 없습니다: "
            f"{sorted(missing_keys)}"
        )

    # 현재 코드가 지원하는 저장 형식인지 확인한다.
    if (
        checkpoint["format_version"]
        != CHECKPOINT_FORMAT_VERSION
    ):
        raise RuntimeError(
            "지원하지 않는 checkpoint 형식입니다: "
            f"{checkpoint['format_version']}"
        )

    # --------------------------------------------------
    # 1. 모델 상태 복원
    # --------------------------------------------------

    # 모델 구조가 다르면 missing 또는 unexpected key가
    # 즉시 표시되도록 strict=True를 사용한다.
    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    # --------------------------------------------------
    # 2. Optimizer 상태 복원
    # --------------------------------------------------

    if optimizer is not None:

        optimizer_state = (
            checkpoint.get(
                "optimizer_state_dict"
            )
        )

        if optimizer_state is None:
            raise KeyError(
                "checkpoint에 "
                "optimizer_state_dict가 없습니다."
            )

        optimizer.load_state_dict(
            optimizer_state
        )

    # --------------------------------------------------
    # 3. Scheduler 상태 복원
    # --------------------------------------------------

    if scheduler is not None:

        scheduler_state = (
            checkpoint.get(
                "scheduler_state_dict"
            )
        )

        if scheduler_state is None:
            raise KeyError(
                "checkpoint에 "
                "scheduler_state_dict가 없습니다."
            )

        scheduler.load_state_dict(
            scheduler_state
        )

    # --------------------------------------------------
    # 4. AMP GradScaler 상태 복원
    # --------------------------------------------------

    if scaler is not None:

        scaler_state = (
            checkpoint.get(
                "scaler_state_dict"
            )
        )

        if scaler_state is None:
            raise KeyError(
                "checkpoint에 "
                "scaler_state_dict가 없습니다."
            )

        scaler.load_state_dict(
            scaler_state
        )

    # --------------------------------------------------
    # 5. EMA 상태 복원
    # --------------------------------------------------

    if ema is not None:

        ema_state = (
            checkpoint.get(
                "ema_state_dict"
            )
        )

        if ema_state is None:
            raise KeyError(
                "checkpoint에 "
                "ema_state_dict가 없습니다."
            )

        ema.load_state_dict(
            ema_state
        )

    # --------------------------------------------------
    # 6. 학습 반복 상태 확인
    # --------------------------------------------------

    epoch = int(
        checkpoint["epoch"]
    )

    global_step = int(
        checkpoint["global_step"]
    )

    best_val_loss = float(
        checkpoint[
            "best_val_loss"
        ]
    )

    # 최고 mAP50-95 복원
    saved_val_result = (
        checkpoint.get(
            "val_loss"
        )
    )

    if isinstance(
        saved_val_result,
        dict,
    ):
        fallback_map = float(
            saved_val_result.get(
                "map",
                float(
                    "-inf"
                ),
            )
        )

    else:
        fallback_map = float(
            "-inf"
        )

    best_map = float(
        checkpoint.get(
            "best_map",
            fallback_map,
        )
    )
    
    if (
        epoch < 0
        or global_step < 0
    ):
        raise ValueError(
            "checkpoint의 epoch 또는 "
            "global_step이 올바르지 않습니다."
        )

    # epoch는 이미 완료된 epoch 수다.
    #
    # 예:
    # start_epoch=10이면
    # range(10, epochs)부터 바로 이어서 학습한다.
    resume_state = {
        "start_epoch": epoch,
        "global_step": global_step,
        "best_val_loss": (
            best_val_loss
        ),
        "best_map": (
            best_map
        ),
        "model_config": (
            checkpoint[
                "model_config"
            ]
        ),
        "train_config": (
            checkpoint.get(
                "train_config"
            )
        ),
        "train_loss": (
            checkpoint.get(
                "train_loss"
            )
        ),
        "val_loss": (
            checkpoint.get(
                "val_loss"
            )
        ),
    }

    return resume_state