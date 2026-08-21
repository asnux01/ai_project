"""YOLO11 학습에 사용할 Optimizer를 생성한다."""

# AdamW Optimizer를 사용하기 위해 PyTorch를 불러온다.
import torch


def build_optimizer(
    model,
    learning_rate=0.001,
    weight_decay=0.0005,
):
    """
    YOLO 모델을 학습시키기 위한
    AdamW Optimizer를 생성한다.

    Args:
        model:
            학습할 YOLO 모델

        learning_rate:
            한 번의 업데이트에서
            모델 가중치를 변경할 정도

        weight_decay:
            모델 가중치가 지나치게 커지는 것을
            억제하기 위한 규제값

    Returns:
        optimizer:
            생성된 AdamW Optimizer
    """

    # --------------------------------------------------
    # 1. Optimizer 설정 유효성 검사
    # --------------------------------------------------

    # bool은 int의 하위 자료형이므로
    # 숫자 검사에서 별도로 제외한다.
    if (
        isinstance(
            learning_rate,
            bool,
        )
        or not isinstance(
            learning_rate,
            (int, float),
        )
    ):
        raise TypeError(
            "learning_rate는 숫자여야 합니다."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate는 0보다 커야 합니다."
        )

    if (
        isinstance(
            weight_decay,
            bool,
        )
        or not isinstance(
            weight_decay,
            (int, float),
        )
    ):
        raise TypeError(
            "weight_decay는 숫자여야 합니다."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay는 0 이상이어야 합니다."
        )

    # --------------------------------------------------
    # 2. 학습 가능한 Parameter 선택
    # --------------------------------------------------

   # 일반 Conv weight 등
    # Weight Decay를 적용할 Parameter
    decay_parameters = []

    # BatchNorm weight 및 bias 등
    # Weight Decay를 적용하지 않을 Parameter
    no_decay_parameters = []


    for (
        name,
        parameter,
    ) in model.named_parameters():

        # freeze된 Parameter는
        # Optimizer에 넣지 않는다.
        if not parameter.requires_grad:
            continue

        # --------------------------------------------------
        # Bias / BatchNorm 계열
        # --------------------------------------------------
        #
        # BatchNorm weight는 일반적으로
        # 1차원 Parameter다.
        #
        # 또한 bias에는 Weight Decay를
        # 적용하지 않는다.
        if (
            parameter.ndim == 1
            or name.endswith(
                ".bias"
            )
        ):

            no_decay_parameters.append(
                parameter
            )

        else:

            # 일반 Conv2d weight 등은
            # Weight Decay를 적용한다.
            decay_parameters.append(
                parameter
            )


    if (
        not decay_parameters
        and not no_decay_parameters
    ):
        raise ValueError(
            "학습 가능한 모델 파라미터가 없습니다."
        )


    # --------------------------------------------------
    # 3. AdamW Optimizer 생성
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                # 일반 weight
                "params": (
                    decay_parameters
                ),

                "weight_decay": (
                    weight_decay
                ),
            },

            {
                # Bias와 BatchNorm 계열
                "params": (
                    no_decay_parameters
                ),

                # 이 Parameter에는
                # Weight Decay를 적용하지 않는다.
                "weight_decay": 0.0,
            },
        ],

        lr=learning_rate,
    )
    
    # train_epoch.py에서 사용할
    # 생성된 Optimizer를 반환한다.
    return optimizer