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

    # requires_grad=True인 Parameter만 가져온다.
    #
    # 일부 계층을 freeze한 경우에는
    # freeze된 Parameter가 Optimizer에서 제외된다.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    # 모든 Parameter가 freeze돼 있다면
    # Optimizer를 생성해도 학습되는 값이 없다.
    if not trainable_parameters:
        raise ValueError(
            "학습 가능한 모델 파라미터가 없습니다."
        )

    # --------------------------------------------------
    # 3. AdamW Optimizer 생성
    # --------------------------------------------------

    # AdamW는 weight decay를
    # gradient 기반 Parameter 갱신과 분리해 적용한다.
    #
    # 현재 프로젝트에서는 구조를 단순하게 유지하기 위해
    # 모든 학습 가능 Parameter에 같은 설정을 사용한다.
    optimizer = torch.optim.AdamW(
        params=trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # train_epoch.py에서 사용할
    # 생성된 Optimizer를 반환한다.
    return optimizer