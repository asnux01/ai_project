# --------------------------------------------------
# Import library
# --------------------------------------------------

import torch


# --------------------------------------------------
# Optimizer 생성 함수
# --------------------------------------------------

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

            Optimizer는 model.parameters()를 통해
            모델 내부의 학습 가능한 가중치와 bias를
            전달받는다.

        learning_rate:
            한 번의 업데이트에서 모델 가중치를
            얼마나 크게 변경할지 정하는 학습률

            기본값:
                0.001

        weight_decay:
            모델 가중치가 지나치게 커지는 것을
            억제하기 위한 규제값

            기본값:
                0.0005

    Returns:
        optimizer:
            생성된 AdamW Optimizer
    """

    # --------------------------------------------------
    # 학습 가능한 Parameter 선택
    # --------------------------------------------------

    # requires_grad=True인 Parameter만 가져온다.
    #
    # Backbone, Neck, Head 중 일부를 freeze한 경우,
    # freeze된 Parameter는 Optimizer에서 제외된다.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


    # --------------------------------------------------
    # AdamW Optimizer 생성
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        # Optimizer가 실제로 수정할 모델 Parameter
        params=trainable_parameters,

        # Learning rate
        lr=learning_rate,

        # Weight decay
        weight_decay=weight_decay,
    )


    # --------------------------------------------------
    # 생성된 Optimizer 반환
    # --------------------------------------------------

    return optimizer