# 라이브러리
import torch


# Optimizer 생성 함수
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
    # 유효성 검사
    # 잘못된 하이퍼파라미터가 optimizer 내부에서
    # 늦게 실패하지 않도록 미리 검사
    if learning_rate <= 0:
        raise ValueError(
            "learning_rate는 0보다 커야 합니다."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay는 0 이상이어야 합니다."
        )
        
    # 학습 가능한 Parameter 선택
    # requires_grad=True인 Parameter만 가져온다.
    # freeze된 파라미터는 optimizer에서 제외
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    
    # 모든 파라미터가 freeze된 경우
    # 의미 없는 optimizer 생성 방해
    if not trainable_parameters:
        raise ValueError(
            "학습 가능한 모델 파라미터가 없습니다."
        )


    # AdamW Optimizer 생성
    optimizer = torch.optim.AdamW(
        params=trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


    # 생성된 Optimizer 반환
    return optimizer