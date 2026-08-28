# 라이브러리
import torch


# Adam Optimizer 생성
def build_optimizer(
    model,
    learning_rate=0.001,
    weight_decay=0.0005,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8
):

    # 학습 가능한 Parameter 가져오기
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    # 학습 가능한 Parameter 확인
    if not parameters:
        raise ValueError(
            "Model has no trainable parameters."
        )

    # Adam Optimizer 생성
    optimizer = torch.optim.Adam(
        params=parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay
    )

    return optimizer