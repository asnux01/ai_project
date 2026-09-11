# 라이브러리
import torch
import torch.nn as nn


# Adam Optimizer 생성
def build_optimizer(
    model,
    learning_rate=0.001,
    weight_decay=0.0005,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8
):

    # Weight decay를 적용할 Parameter
    decay_parameters = []
    
    # BatchNorm 등 Weight decay를 적용하지 않을 Parameter
    norm_parameters = []
    
    # Bias Parameter
    bias_parameters = []
    
    # Module 단위로 Parameter 분류
    for module in model.modules():
        
        for parameter_name, parameter in module.named_parameters(recurse=False):
            
            if not parameter.requires_grad:
                continue
            
            # Bias에는 Weight decay를 적용하지 않음
            if parameter_name == "bias":
                bias_parameters.append(parameter)
                
            # BatchNorm 계열에도 Weight decay를 적용하지 않음
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                norm_parameters.append(parameter)
                
            # Conv / Linear 등의 Weight에는 Weight decay 적용
            else:
                decay_parameters.append(parameter)
    
    # 학습 가능한 Parameter 확인
    num_trainable_parameters = (
        len(decay_parameters)
        + len(norm_parameters)
        + len(bias_parameters)
    )

    if num_trainable_parameters == 0:
        raise ValueError("Model은 trainable parameters가 없습니다.")

    # Parameter Group 구성
    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
            "param_group": "decay"
        },
        {
            "params": norm_parameters,
            "weight_decay": 0.0,
            "param_group": "norm"
        },
        {
            "params": bias_parameters,
            "weight_decay": 0.0,
            "param_group": "bias"
        }
    ]
    
    # Adam Optimizer 생성
    optimizer = torch.optim.Adam(
        params=parameter_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay
    )

    return optimizer