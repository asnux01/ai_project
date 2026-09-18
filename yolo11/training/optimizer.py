# 라이브러리
import torch
import torch.nn as nn


# Adam Optimizer 생성
def build_optimizer(
    model,
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0005
):

    # Weight decay를 적용할 Parameter
    decay_parameters = []
    
    # BatchNorm 등 Weight decay를 적용하지 않을 Parameter
    norm_parameters = []
    
    # Bias Parameter
    bias_parameters = []
    
    # Normalization Layer
    norm_layers = tuple(
        module_type
        for module_name, module_type
        in nn.__dict__.items()
        if "Norm" in module_name
    )
    
    # Module 단위로 Parameter 분류
    for module in model.modules():
        
        for parameter_name, parameter in module.named_parameters(recurse=False):
            
            if not parameter.requires_grad:
                continue
            
            # Bias에는 Weight decay를 적용하지 않음
            if parameter_name == "bias":
                bias_parameters.append(parameter)
                
            # Normalization
            elif isinstance(module, norm_layers):
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

    # SGD Optimzer 생성
    optimizer = torch.optim.SGD(
        params=bias_parameters,
        lr=learning_rate,
        momentum=momentum,
        nesterov=True,
        weight_decay=0.0
    )
    
    # Bias Group
    optimizer.param_groups[0]["param_group"] = "bias"
    
    # Decay Group
    optimizer.add_param_group({
        "params": decay_parameters,
        "weight_decay": weight_decay,
        "param_group": "decay"
    })
    
    # Norm Group
    optimizer.add_param_group({
        "params": norm_parameters,
        "weight_decay": 0.0,
        "param_group": "norm"
    })
    
    return optimizer