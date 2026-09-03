# 라이브러리
import copy
import math

import torch


# Exponential Moving Average
class ModelEMA:

    def __init__(
        self,
        model,
        decay=0.9999,
        tau=2000
    ):

        # Model 복사
        self.ema = copy.deepcopy(model).eval()

        # EMA Model Gradient 비활성화
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

        # EMA 설정 저장
        self.decay = decay
        self.tau = tau

        # EMA Update 횟수 초기화
        self.updates = 0


    def _get_decay(self):

        # Update 횟수 증가
        self.updates += 1

        # 초기 학습에서 EMA Decay를 점진적으로 증가
        decay = (
            self.decay
            * (1.0 - math.exp(-self.updates / self.tau))
        )

        return decay


    @torch.no_grad()
    def update(
        self,
        model
    ):

        # 현재 EMA Decay 계산
        decay = self._get_decay()

        # 현재 Model Parameter 가져오기
        model_state = model.state_dict()

        # EMA Model Parameter 업데이트
        for name, ema_value in self.ema.state_dict().items():

            # 현재 Model 값 가져오기
            model_value = model_state[name].detach()

            # Floating Point Tensor만 EMA 적용
            if torch.is_floating_point(ema_value):

                ema_value.mul_(decay)
                ema_value.add_(model_value, alpha=1.0 - decay)

            # 정수형 Buffer는 현재 Model 값 복사
            else:
                ema_value.copy_(model_value)


    @torch.no_grad()
    def set(
        self,
        model
    ):

        # Model Parameter를 EMA Model에 그대로 복사
        self.ema.load_state_dict(model.state_dict())


    def state_dict(self):

        # EMA Model Parameter 반환
        return self.ema.state_dict()