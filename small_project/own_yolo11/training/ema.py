"""학습 모델 Parameter의 지수 이동평균 모델을 관리한다."""

# EMA 검증용 모델을
# 학습 모델과 분리해서 만들 때 사용한다.
from copy import deepcopy

# 학습 초기 EMA decay를
# 부드럽게 증가시킬 때 사용한다.
import math

# Parameter 갱신과 nn.Module 검사에 사용한다.
import torch
import torch.nn as nn


class ModelEMA:
    """
    학습 모델의 Exponential Moving Average
    복사본을 관리한다.
    """

    def __init__(
        self,
        model,
        decay=0.9999,
        tau=2000.0,
        updates=0,
    ):
        """
        Args:
            model:
                이동평균의 기준이 되는 학습 모델

            decay:
                이전 EMA Parameter를 유지할 최대 비율

            tau:
                학습 초기에 decay를
                천천히 증가시키는 정도

            updates:
                checkpoint에서 복원할
                기존 EMA 갱신 횟수
        """

        # --------------------------------------------------
        # 1. 입력값 유효성 검사
        # --------------------------------------------------

        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model은 nn.Module이어야 합니다."
            )

        if (
            isinstance(
                decay,
                bool,
            )
            or not isinstance(
                decay,
                (int, float),
            )
        ):
            raise TypeError(
                "decay는 숫자여야 합니다."
            )

        if not (
            0.0
            < decay
            < 1.0
        ):
            raise ValueError(
                "decay는 0과 1 사이여야 합니다."
            )

        if (
            isinstance(
                tau,
                bool,
            )
            or not isinstance(
                tau,
                (int, float),
            )
        ):
            raise TypeError(
                "tau는 숫자여야 합니다."
            )

        if tau <= 0:
            raise ValueError(
                "tau는 0보다 커야 합니다."
            )

        if (
            isinstance(
                updates,
                bool,
            )
            or not isinstance(
                updates,
                int,
            )
        ):
            raise TypeError(
                "updates는 정수여야 합니다."
            )

        if updates < 0:
            raise ValueError(
                "updates는 0 이상이어야 합니다."
            )

        # --------------------------------------------------
        # 2. EMA 전용 모델 생성
        # --------------------------------------------------

        # 학습 모델과 Parameter 저장 공간을 공유하지 않는
        # 독립적인 모델 복사본을 만든다.
        self.ema = deepcopy(
            model
        ).eval()

        # EMA 모델은 backward 대상이 아니므로
        # 모든 Parameter의 gradient 계산을 끈다.
        for parameter in (
            self.ema.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        # EMA 계산 설정과 현재 상태를 저장한다.
        self.decay = float(
            decay
        )

        self.tau = float(
            tau
        )

        self.updates = int(
            updates
        )

    def _current_decay(self):
        """
        현재 update 횟수에 맞는
        EMA decay를 계산한다.
        """

        # 첫 update부터 0.9999 같은 큰 값을 사용하면
        # 초기 모델의 영향이 지나치게 오래 유지될 수 있다.
        #
        # 학습 초반에는 작은 decay를 사용하고,
        # update가 증가하면 설정된 최대 decay에 가까워진다.
        return (
            self.decay
            * (
                1.0
                - math.exp(
                    -self.updates
                    / self.tau
                )
            )
        )

    @torch.no_grad()
    def update(
        self,
        model,
    ):
        """
        optimizer 갱신이 끝난 학습 모델을
        EMA 모델에 반영한다.
        """

        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model은 nn.Module이어야 합니다."
            )

        # optimizer update 횟수와 동일하게
        # EMA update 횟수도 한 번 증가시킨다.
        self.updates += 1

        # 현재 학습 단계에서 사용할 decay를 계산한다.
        decay = (
            self._current_decay()
        )

        # 학습 모델과 EMA 모델의
        # 모든 Parameter와 Buffer를 가져온다.
        model_state = (
            model.state_dict()
        )

        ema_state = (
            self.ema.state_dict()
        )

        # 두 모델의 구조가 다르면
        # 일부 Parameter가 조용히 누락될 수 있으므로 중단한다.
        if (
            model_state.keys()
            != ema_state.keys()
        ):
            raise RuntimeError(
                "학습 모델과 EMA 모델의 "
                "state_dict 구조가 다릅니다."
            )

        # 같은 이름의 Parameter와 Buffer를 순회한다.
        for (
            key,
            ema_value,
        ) in ema_state.items():

            # EMA 계산에는 gradient가 필요하지 않으므로
            # 학습 모델 값을 계산 그래프에서 분리한다.
            model_value = (
                model_state[key]
                .detach()
            )

            # float Parameter와 Buffer는
            # 지수 이동평균을 계산한다.
            if torch.is_floating_point(
                ema_value
            ):
                # EMA 공식:
                #
                # new_ema =
                #     old_ema × decay
                #     + current_model × (1 - decay)
                ema_value.mul_(
                    decay
                ).add_(
                    model_value.to(
                        device=(
                            ema_value.device
                        ),
                        dtype=(
                            ema_value.dtype
                        ),
                    ),
                    alpha=(
                        1.0
                        - decay
                    ),
                )

            else:
                # BatchNorm의 num_batches_tracked 같은
                # 정수형 Buffer는 평균을 계산할 수 없다.
                #
                # 이런 값은 현재 모델 값을 그대로 복사한다.
                ema_value.copy_(
                    model_value.to(
                        device=(
                            ema_value.device
                        ),
                    )
                )

    def state_dict(self):
        """
        EMA 모델과 상태를
        checkpoint에 저장할 딕셔너리로 반환한다.
        """

        return {
            # EMA 모델 Parameter 및 Buffer
            "ema_state_dict": (
                self.ema.state_dict()
            ),

            # EMA 계산에 사용한 최대 decay
            "decay": self.decay,

            # 초기 decay 증가 속도
            "tau": self.tau,

            # 지금까지 실행한 EMA update 수
            "updates": self.updates,
        }

    def load_state_dict(
        self,
        state_dict,
    ):
        """
        checkpoint에서 EMA Parameter와
        갱신 상태를 복원한다.
        """

        if not isinstance(
            state_dict,
            dict,
        ):
            raise TypeError(
                "EMA state_dict는 dict여야 합니다."
            )

        # EMA를 완전하게 복원하는 데 필요한 key
        required_keys = {
            "ema_state_dict",
            "decay",
            "tau",
            "updates",
        }

        # checkpoint에 없는 key를 확인한다.
        missing_keys = (
            required_keys
            - state_dict.keys()
        )

        if missing_keys:
            raise KeyError(
                "EMA checkpoint에 "
                "필요한 값이 없습니다: "
                f"{sorted(missing_keys)}"
            )

        # 저장된 기본 자료형을
        # 현재 코드가 사용하는 자료형으로 변환한다.
        decay = float(
            state_dict["decay"]
        )

        tau = float(
            state_dict["tau"]
        )

        updates = int(
            state_dict["updates"]
        )

        # 손상되거나 잘못된 checkpoint 값을 검사한다.
        if not (
            0.0
            < decay
            < 1.0
        ):
            raise ValueError(
                "저장된 EMA decay가 "
                "올바르지 않습니다."
            )

        if (
            tau <= 0
            or updates < 0
        ):
            raise ValueError(
                "저장된 EMA tau 또는 updates가 "
                "올바르지 않습니다."
            )

        # EMA 모델 구조가 다르면 strict=True에 의해
        # missing key 또는 unexpected key 오류가 발생한다.
        self.ema.load_state_dict(
            state_dict[
                "ema_state_dict"
            ],
            strict=True,
        )

        # EMA 계산 상태를 복원한다.
        self.decay = decay
        self.tau = tau
        self.updates = updates