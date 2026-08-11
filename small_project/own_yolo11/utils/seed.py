"""학습 결과의 재현성을 높이기 위한 난수 설정 함수"""
# 라이브러리 
import os
import random
import numpy as np
import torch


def set_seed(
    seed,
    deterministic=False,
):
    """
    Python, NumPy, PyTorch가 사용하는
    난수 seed를 설정한다.

    Args:
        seed:
            0 이상의 정수 seed

        deterministic:
            True이면 cuDNN의 결정적 연산을 우선한다.

            재현성은 높아지지만
            학습 속도가 느려질 수 있다.
    """

    # 유효성 검사
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise TypeError(
            "seed는 정수여야 합니다."
        )

    if seed < 0:
        raise ValueError(
            "seed는 0 이상이어야 합니다."
        )

    if not isinstance(
        deterministic,
        bool
    ):
        raise TypeError(
            "deterministic은 bool이어야 합니다."
        )

    # PYTHONHASHSEED는 새로 시작되는 Python 프로세스의
    # 문자열 hash 순서를 재현하는 데 사용
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python 기본 난수 생성기 seed 설정
    random.seed(seed)

    # NumPy 난수 생성기 seed 설정
    np.random.seed(seed)

    # CPU에서 사용하는 PyTorch 난수 생성기 seed 설정
    torch.manual_seed(seed)

    # CUDA를 사용할 수 있을 때만
    # GPU 난수 생성기를 설정한다.
    if torch.cuda.is_available():

        # 현재 CUDA 장치의 seed를 설정한다.
        torch.cuda.manual_seed(seed)

        # 여러 GPU를 사용하는 경우
        # 모든 CUDA 장치의 seed를 설정한다.
        torch.cuda.manual_seed_all(seed)

    # deterministic=False일 때는 고정 입력 크기에서
    # cuDNN이 빠른 알고리즘을 선택할 수 있도록
    # benchmark를 사용한다.
    torch.backends.cudnn.deterministic = (
        deterministic
    )

    torch.backends.cudnn.benchmark = (
        not deterministic
    )


def seed_worker(worker_id):
    """
    각 DataLoader worker가 서로 다른
    재현 가능한 seed를 사용하게 한다.
    """

    # worker_id는 DataLoader가 전달하지만,
    # 실제 seed는 PyTorch가 worker마다 생성한 값을 사용한다.
    _ = worker_id

    # PyTorch의 seed는 더 큰 정수 범위를 사용할 수 있으므로
    # NumPy가 지원하는 32비트 범위로 변환한다.
    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    # 현재 worker 내부의 NumPy와 Python random이
    # PyTorch worker seed에 맞춰 동작하게 한다.
    np.random.seed(worker_seed)
    random.seed(worker_seed)