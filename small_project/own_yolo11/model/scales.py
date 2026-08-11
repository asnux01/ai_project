"""YOLO11 모델 크기별 스케일 설정.

학습 코드와 추론 코드에서는 depth/width 값을 직접 전달하지 않고
'n', 's', 'm', 'l', 'x' 중 하나의 이름만 선택한다.
"""


# 각 항목은 다음 순서로 구성된다.
#
# (
#     depth_factor,
#     width_factor,
#     max_channels,
# )
#
# 한 파일에서 관리해 학습과 추론이 같은 설정을 사용하게 한다.
YOLO11_SCALES = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


def normalize_scale_name(scale):
    """스케일 이름을 검사하고 소문자로 정규화한다."""

    # 숫자나 None이 전달되면 잘못된 자료형임을 알린다.
    if not isinstance(
        scale,
        str,
    ):
        raise TypeError(
            "scale은 'n', 's', 'm', 'l', 'x' 중 "
            "하나인 문자열이어야 합니다."
        )

    # 공백을 제거하고 소문자로 변환한다.
    scale = (
        scale.strip().lower()
    )

    # 정의되지 않은 스케일이면 사용 가능한 목록을 출력한다.
    if scale not in YOLO11_SCALES:
        available = ", ".join(
            YOLO11_SCALES.keys()
        )

        raise ValueError(
            f"지원하지 않는 scale입니다: {scale!r}. "
            f"사용 가능: {available}"
        )

    return scale


def get_scale_factors(scale):
    """선택한 스케일의 depth, width, max_channels를 반환한다."""

    # 딕셔너리를 조회하기 전에 이름을 검사한다.
    scale = normalize_scale_name(
        scale
    )

    return YOLO11_SCALES[
        scale
    ]