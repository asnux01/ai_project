#!/usr/bin/env python3

"""
main.py

GStreamer 카메라 입력 모듈과 화면 출력 모듈을 연결한다.

현재 구조:

    cam_input.py
        ↓
    NumPy BGR 프레임
        ↓
    process_frame()
        ↓
    display_output.py
        ↓
    ximagesink 화면 출력

현재 process_frame()에서는 아무런 영상 처리나
YOLO 추론을 하지 않고 원본 프레임을 그대로 반환한다.

나중에는 process_frame() 또는 별도 추론 모듈 위치에
다음 처리를 추가할 수 있다.

    NumPy 프레임
        ↓
    YOLO 전처리
        ↓
    PyTorch Tensor
        ↓
    YOLO11 추론
        ↓
    검출 박스 표시
        ↓
    NumPy 출력 프레임
"""

from __future__ import annotations

import time

import numpy as np

# camera_input.py의 이름을 cam_input.py로 변경했으므로
# import 대상도 cam_input으로 변경한다.
from cam_input import GStreamerCameraInput

from display_output import GStreamerDisplayOutput


# ============================================================
# 1. 공통 영상 설정
# ============================================================

# 실제 USB 웹캠 영상 장치
CAMERA_DEVICE = "/dev/video0"

# 입력 및 출력 해상도
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 목표 카메라 FPS
FRAME_RATE = 30

# 현재 DISPLAY=:1 환경에서는 NV-GLX 확장이 없기 때문에
# autovideosink 대신 일반 X11 출력 요소를 사용한다.
VIDEO_SINK = "ximagesink"


# ============================================================
# 2. 중간 처리 함수
# ============================================================

def process_frame(
    frame: np.ndarray,
) -> np.ndarray:
    """
    입력 프레임을 처리하는 함수다.

    현재 동작:
        입력 프레임을 아무런 수정 없이 그대로 반환한다.

    나중의 동작 예:

        1. NumPy BGR 프레임 전처리
        2. PyTorch Tensor 변환
        3. YOLO11 추론
        4. 검출 좌표 후처리
        5. 프레임에 Bounding Box 표시
        6. 화면 출력용 NumPy BGR 프레임 반환

    매개변수
    --------------------------------------------------------
    frame:
        카메라 입력 모듈에서 받은 BGR NumPy 프레임

        shape:
            (720, 1280, 3)

        dtype:
            uint8

    반환값
    --------------------------------------------------------
    현재는 입력 frame을 그대로 반환한다.
    """

    return frame


# ============================================================
# 3. 메인 실행 함수
# ============================================================

def main() -> None:
    """
    GStreamer 입력과 출력을 실행하고
    카메라 프레임을 화면으로 전달한다.
    """

    # --------------------------------------------------------
    # 카메라 입력 객체
    # --------------------------------------------------------
    #
    # cam_input.py 내부의 입력 파이프라인:
    #
    #   v4l2src
    #   → MJPG
    #   → jpegdec
    #   → videoconvert
    #   → BGR
    #   → appsink
    #   → NumPy
    camera = GStreamerCameraInput(
        device=CAMERA_DEVICE,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=FRAME_RATE,
    )

    # --------------------------------------------------------
    # 화면 출력 객체
    # --------------------------------------------------------
    #
    # display_output.py 내부의 출력 파이프라인:
    #
    #   NumPy
    #   → appsrc
    #   → queue
    #   → videoconvert
    #   → ximagesink
    display = GStreamerDisplayOutput(
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=FRAME_RATE,

        # NV-GLX 오류를 방지하기 위해
        # ximagesink를 명시적으로 사용한다.
        video_sink=VIDEO_SINK,
    )

    # FPS 측정 시작 시각
    report_start_time = time.perf_counter()

    # 현재 측정 구간에서 출력한 프레임 수
    report_frame_count = 0

    # Python 구간 처리 시간 누적값
    accumulated_bridge_time_ms = 0.0

    # 지금까지 전달한 전체 프레임 수
    total_frame_count = 0

    try:
        # ----------------------------------------------------
        # 1. 출력 파이프라인 시작
        # ----------------------------------------------------
        #
        # 카메라 프레임이 도착하자마자 appsrc로 넣을 수 있도록
        # 출력 파이프라인을 먼저 준비한다.
        #
        # appsrc는 첫 프레임이 들어오기 전까지
        # 비동기 상태 전환 결과를 반환할 수 있다.
        display.start()

        # ----------------------------------------------------
        # 2. 입력 파이프라인 시작
        # ----------------------------------------------------
        #
        # 이 시점부터 /dev/video0이 열리고
        # 웹캠 LED가 켜진다.
        camera.start()

        print()
        print("=" * 60)
        print("GStreamer 웹캠 입력 → Python → 화면 출력")
        print("=" * 60)
        print(f"카메라 장치: {CAMERA_DEVICE}")
        print(
            f"영상 설정: "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT}, "
            f"{FRAME_RATE} FPS"
        )
        print(f"출력 요소: {VIDEO_SINK}")
        print("현재 YOLO 추론: 사용하지 않음")
        print("종료 방법: 터미널에서 Ctrl+C")
        print()

        while True:
            # ------------------------------------------------
            # 3. 카메라 프레임 입력
            # ------------------------------------------------
            #
            # appsink에서 카메라 프레임 한 장을 가져온다.
            #
            # 1초 동안 프레임이 없으면 None을 반환한다.
            packet = camera.read(
                timeout_ms=1000
            )

            if packet is None:
                print(
                    "[경고] 1초 동안 새로운 "
                    "카메라 프레임이 없습니다."
                )
                continue

            # 카메라가 전달한 실제 NumPy BGR 프레임
            input_frame = packet.frame

            # ------------------------------------------------
            # 4. 중간 영상 처리
            # ------------------------------------------------
            #
            # 현재는 아무런 처리를 하지 않고
            # 입력 프레임을 그대로 반환한다.
            #
            # 나중에는 이 위치에 YOLO11 추론 모듈을
            # 연결하게 된다.
            output_frame = process_frame(
                input_frame
            )

            # process_frame()이 실수로 NumPy 배열이 아닌
            # 다른 값을 반환했는지 확인한다.
            if not isinstance(
                output_frame,
                np.ndarray,
            ):
                raise TypeError(
                    "process_frame()은 NumPy 배열을 "
                    "반환해야 합니다.\n"
                    f"현재 타입: {type(output_frame)}"
                )

            # ------------------------------------------------
            # 5. 화면 출력
            # ------------------------------------------------
            #
            # NumPy BGR 프레임을 appsrc에 넣는다.
            #
            # 이후 프레임은 다음 경로로 이동한다.
            #
            #   appsrc
            #   → queue
            #   → videoconvert
            #   → ximagesink
            display.show(
                output_frame
            )

            report_frame_count += 1
            total_frame_count += 1

            # ------------------------------------------------
            # 6. Python 전달 구간 시간 측정
            # ------------------------------------------------
            #
            # packet.received_time_ns는 cam_input.py에서
            # appsink 프레임을 Python으로 가져온 직후 시각이다.
            #
            # 따라서 아래 값은 대략 다음 구간의 시간이다.
            #
            #   appsink에서 프레임 수신
            #   → process_frame()
            #   → appsrc에 push-buffer 완료
            bridge_time_ms = (
                time.monotonic_ns()
                - packet.received_time_ns
            ) / 1_000_000.0

            accumulated_bridge_time_ms += (
                bridge_time_ms
            )

            current_time = time.perf_counter()

            elapsed_time = (
                current_time
                - report_start_time
            )

            # ------------------------------------------------
            # 7. 약 1초마다 성능 정보 출력
            # ------------------------------------------------
            #
            # 매 프레임마다 print하면 터미널 출력 자체가
            # 성능을 떨어뜨릴 수 있으므로 1초마다 출력한다.
            if elapsed_time >= 1.0:
                transfer_fps = (
                    report_frame_count
                    / elapsed_time
                )

                if report_frame_count > 0:
                    average_bridge_time_ms = (
                        accumulated_bridge_time_ms
                        / report_frame_count
                    )
                else:
                    average_bridge_time_ms = 0.0

                print(
                    "[상태] "
                    f"전달 FPS={transfer_fps:.2f}, "
                    f"Python 구간 평균="
                    f"{average_bridge_time_ms:.2f} ms, "
                    f"전체 프레임={total_frame_count}, "
                    f"shape={output_frame.shape}"
                )

                # 다음 측정 구간을 위해 초기화한다.
                report_start_time = current_time
                report_frame_count = 0
                accumulated_bridge_time_ms = 0.0

    except KeyboardInterrupt:
        print()
        print("[종료] Ctrl+C가 입력되었습니다.")

    except (
        RuntimeError,
        ValueError,
        TypeError,
    ) as error:
        print()
        print("[실행 오류]")
        print(error)

    finally:
        # ----------------------------------------------------
        # 8. 입력 및 출력 파이프라인 종료
        # ----------------------------------------------------
        #
        # 입력을 먼저 종료해 새로운 프레임 유입을 막는다.
        camera.stop()

        # 출력 파이프라인을 종료한다.
        display.stop()

        print(
            f"[프로그램 종료] "
            f"전체 전달 프레임={total_frame_count}"
        )


# ============================================================
# 4. 파일 직접 실행
# ============================================================

if __name__ == "__main__":
    main()