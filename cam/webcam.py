#!/usr/bin/env python3

"""
Jetson USB 웹캠 실시간 출력 프로그램

현재 확인된 웹캠 정보:
    실제 영상 장치 : /dev/video0
    픽셀 포맷      : MJPG
    입력 해상도    : 1280 x 720
    입력 FPS       : 30

프로그램 동작 순서:
    1. OpenCV를 통해 /dev/video0을 V4L2 방식으로 연다.
    2. 웹캠에 MJPG, 1280x720, 30 FPS를 요청한다.
    3. 실제로 적용된 카메라 설정을 확인한다.
    4. 카메라에서 프레임을 한 장씩 계속 읽는다.
    5. 읽은 프레임을 Jetson 모니터에 출력한다.
    6. q 또는 ESC 키를 누르면 안전하게 종료한다.

현재 코드는 카메라 영상 출력만 수행한다.
나중에는 cap.read()로 얻은 frame을 YOLO 모델에 전달하면 된다.
"""

import os
import sys
import time

import cv2


# =========================================================
# 1. 웹캠 설정값
# =========================================================

# 실제 영상 캡처 장치다.
#
# /dev/video0:
#   웹캠의 실제 영상 데이터를 제공하는 장치
#
# /dev/video1:
#   Metadata Capture 장치이므로 여기서는 사용하지 않는다.
CAMERA_DEVICE = "/dev/video0"


# 카메라가 영상을 전송할 때 사용할 픽셀 포맷이다.
#
# MJPG:
#   각 영상 프레임을 JPEG 형태로 압축해서 USB로 전송한다.
#   1280x720, 30 FPS와 같은 고해상도 실시간 입력에 적합하다.
#
# 카메라에서 MJPG로 전송하지만 OpenCV의 cap.read()를 거치면
# frame은 압축이 풀린 BGR 형식의 NumPy 배열로 전달된다.
CAMERA_FORMAT = "MJPG"


# 웹캠으로부터 받아올 영상의 너비와 높이다.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# 카메라에 요청할 초당 프레임 수다.
TARGET_FPS = 30


# OpenCV 출력 창의 이름이다.
WINDOW_NAME = "Jetson USB Webcam"


def fourcc_number_to_string(fourcc_number):
    """
    OpenCV가 숫자로 반환한 FOURCC 값을 문자열로 바꾼다.

    FOURCC는 픽셀 포맷을 네 글자로 나타낸 코드다.

    예:
        숫자로 저장된 FOURCC → "MJPG"

    매개변수:
        fourcc_number:
            cap.get(cv2.CAP_PROP_FOURCC)로 얻은 숫자 값

    반환값:
        "MJPG", "YUYV"와 같은 네 글자 문자열
    """

    # 혹시 실수형으로 반환될 수 있으므로 정수로 변환한다.
    value = int(fourcc_number)

    # 32비트 정수 안에 저장된 네 개의 문자를 한 글자씩 꺼낸다.
    character_1 = chr((value >> 0) & 0xFF)
    character_2 = chr((value >> 8) & 0xFF)
    character_3 = chr((value >> 16) & 0xFF)
    character_4 = chr((value >> 24) & 0xFF)

    return (
        character_1
        + character_2
        + character_3
        + character_4
    )


def open_camera():
    """
    USB 웹캠을 열고 영상 포맷, 해상도, FPS를 설정한다.

    반환값:
        정상적으로 열린 cv2.VideoCapture 객체

    실패하는 경우:
        None 반환
    """

    # -----------------------------------------------------
    # 카메라 장치 파일 존재 여부 확인
    # -----------------------------------------------------
    if not os.path.exists(CAMERA_DEVICE):
        print(
            f"[오류] 카메라 장치가 존재하지 않습니다: "
            f"{CAMERA_DEVICE}"
        )
        print("ls -l /dev/video* 명령으로 장치를 확인하세요.")

        return None

    # -----------------------------------------------------
    # V4L2 백엔드를 사용해 웹캠 열기
    # -----------------------------------------------------
    #
    # 첫 번째 인자:
    #   사용할 카메라 장치 경로
    #
    # 두 번째 인자:
    #   Linux V4L2 백엔드를 명시적으로 사용
    #
    # OpenCV가 임의의 영상 백엔드를 선택하게 두지 않고,
    # USB 웹캠이 연결된 V4L2 경로를 사용하도록 지정한다.
    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    # isOpened()는 카메라 장치를 정상적으로 열었는지 확인한다.
    if not cap.isOpened():
        print(
            f"[오류] 카메라를 열지 못했습니다: "
            f"{CAMERA_DEVICE}"
        )
        print("다음을 확인하세요.")
        print("1. 현재 사용자가 video 그룹에 포함되어 있는지")
        print("2. 다른 프로그램이 카메라를 사용 중인지")
        print("3. /dev/video0이 존재하는지")

        return None

    # -----------------------------------------------------
    # 카메라 픽셀 포맷 설정
    # -----------------------------------------------------
    #
    # VideoWriter_fourcc(*"MJPG")는 문자열 MJPG를
    # OpenCV가 사용하는 FOURCC 숫자로 변환한다.
    mjpg_fourcc = cv2.VideoWriter_fourcc(
        *CAMERA_FORMAT
    )

    # 웹캠에 MJPG 포맷 사용을 요청한다.
    format_success = cap.set(
        cv2.CAP_PROP_FOURCC,
        mjpg_fourcc
    )

    # -----------------------------------------------------
    # 카메라 해상도 설정
    # -----------------------------------------------------

    # 웹캠 영상 너비를 1280으로 요청한다.
    width_success = cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    # 웹캠 영상 높이를 720으로 요청한다.
    height_success = cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    # -----------------------------------------------------
    # 카메라 FPS 설정
    # -----------------------------------------------------

    # 웹캠에 초당 30프레임을 요청한다.
    fps_success = cap.set(
        cv2.CAP_PROP_FPS,
        TARGET_FPS
    )

    # -----------------------------------------------------
    # 카메라 내부 버퍼 크기 설정
    # -----------------------------------------------------
    #
    # 카메라에서 들어온 오래된 프레임이 버퍼에 많이 쌓이면
    # 실제 움직임보다 화면이 늦게 표시될 수 있다.
    #
    # 버퍼를 1프레임으로 요청하면 지연을 줄이는 데 도움이 된다.
    # 단, OpenCV 또는 드라이버에 따라 이 설정은 무시될 수 있다.
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    print("[카메라 설정 요청 결과]")
    print(
        f"MJPG 설정 요청: "
        f"{'성공' if format_success else '실패 또는 미지원'}"
    )
    print(
        f"너비 {FRAME_WIDTH} 설정 요청: "
        f"{'성공' if width_success else '실패 또는 미지원'}"
    )
    print(
        f"높이 {FRAME_HEIGHT} 설정 요청: "
        f"{'성공' if height_success else '실패 또는 미지원'}"
    )
    print(
        f"FPS {TARGET_FPS} 설정 요청: "
        f"{'성공' if fps_success else '실패 또는 미지원'}"
    )

    return cap


def print_actual_camera_settings(cap):
    """
    카메라에 실제로 적용된 설정을 확인해서 출력한다.

    cap.set()은 카메라에 설정을 요청하는 함수다.
    요청이 성공한 것처럼 보여도 카메라가 다른 값을 적용할 수 있다.

    따라서 cap.get()으로 실제 값을 반드시 다시 확인해야 한다.
    """

    # 실제로 적용된 영상 너비를 읽는다.
    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    # 실제로 적용된 영상 높이를 읽는다.
    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # 실제로 적용된 FPS를 읽는다.
    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    # 실제로 적용된 FOURCC 픽셀 포맷을 읽는다.
    actual_fourcc_number = cap.get(
        cv2.CAP_PROP_FOURCC
    )

    actual_fourcc_string = fourcc_number_to_string(
        actual_fourcc_number
    )

    # OpenCV가 어떤 영상 백엔드를 사용하는지 확인한다.
    backend_name = "확인할 수 없음"

    try:
        backend_name = cap.getBackendName()
    except cv2.error:
        # 일부 OpenCV 버전에서는 getBackendName()이
        # 제대로 지원되지 않을 수 있다.
        pass

    print("\n[실제 적용된 카메라 설정]")
    print(f"장치 경로: {CAMERA_DEVICE}")
    print(f"OpenCV 백엔드: {backend_name}")
    print(f"픽셀 포맷: {actual_fourcc_string}")
    print(f"해상도: {actual_width} x {actual_height}")
    print(f"카메라 보고 FPS: {actual_fps:.2f}")

    # 요청한 값과 실제 값이 다른 경우 경고를 출력한다.
    if actual_width != FRAME_WIDTH:
        print(
            f"[경고] 요청한 너비 {FRAME_WIDTH}과 "
            f"실제 너비 {actual_width}이 다릅니다."
        )

    if actual_height != FRAME_HEIGHT:
        print(
            f"[경고] 요청한 높이 {FRAME_HEIGHT}과 "
            f"실제 높이 {actual_height}이 다릅니다."
        )

    if actual_fourcc_string != CAMERA_FORMAT:
        print(
            f"[경고] 요청한 포맷 {CAMERA_FORMAT}과 "
            f"실제 포맷 {actual_fourcc_string}이 다릅니다."
        )


def draw_status_information(frame, display_fps):
    """
    웹캠 영상 위에 현재 해상도와 출력 FPS를 표시한다.

    매개변수:
        frame:
            cap.read()로 받은 카메라 영상 한 장

        display_fps:
            프로그램이 실제로 처리하고 출력하는 FPS
    """

    # frame.shape은 일반적으로 다음 구조다.
    #
    # (영상 높이, 영상 너비, 색상 채널 수)
    #
    # 예:
    # (720, 1280, 3)
    frame_height, frame_width = frame.shape[:2]

    resolution_text = (
        f"Frame: {frame_width}x{frame_height}"
    )

    fps_text = (
        f"Display FPS: {display_fps:.1f}"
    )

    # 영상 위에 해상도를 표시한다.
    cv2.putText(
        frame,                         # 글자를 그릴 영상
        resolution_text,               # 표시할 문자열
        (20, 35),                      # 글자 시작 좌표
        cv2.FONT_HERSHEY_SIMPLEX,       # 글꼴
        0.8,                           # 글자 크기
        (0, 255, 0),                   # BGR 기준 초록색
        2,                             # 글자 두께
        cv2.LINE_AA                    # 부드러운 글자 테두리
    )

    # 영상 위에 실제 출력 FPS를 표시한다.
    cv2.putText(
        frame,
        fps_text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


def main():
    """
    프로그램 전체 실행 함수
    """

    print("[Jetson USB 웹캠 프로그램 시작]")

    # -----------------------------------------------------
    # 1. 카메라 열기 및 설정
    # -----------------------------------------------------
    cap = open_camera()

    # open_camera()가 None을 반환하면 카메라 열기에 실패한 것이다.
    if cap is None:
        sys.exit(1)

    try:
        # -------------------------------------------------
        # 2. 실제 적용된 카메라 설정 출력
        # -------------------------------------------------
        print_actual_camera_settings(cap)

        # -------------------------------------------------
        # 3. 카메라 초기 안정화
        # -------------------------------------------------
        #
        # 웹캠은 처음 열린 직후 자동 노출, 자동 화이트밸런스,
        # 자동 초점 등이 안정화되지 않았을 수 있다.
        #
        # 초기 프레임 일부를 읽고 버려서 영상을 안정화한다.
        warmup_success = False

        print("\n[카메라 초기화 중]")

        for frame_index in range(15):
            ret, frame = cap.read()

            if ret and frame is not None:
                warmup_success = True

            # 초기화 상태를 잠깐 기다린다.
            time.sleep(0.01)

        if not warmup_success:
            print("[오류] 초기 카메라 프레임을 받지 못했습니다.")
            sys.exit(1)

        # -------------------------------------------------
        # 4. 화면 출력 창 생성
        # -------------------------------------------------
        #
        # WINDOW_NORMAL:
        #   사용자가 창 크기를 조절할 수 있게 한다.
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        # 출력 창의 초기 크기를 카메라 해상도와 맞춘다.
        cv2.resizeWindow(
            WINDOW_NAME,
            FRAME_WIDTH,
            FRAME_HEIGHT
        )

        print("\n[실시간 영상 출력 시작]")
        print("종료하려면 영상 창에서 q 또는 ESC를 누르세요.")

        # FPS 계산에 사용할 이전 시간
        previous_time = time.perf_counter()

        # 화면에 표시할 FPS 값
        display_fps = 0.0

        # 연속 프레임 읽기 실패 횟수
        consecutive_failures = 0

        # -------------------------------------------------
        # 5. 카메라 영상 반복 처리
        # -------------------------------------------------
        while True:
            # 카메라에서 영상 프레임 한 장을 읽는다.
            #
            # ret:
            #   프레임 읽기 성공 여부
            #
            # frame:
            #   웹캠에서 받은 실제 이미지 데이터
            #
            # frame은 NumPy 배열이며 OpenCV에서는
            # 기본적으로 BGR 색상 순서를 사용한다.
            ret, frame = cap.read()

            # 프레임을 읽지 못한 경우
            if not ret or frame is None:
                consecutive_failures += 1

                print(
                    "[경고] 프레임 읽기 실패: "
                    f"{consecutive_failures}/10"
                )

                # 일시적인 오류가 아니라 10번 연속 실패하면 종료한다.
                if consecutive_failures >= 10:
                    print(
                        "[오류] 카메라 스트림을 "
                        "유지할 수 없습니다."
                    )
                    break

                continue

            # 프레임 읽기에 성공했으므로 실패 횟수를 초기화한다.
            consecutive_failures = 0

            # -------------------------------------------------
            # 나중에 YOLO 추론 코드를 넣을 위치
            # -------------------------------------------------
            #
            # 현재 frame에는 웹캠에서 받은 영상 한 장이 들어 있다.
            #
            # 이후에는 이 위치에 다음 과정을 추가하게 된다.
            #
            # input_tensor = preprocess(frame)
            # predictions = model(input_tensor)
            # frame = draw_detection_results(frame, predictions)
            #
            # 지금 단계에서는 YOLO 처리를 하지 않고
            # 원본 frame을 그대로 모니터에 출력한다.

            # -------------------------------------------------
            # 6. 실제 처리 FPS 계산
            # -------------------------------------------------
            current_time = time.perf_counter()
            elapsed_time = current_time - previous_time
            previous_time = current_time

            if elapsed_time > 0:
                current_fps = 1.0 / elapsed_time

                # 매 프레임 FPS 값은 조금씩 흔들릴 수 있다.
                # 이전 FPS의 90%와 현재 FPS의 10%를 섞어서
                # 화면에 표시되는 값을 부드럽게 만든다.
                if display_fps == 0.0:
                    display_fps = current_fps
                else:
                    display_fps = (
                        display_fps * 0.9
                        + current_fps * 0.1
                    )

            # 영상 위에 해상도와 FPS를 표시한다.
            draw_status_information(
                frame,
                display_fps
            )

            # -------------------------------------------------
            # 7. Jetson 모니터에 영상 출력
            # -------------------------------------------------
            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            # waitKey()의 역할:
            #
            # 1. OpenCV 창을 화면에 실제로 갱신
            # 2. 키보드 입력 확인
            #
            # 숫자 1은 최대 약 1ms 동안 키 입력을 기다린다는 뜻이다.
            key = cv2.waitKey(1) & 0xFF

            # q 키 또는 ESC 키를 누르면 반복문을 종료한다.
            if key == ord("q") or key == 27:
                print("\n[알림] 종료 키가 입력되었습니다.")
                break

    finally:
        # -------------------------------------------------
        # 8. 프로그램 종료 및 자원 반환
        # -------------------------------------------------

        # 카메라 장치를 반환한다.
        #
        # release()를 하지 않으면 프로그램 종료 후에도
        # 카메라가 사용 중인 것처럼 남을 수 있다.
        cap.release()

        # OpenCV가 생성한 모든 출력 창을 닫는다.
        cv2.destroyAllWindows()

        print("[알림] 카메라와 출력 창을 종료했습니다.")


# 이 Python 파일을 직접 실행한 경우에만 main()을 호출한다.
#
# 다른 파일에서 이 파일을 import할 때는 main()이 자동 실행되지 않는다.
if __name__ == "__main__":
    main()