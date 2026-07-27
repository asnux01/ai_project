#!/usr/bin/env python3

"""
cam_input.py

GStreamer를 이용하여 USB 웹캠으로부터 프레임을 받아오는 입력 모듈.

데이터 흐름:

    USB 웹캠
        ↓
    /dev/video0
        ↓
    v4l2src
        ↓
    MJPG 1280x720, 30 FPS
        ↓
    jpegdec
        ↓
    videoconvert
        ↓
    BGR 영상
        ↓
    appsink
        ↓
    Python NumPy 배열

이 모듈은 화면 출력을 담당하지 않는다.
카메라에서 프레임을 받아 Python으로 전달하는 역할만 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

# PyGObject를 통해 GStreamer Python API를 사용한다.
import gi

# 사용할 GStreamer API 버전을 지정한다.
# from gi.repository import Gst보다 먼저 실행해야 한다.
gi.require_version("Gst", "1.0")

from gi.repository import Gst

# GStreamer 버퍼를 영상 배열로 변환하기 위해 사용한다.
import numpy as np


# GStreamer 라이브러리를 초기화한다.
#
# 프로그램에서 GStreamer 기능을 사용하기 전에 실행해야 한다.
# 여러 모듈에서 호출해도 문제없이 사용할 수 있다.
Gst.init(None)


# ============================================================
# 1. 프레임 자료구조
# ============================================================

@dataclass(frozen=True)
class CameraFrame:
    """
    카메라에서 가져온 프레임과 관련 정보를 하나로 묶는다.

    frozen=True:
        객체가 생성된 후 sequence나 frame 참조를
        다른 값으로 변경하지 못하게 한다.
    """

    # Python에서 성공적으로 꺼낸 프레임의 순서 번호
    #
    # 첫 번째 프레임은 1, 두 번째는 2처럼 증가한다.
    sequence: int

    # appsink에서 프레임을 꺼낸 직후의 Python 시각
    #
    # time.monotonic_ns()를 사용하며 단위는 나노초다.
    # 나중에 Python 내부 처리 시간을 측정하는 데 사용한다.
    received_time_ns: int

    # GStreamer가 프레임에 기록한 PTS
    #
    # PTS:
    #   Presentation Timestamp
    #
    # 프레임이 언제 표시되어야 하는지를 나타내는 시간이다.
    # PTS가 없는 경우에는 None이다.
    gst_pts_ns: Optional[int]

    # 실제 영상 데이터
    #
    # 자료형:
    #   numpy.ndarray
    #
    # 배열 형태:
    #   (720, 1280, 3)
    #
    # 데이터 형식:
    #   uint8
    #
    # 색상 채널 순서:
    #   BGR
    frame: np.ndarray


# ============================================================
# 2. GStreamer 카메라 입력 클래스
# ============================================================

class GStreamerCameraInput:
    """
    GStreamer appsink를 사용해 웹캠 프레임을 받아오는 클래스.

    주요 기능:
        1. /dev/video0 열기
        2. MJPG, 해상도, FPS 설정
        3. MJPG 프레임 압축 해제
        4. BGR 영상으로 변환
        5. appsink를 통해 NumPy 배열 반환
        6. 카메라 종료

    appsink는 내부적으로 프레임 큐를 가진다.

    이 코드에서는:

        max-buffers=1
        drop=true

    를 적용해 Python 처리가 느려졌을 때 오래된 프레임이
    계속 쌓이지 않도록 한다.
    """

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        """
        입력 파이프라인을 구성한다.

        이 시점에는 카메라가 아직 켜지지 않는다.
        start()를 호출해야 카메라가 실행된다.
        """

        # 사용할 카메라 장치
        self.device = device

        # 카메라 입력 크기
        self.width = width
        self.height = height

        # 카메라 입력 FPS
        self.fps = fps

        # 지금까지 Python이 받은 프레임 수
        self._sequence = 0

        # 현재 파이프라인 실행 여부
        self._running = False

        # BGR 영상 한 장의 예상 바이트 크기
        #
        # BGR은 한 픽셀당 3바이트를 사용한다.
        #
        # 1280 × 720 × 3 = 2,764,800바이트
        self._expected_frame_bytes = (
            self.width
            * self.height
            * 3
        )

        # ----------------------------------------------------
        # GStreamer 입력 파이프라인
        # ----------------------------------------------------
        #
        # v4l2src:
        #   /dev/video0 웹캠에서 데이터를 받는다.
        #
        # image/jpeg:
        #   웹캠이 MJPG 형식으로 데이터를 보낸다는 뜻이다.
        #
        # jpegdec:
        #   각 MJPG 프레임의 JPEG 압축을 해제한다.
        #
        # videoconvert:
        #   디코딩된 영상을 원하는 색상 형식으로 변환한다.
        #
        # video/x-raw,format=BGR:
        #   Python에서 사용하기 편하도록 BGR로 고정한다.
        #
        # queue:
        #   입력 처리와 appsink 처리를 분리한다.
        #
        # appsink:
        #   GStreamer 프레임을 Python이 꺼낼 수 있게 한다.
        pipeline_description = (
            f"v4l2src "
            f"device={self.device} "
            f"do-timestamp=true ! "

            f"image/jpeg,"
            f"width={self.width},"
            f"height={self.height},"
            f"framerate={self.fps}/1 ! "

            f"jpegdec ! "

            f"videoconvert ! "

            f"video/x-raw,"
            f"format=BGR,"
            f"width={self.width},"
            f"height={self.height},"
            f"framerate={self.fps}/1 ! "

            f"queue "
            f"max-size-buffers=1 "
            f"max-size-bytes=0 "
            f"max-size-time=0 "
            f"leaky=downstream ! "

            f"appsink "
            f"name=camera_sink "
            f"sync=false "
            f"max-buffers=1 "
            f"drop=true "
            f"wait-on-eos=false"
        )

        try:
            # 문자열로 작성한 파이프라인을
            # 실제 GStreamer 파이프라인 객체로 만든다.
            self._pipeline = Gst.parse_launch(
                pipeline_description
            )

        except Exception as error:
            raise RuntimeError(
                "GStreamer 입력 파이프라인을 만들지 못했습니다.\n"
                f"파이프라인: {pipeline_description}\n"
                f"원인: {error}"
            ) from error

        # 이름이 camera_sink인 appsink 요소를 가져온다.
        self._appsink = self._pipeline.get_by_name(
            "camera_sink"
        )

        if self._appsink is None:
            raise RuntimeError(
                "입력 파이프라인에서 appsink를 찾지 못했습니다."
            )

        # bus는 파이프라인에서 발생한 오류와
        # 스트림 종료 메시지를 전달한다.
        self._bus = self._pipeline.get_bus()

    def start(self) -> None:
        """
        입력 파이프라인을 실행해 웹캠을 켠다.
        """

        if self._running:
            raise RuntimeError(
                "입력 파이프라인이 이미 실행 중입니다."
            )

        self._sequence = 0

        # 파이프라인을 PLAYING 상태로 변경한다.
        #
        # 이 시점부터:
        #   /dev/video0 열기
        #   카메라 스트리밍 시작
        #   MJPG 디코딩 시작
        state_result = self._pipeline.set_state(
            Gst.State.PLAYING
        )

        if state_result == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(
                Gst.State.NULL
            )

            raise RuntimeError(
                f"웹캠 입력을 시작하지 못했습니다: "
                f"{self.device}"
            )

        # 비동기 상태 전환이 완료될 때까지 최대 5초 기다린다.
        _result, current_state, _pending_state = (
            self._pipeline.get_state(
                5 * Gst.SECOND
            )
        )

        if current_state != Gst.State.PLAYING:
            self._pipeline.set_state(
                Gst.State.NULL
            )

            raise RuntimeError(
                "입력 파이프라인이 PLAYING 상태에 "
                "도달하지 못했습니다."
            )

        self._running = True

        print("[입력] 웹캠 시작")
        print(f"[입력] 장치: {self.device}")
        print(
            f"[입력] 설정: MJPG "
            f"{self.width}x{self.height} "
            f"{self.fps} FPS"
        )

    def _check_bus_error(self) -> None:
        """
        입력 파이프라인에서 오류나 EOS가 발생했는지 확인한다.
        """

        message = self._bus.timed_pop_filtered(
            0,
            Gst.MessageType.ERROR
            | Gst.MessageType.EOS,
        )

        if message is None:
            return

        # GStreamer 오류 처리
        if message.type == Gst.MessageType.ERROR:
            error, debug_information = (
                message.parse_error()
            )

            raise RuntimeError(
                "[입력] GStreamer 오류\n"
                f"내용: {error}\n"
                f"디버그 정보: {debug_information}"
            )

        # EOS는 End Of Stream을 의미한다.
        if message.type == Gst.MessageType.EOS:
            raise RuntimeError(
                "[입력] 카메라 스트림이 종료되었습니다."
            )

    def read(
        self,
        timeout_ms: int = 1000,
    ) -> Optional[CameraFrame]:
        """
        appsink에서 카메라 프레임 한 장을 가져온다.

        매개변수:
            timeout_ms:
                새 프레임을 기다릴 최대 시간

        반환값:
            CameraFrame:
                프레임을 정상적으로 받았을 때

            None:
                제한 시간 동안 새 프레임이 없을 때
        """

        if not self._running:
            raise RuntimeError(
                "입력 파이프라인이 실행 중이 아닙니다."
            )

        # 현재까지 발생한 GStreamer 오류 확인
        self._check_bus_error()

        # 밀리초를 GStreamer 나노초 단위로 변환한다.
        timeout_ns = int(
            timeout_ms * Gst.MSECOND
        )

        # try-pull-sample:
        #   appsink에서 프레임을 꺼낸다.
        #
        # timeout 안에 프레임이 없으면 None을 반환한다.
        # 프로그램이 카메라를 무한히 기다리는 것을 방지한다.
        sample = self._appsink.emit(
            "try-pull-sample",
            timeout_ns,
        )

        if sample is None:
            self._check_bus_error()
            return None

        # ----------------------------------------------------
        # 프레임 형식 정보 확인
        # ----------------------------------------------------

        caps = sample.get_caps()

        if caps is None or caps.get_size() == 0:
            raise RuntimeError(
                "[입력] 프레임 caps 정보를 받지 못했습니다."
            )

        structure = caps.get_structure(0)

        actual_width = int(
            structure.get_value("width")
        )

        actual_height = int(
            structure.get_value("height")
        )

        actual_format = structure.get_value(
            "format"
        )

        if (
            actual_width != self.width
            or actual_height != self.height
        ):
            raise RuntimeError(
                "[입력] 예상과 다른 프레임 크기입니다: "
                f"{actual_width}x{actual_height}"
            )

        if actual_format != "BGR":
            raise RuntimeError(
                "[입력] 예상과 다른 픽셀 형식입니다: "
                f"{actual_format}"
            )

        # ----------------------------------------------------
        # GStreamer 영상 버퍼 가져오기
        # ----------------------------------------------------

        buffer = sample.get_buffer()

        if buffer is None:
            raise RuntimeError(
                "[입력] sample에서 buffer를 얻지 못했습니다."
            )

        # 버퍼를 읽기 전용으로 메모리에 연결한다.
        map_success, map_information = buffer.map(
            Gst.MapFlags.READ
        )

        if not map_success:
            raise RuntimeError(
                "[입력] GStreamer buffer map에 실패했습니다."
            )

        try:
            # GStreamer 바이트 데이터를 uint8 NumPy 배열로 해석한다.
            raw_array = np.frombuffer(
                map_information.data,
                dtype=np.uint8,
            )

            if (
                raw_array.size
                < self._expected_frame_bytes
            ):
                raise RuntimeError(
                    "[입력] 프레임 데이터가 예상보다 작습니다: "
                    f"{raw_array.size} "
                    f"< {self._expected_frame_bytes}"
                )

            # 1차원 바이트 배열을 영상 형태로 변환한다.
            #
            # 결과:
            #   (720, 1280, 3)
            #
            # copy()가 필요한 이유:
            #   buffer.unmap() 이후에도 frame을 사용해야 하기 때문이다.
            frame = raw_array[
                :self._expected_frame_bytes
            ].reshape(
                self.height,
                self.width,
                3,
            ).copy()

        finally:
            # GStreamer 버퍼의 메모리 연결을 반드시 해제한다.
            buffer.unmap(
                map_information
            )

        self._sequence += 1

        # GStreamer PTS 확인
        if buffer.pts == Gst.CLOCK_TIME_NONE:
            gst_pts_ns = None
        else:
            gst_pts_ns = int(buffer.pts)

        return CameraFrame(
            sequence=self._sequence,
            received_time_ns=time.monotonic_ns(),
            gst_pts_ns=gst_pts_ns,
            frame=frame,
        )

    def stop(self) -> None:
        """
        카메라 입력 파이프라인을 종료한다.
        """

        if not self._running:
            return

        # NULL 상태로 변경하면:
        #   카메라 스트리밍 중지
        #   /dev/video0 반환
        #   디코더 및 버퍼 자원 해제
        self._pipeline.set_state(
            Gst.State.NULL
        )

        self._running = False

        print("[입력] 웹캠 종료")