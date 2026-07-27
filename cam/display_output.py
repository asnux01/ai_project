#!/usr/bin/env python3

"""
display_output.py

Python의 BGR NumPy 프레임을 GStreamer의 appsrc로 전달하여
화면에 출력하는 모듈이다.

데이터 흐름:

    NumPy BGR 프레임
        ↓
    appsrc
        ↓
    queue
        ↓
    videoconvert
        ↓
    ximagesink
        ↓
    화면 출력

수정된 핵심 내용
------------------------------------------------------------
1. autovideosink 대신 ximagesink를 사용한다.
   - 현재 DISPLAY=:1 환경에는 NV-GLX 확장이 없다.
   - autovideosink가 OpenGL 기반 출력 요소를 선택하는 것을 방지한다.

2. 출력 파이프라인이 즉시 PLAYING 상태에 도달하는지
   강제로 검사하지 않는다.
   - appsrc는 첫 프레임이 들어오기 전까지 상태 전환이
     비동기적으로 진행될 수 있다.
   - FAILURE가 반환된 경우에만 시작 실패로 처리한다.

3. appsrc에 넣는 각 프레임에 PTS와 duration을 직접 설정한다.

4. 출력이 늦어질 때 프레임이 계속 쌓이지 않도록
   queue의 크기를 한 프레임으로 제한한다.
"""

from __future__ import annotations

from typing import Optional

import gi

# 사용할 GStreamer API 버전을 지정한다.
# from gi.repository import Gst보다 먼저 실행해야 한다.
gi.require_version("Gst", "1.0")

from gi.repository import Gst

import numpy as np


# GStreamer 라이브러리 초기화
Gst.init(None)


class GStreamerDisplayOutput:
    """
    NumPy BGR 프레임을 GStreamer로 전달하여 화면에 표시한다.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        video_sink: str = "ximagesink",
    ) -> None:
        """
        출력 파이프라인을 생성한다.

        이 단계에서는 아직 화면 출력이 시작되지 않는다.
        start()를 호출해야 파이프라인이 실행된다.

        매개변수
        --------------------------------------------------------
        width:
            출력 영상 너비

        height:
            출력 영상 높이

        fps:
            출력 프레임률

        video_sink:
            사용할 GStreamer 영상 출력 요소

            현재 환경에서는 NV-GLX 오류를 피하기 위해
            ximagesink를 기본값으로 사용한다.
        """

        if width <= 0:
            raise ValueError(
                f"width는 1 이상이어야 합니다: {width}"
            )

        if height <= 0:
            raise ValueError(
                f"height는 1 이상이어야 합니다: {height}"
            )

        if fps <= 0:
            raise ValueError(
                f"fps는 1 이상이어야 합니다: {fps}"
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.video_sink = video_sink

        # 현재 출력 파이프라인이 실행 중인지 나타낸다.
        self._running = False

        # 다음 프레임에 사용할 PTS
        #
        # 첫 번째 프레임은 0부터 시작한다.
        self._next_pts_ns = 0

        # 한 프레임의 재생 시간
        #
        # 예:
        #   30 FPS
        #   → 1초 / 30
        #   → 약 33.33ms
        self._frame_duration_ns = int(
            Gst.SECOND / self.fps
        )

        # BGR 영상 한 장의 바이트 크기
        #
        # BGR은 픽셀 하나당 3바이트를 사용한다.
        self._frame_bytes = (
            self.width
            * self.height
            * 3
        )

        # ----------------------------------------------------
        # GStreamer 출력 파이프라인
        # ----------------------------------------------------
        #
        # appsrc:
        #   Python이 생성하거나 처리한 프레임을
        #   GStreamer 파이프라인에 넣는다.
        #
        # is-live=true:
        #   파일이 아니라 실시간 영상 소스로 취급한다.
        #
        # block=true:
        #   appsrc의 내부 버퍼가 가득 찼을 때
        #   무한히 데이터를 쌓지 않고 잠시 기다린다.
        #
        # format=time:
        #   버퍼의 PTS와 duration을 시간 단위로 해석한다.
        #
        # queue:
        #   appsrc 처리와 화면 출력을 별도 실행 흐름으로 분리한다.
        #
        # max-size-buffers=1:
        #   queue에 영상 한 장만 유지한다.
        #
        # leaky=downstream:
        #   출력이 늦어 queue가 가득 차면
        #   오래된 프레임을 버린다.
        #
        # videoconvert:
        #   BGR 프레임을 ximagesink가 받을 수 있는
        #   영상 형식으로 변환한다.
        #
        # ximagesink:
        #   OpenGL이 아닌 X11 방식으로 화면을 출력한다.
        #
        # sync=false:
        #   GStreamer Clock을 기다리지 않고
        #   프레임이 도착하면 가능한 한 빨리 표시한다.
        #
        # async=false:
        #   첫 프레임이 들어오기 전 상태 전환 때문에
        #   파이프라인 시작이 지연되는 것을 줄인다.
        pipeline_description = (
            "appsrc "
            "name=display_src "
            "is-live=true "
            "block=true "
            "format=time "
            "do-timestamp=false "
            "! "

            "queue "
            "max-size-buffers=1 "
            "max-size-bytes=0 "
            "max-size-time=0 "
            "leaky=downstream "
            "! "

            "videoconvert "
            "! "

            f"{self.video_sink} "
            "sync=false "
            "async=false"
        )

        try:
            # 문자열로 작성된 파이프라인을
            # 실제 GStreamer 파이프라인 객체로 만든다.
            self._pipeline = Gst.parse_launch(
                pipeline_description
            )

        except Exception as error:
            raise RuntimeError(
                "GStreamer 출력 파이프라인을 "
                "생성하지 못했습니다.\n"
                f"파이프라인: {pipeline_description}\n"
                f"원인: {error}"
            ) from error

        # 이름이 display_src인 appsrc 요소를 가져온다.
        self._appsrc = self._pipeline.get_by_name(
            "display_src"
        )

        if self._appsrc is None:
            self._pipeline.set_state(
                Gst.State.NULL
            )

            raise RuntimeError(
                "출력 파이프라인에서 "
                "display_src appsrc를 찾지 못했습니다."
            )

        # ----------------------------------------------------
        # appsrc가 받을 입력 영상 형식 지정
        # ----------------------------------------------------
        #
        # Python은 다음 형식의 프레임을 appsrc에 넣는다.
        #
        # format:
        #   BGR
        #
        # width, height:
        #   설정된 출력 해상도
        #
        # framerate:
        #   설정된 출력 FPS
        caps_description = (
            "video/x-raw,"
            "format=BGR,"
            f"width={self.width},"
            f"height={self.height},"
            f"framerate={self.fps}/1,"
            "pixel-aspect-ratio=1/1"
        )

        caps = Gst.Caps.from_string(
            caps_description
        )

        if caps is None or caps.is_empty():
            self._pipeline.set_state(
                Gst.State.NULL
            )

            raise RuntimeError(
                "출력용 GStreamer Caps를 "
                "생성하지 못했습니다.\n"
                f"Caps: {caps_description}"
            )

        self._appsrc.set_property(
            "caps",
            caps,
        )

        # appsrc 내부에 쌓일 수 있는 데이터 크기를
        # 약 두 프레임 크기로 제한한다.
        #
        # block=true이므로 이 크기를 초과하면
        # push-buffer가 잠시 기다리게 된다.
        self._appsrc.set_property(
            "max-bytes",
            self._frame_bytes * 2,
        )

        # GStreamer 오류와 EOS 메시지를 확인하기 위한 Bus
        self._bus = self._pipeline.get_bus()

    def _get_state_result_name(
        self,
        state_result: Gst.StateChangeReturn,
    ) -> str:
        """
        StateChangeReturn 값을 사람이 읽기 쉬운 문자열로 바꾼다.
        """

        value_nick = getattr(
            state_result,
            "value_nick",
            None,
        )

        if value_nick is not None:
            return str(value_nick)

        return str(state_result)

    def _pop_bus_message(
        self,
        timeout_ns: int = 0,
    ) -> Optional[Gst.Message]:
        """
        Bus에서 ERROR 또는 EOS 메시지를 하나 가져온다.

        메시지가 없으면 None을 반환한다.
        """

        return self._bus.timed_pop_filtered(
            timeout_ns,
            Gst.MessageType.ERROR
            | Gst.MessageType.EOS,
        )

    def _check_bus_error(self) -> None:
        """
        출력 파이프라인에서 ERROR 또는 EOS가 발생했는지 확인한다.
        """

        message = self._pop_bus_message(
            timeout_ns=0
        )

        if message is None:
            return

        if message.type == Gst.MessageType.ERROR:
            error, debug_information = (
                message.parse_error()
            )

            raise RuntimeError(
                "[출력] GStreamer 오류가 발생했습니다.\n"
                f"오류 내용: {error}\n"
                f"디버그 정보: {debug_information}"
            )

        if message.type == Gst.MessageType.EOS:
            raise RuntimeError(
                "[출력] GStreamer 출력 스트림이 "
                "종료되었습니다."
            )

    def start(self) -> None:
        """
        출력 파이프라인을 시작한다.

        기존 코드와 달리 PLAYING 상태에 즉시 도달했는지는
        강제로 검사하지 않는다.

        appsrc 기반 실시간 파이프라인은 첫 번째 프레임이
        push되기 전까지 상태 전환이 비동기적으로 진행될 수 있다.

        따라서 set_state()가 FAILURE를 반환한 경우에만
        시작 실패로 처리한다.
        """

        if self._running:
            raise RuntimeError(
                "출력 파이프라인이 이미 실행 중입니다."
            )

        # 새로운 실행을 시작할 때 PTS를 다시 0부터 시작한다.
        self._next_pts_ns = 0

        state_result = self._pipeline.set_state(
            Gst.State.PLAYING
        )

        # FAILURE만 실제 시작 실패로 처리한다.
        #
        # SUCCESS, ASYNC, NO_PREROLL은
        # 정상적인 상태 전환 결과일 수 있다.
        if state_result == Gst.StateChangeReturn.FAILURE:
            message = self._pop_bus_message(
                timeout_ns=Gst.SECOND
            )

            self._pipeline.set_state(
                Gst.State.NULL
            )

            if (
                message is not None
                and message.type == Gst.MessageType.ERROR
            ):
                error, debug_information = (
                    message.parse_error()
                )

                raise RuntimeError(
                    "출력 파이프라인을 시작하지 "
                    "못했습니다.\n"
                    f"GStreamer 오류: {error}\n"
                    f"디버그 정보: {debug_information}"
                )

            raise RuntimeError(
                "출력 파이프라인을 시작하지 "
                "못했습니다."
            )

        self._running = True

        print(
            f"[출력] 화면 출력 준비: "
            f"{self.video_sink}"
        )

        print(
            "[출력] 상태 전환 결과: "
            f"{self._get_state_result_name(state_result)}"
        )

    def show(
        self,
        frame: np.ndarray,
    ) -> None:
        """
        NumPy BGR 프레임 한 장을 GStreamer appsrc에 전달한다.

        요구되는 입력 형식:

            shape:
                (height, width, 3)

            dtype:
                numpy.uint8

            색상 순서:
                BGR
        """

        if not self._running:
            raise RuntimeError(
                "출력 파이프라인이 실행 중이 아닙니다."
            )

        # 이전 프레임 처리 중 발생한 GStreamer 오류가
        # 있는지 먼저 확인한다.
        self._check_bus_error()

        expected_shape = (
            self.height,
            self.width,
            3,
        )

        if frame.shape != expected_shape:
            raise ValueError(
                "출력 프레임의 shape이 올바르지 않습니다.\n"
                f"현재 shape: {frame.shape}\n"
                f"예상 shape: {expected_shape}"
            )

        if frame.dtype != np.uint8:
            raise ValueError(
                "출력 프레임의 dtype이 올바르지 않습니다.\n"
                f"현재 dtype: {frame.dtype}\n"
                "예상 dtype: uint8"
            )

        # 일부 영상 연산이나 슬라이싱 후에는
        # NumPy 배열이 비연속 메모리를 가질 수 있다.
        #
        # appsrc에 전달하기 전에 C 연속 배열로 바꾼다.
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(
                frame
            )

        # NumPy 배열을 연속된 바이트 데이터로 변환한다.
        frame_data = frame.tobytes()

        if len(frame_data) != self._frame_bytes:
            raise ValueError(
                "출력 프레임의 바이트 크기가 "
                "올바르지 않습니다.\n"
                f"현재 크기: {len(frame_data)}\n"
                f"예상 크기: {self._frame_bytes}"
            )

        # 영상 한 장을 저장할 GStreamer Buffer를 생성한다.
        buffer = Gst.Buffer.new_allocate(
            None,
            len(frame_data),
            None,
        )

        if buffer is None:
            raise RuntimeError(
                "[출력] GStreamer Buffer를 "
                "생성하지 못했습니다."
            )

        # NumPy 영상 데이터를 Gst.Buffer에 복사한다.
        written_bytes = buffer.fill(
            0,
            frame_data,
        )

        if written_bytes != len(frame_data):
            raise RuntimeError(
                "[출력] 프레임 전체를 Gst.Buffer에 "
                "복사하지 못했습니다.\n"
                f"복사된 크기: {written_bytes}\n"
                f"예상 크기: {len(frame_data)}"
            )

        # ----------------------------------------------------
        # 프레임 타임스탬프 설정
        # ----------------------------------------------------
        #
        # PTS:
        #   이 프레임이 표시되어야 하는 시간
        #
        # DTS:
        #   이 프레임이 디코딩되어야 하는 시간
        #
        # 현재는 압축되지 않은 raw 영상이므로
        # PTS와 DTS를 같은 값으로 사용한다.
        buffer.pts = self._next_pts_ns
        buffer.dts = self._next_pts_ns
        buffer.duration = self._frame_duration_ns

        self._next_pts_ns += (
            self._frame_duration_ns
        )

        # appsrc에 영상 프레임을 전달한다.
        flow_result = self._appsrc.emit(
            "push-buffer",
            buffer,
        )

        if flow_result != Gst.FlowReturn.OK:
            self._check_bus_error()

            raise RuntimeError(
                "[출력] appsrc push-buffer가 "
                "실패했습니다.\n"
                f"FlowReturn: {flow_result}"
            )

    def stop(self) -> None:
        """
        출력 파이프라인을 종료하고 자원을 해제한다.
        """

        if not self._running:
            return

        print("[출력] 화면 출력 종료 처리")

        try:
            # appsrc에 더 이상 프레임이 없음을 알린다.
            self._appsrc.emit(
                "end-of-stream"
            )

        except Exception:
            # 종료 중 발생한 EOS 오류는 무시하고
            # 반드시 NULL 상태로 전환한다.
            pass

        finally:
            self._pipeline.set_state(
                Gst.State.NULL
            )

            self._running = False
            self._next_pts_ns = 0

            print("[출력] 화면 출력 종료 완료")