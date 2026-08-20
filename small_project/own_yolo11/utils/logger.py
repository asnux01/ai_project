"""학습 메시지와 처리되지 않은 오류를 콘솔과 파일에 기록한다."""

# Python 표준 라이브러리
import faulthandler
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path


def setup_logger(
    log_dir,
    logger_name="own_yolo11",
    level=logging.INFO,
):
    """
    학습용 logger를 생성하고 로그 파일 경로를 반환한다.

    다음 내용을 기록한다.

    - 일반 학습 진행 메시지
    - 처리되지 않은 Python 예외
    - DataLoader 등 보조 스레드에서 발생한 예외
    - Ctrl+C에 의한 사용자 중단
    - SIGABRT, SIGSEGV 등 치명적인 프로세스 종료 traceback

    Args:
        log_dir:
            로그 파일을 저장할 디렉터리

        logger_name:
            생성할 logger의 고유 이름

        level:
            기록할 최소 로그 수준

    Returns:
        logger:
            콘솔과 파일에 연결된 logger

        log_path:
            이번 실행에서 사용하는 로그 파일 경로
    """

    # 문자열 경로를 Path 객체로 변환한다.
    log_dir = Path(log_dir)

    # 로그 디렉터리가 없으면 생성한다.
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 같은 날 여러 번 실행해도 파일 이름이 겹치지 않도록
    # 날짜와 시간을 파일 이름에 포함한다.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        log_dir
        / f"train_{timestamp}.log"
    )

    # 동일한 이름의 logger를 가져온다.
    logger = logging.getLogger(
        logger_name
    )

    # 지정한 수준 이상의 메시지를 기록한다.
    logger.setLevel(level)

    # root logger로 메시지가 전달되어
    # 중복 출력되는 것을 방지한다.
    logger.propagate = False

    # 기존 faulthandler가 이전 로그 파일을 사용 중이면
    # handler를 닫기 전에 먼저 비활성화한다.
    if faulthandler.is_enabled():
        faulthandler.disable()

    # 같은 프로세스에서 setup_logger가 다시 호출될 경우
    # 기존 handler를 제거하고 파일을 닫는다.
    for handler in list(
        logger.handlers
    ):
        logger.removeHandler(handler)

        try:
            handler.flush()
        except Exception:
            pass

        try:
            handler.close()
        except Exception:
            pass

    # 콘솔과 로그 파일에 사용할 출력 형식
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 학습 진행 상황을 터미널에 출력한다.
    console_handler = logging.StreamHandler(
        stream=sys.stdout
    )

    console_handler.setLevel(level)
    console_handler.setFormatter(
        formatter
    )

    # 학습 진행 상황을 UTF-8 로그 파일에 기록한다.
    file_handler = logging.FileHandler(
        filename=log_path,
        mode="a",
        encoding="utf-8",
    )

    file_handler.setLevel(level)
    file_handler.setFormatter(
        formatter
    )

    # 하나의 logger가 콘솔과 파일에 동시에 기록하도록 한다.
    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    def flush_handlers():
        """
        handler의 버퍼에 남아 있는 내용을 즉시 기록한다.

        프로그램이 오류로 종료되기 직전에 로그 내용이
        파일에 남지 않는 문제를 줄이기 위해 사용한다.
        """

        for current_handler in logger.handlers:
            try:
                current_handler.flush()
            except Exception:
                pass

    def handle_uncaught_exception(
        exception_type,
        exception_value,
        exception_traceback,
    ):
        """
        main 스레드에서 처리되지 않은 예외를 기록한다.

        코드에서 별도의 try/except로 처리되지 않은 예외가
        프로그램을 종료시키기 직전에 호출된다.
        """

        # 사용자가 Ctrl+C를 누른 경우 traceback 대신
        # 사용자 중단이라는 내용을 명확하게 기록한다.
        if issubclass(
            exception_type,
            KeyboardInterrupt,
        ):
            logger.warning(
                "사용자가 Ctrl+C를 눌러 학습을 중단했습니다."
            )

        else:
            # exc_info에 실제 예외 정보를 전달하면
            # 오류 메시지와 전체 traceback이 함께 기록된다.
            logger.critical(
                "처리되지 않은 예외로 학습이 중단되었습니다.",
                exc_info=(
                    exception_type,
                    exception_value,
                    exception_traceback,
                ),
            )

        flush_handlers()

    def handle_thread_exception(args):
        """
        보조 스레드에서 처리되지 않은 예외를 기록한다.

        DataLoader 관련 스레드나 별도로 생성한 스레드에서
        발생한 예외를 로그 파일에 남긴다.
        """

        thread_name = (
            args.thread.name
            if args.thread is not None
            else "unknown"
        )

        logger.critical(
            "보조 스레드 '%s'에서 처리되지 않은 예외가 "
            "발생했습니다.",
            thread_name,
            exc_info=(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
            ),
        )

        flush_handlers()

    def handle_unraisable_exception(args):
        """
        소멸자나 객체 정리 과정에서 발생한 예외를 기록한다.

        일반적인 try/except나 sys.excepthook으로 처리되지 않는
        예외를 확인하기 위해 사용한다.
        """

        object_description = (
            repr(args.object)
            if args.object is not None
            else "unknown"
        )

        logger.error(
            "객체 정리 과정에서 처리할 수 없는 예외가 "
            "발생했습니다. 객체: %s",
            object_description,
            exc_info=(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
            ),
        )

        flush_handlers()

    # main 스레드에서 발생한 처리되지 않은 예외를 기록한다.
    sys.excepthook = (
        handle_uncaught_exception
    )

    # DataLoader 등의 보조 스레드에서 발생한
    # 처리되지 않은 예외를 기록한다.
    threading.excepthook = (
        handle_thread_exception
    )

    # 객체 소멸 또는 정리 과정에서 발생한
    # 처리할 수 없는 예외를 기록한다.
    sys.unraisablehook = (
        handle_unraisable_exception
    )

    # SIGABRT, SIGSEGV 등으로 Python 프로세스가
    # 강제 종료되는 경우 당시의 Python traceback을
    # 현재 로그 파일에 기록한다.
    faulthandler.enable(
        file=file_handler.stream,
        all_threads=True,
    )

    # 로그 파일이 실제로 생성되었는지 확인할 수 있도록
    # 첫 번째 메시지로 로그 경로를 기록한다.
    logger.info(
        "로그 파일: %s",
        log_path,
    )

    # train.py에서 사용할 logger와 로그 파일 경로를 반환한다.
    return logger, log_path