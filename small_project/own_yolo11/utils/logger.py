"""학습 메시지를 콘솔과 파일에 동시에 기록한다."""
# 라이브러리
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(
    log_dir,
    logger_name="own_yolo11",
    level=logging.INFO
):
    """
    학습용 logger를 만들고
    생성된 로그 파일 경로도 반환한다.

    같은 Python 프로세스에서 함수를 여러 번 호출해도
    handler가 중복되어 메시지가 반복 출력되지 않도록 처리한다.

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

    # 문자열로 전달된 경로도 처리할 수 있도록
    # Path 객체로 변환한다.
    log_dir = Path(log_dir)

    # logs 폴더가 없으면
    # 상위 폴더까지 함께 생성한다.
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 같은 날 여러 번 실행해도 파일이 겹치지 않도록
    # 연월일과 시분초를 파일 이름에 포함한다.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        log_dir
        / f"train_{timestamp}.log"
    )

    # 동일한 이름의 logger를 가져오거나
    # 존재하지 않으면 새로 생성한다.
    logger = logging.getLogger(
        logger_name
    )

    # INFO 이상의 메시지를 기록한다.
    logger.setLevel(level)

    # root logger로 메시지가 다시 전달되어
    # 중복 출력되는 것을 방지한다.
    logger.propagate = False

    # Notebook 또는 테스트에서 logger를 다시 만들었을 때
    # 이전 file handler가 열린 상태로 남는 것을 방지한다.
    for handler in list(
        logger.handlers
    ):
        logger.removeHandler(handler)
        handler.close()

    # 콘솔과 파일에서 동일하게 사용할
    # 시간 | 로그 수준 | 메시지 형식을 만든다.
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 사용자가 터미널에서 학습 진행 상황을
    # 바로 확인할 수 있도록 한다.
    console_handler = (
        logging.StreamHandler(
            stream=sys.stdout
        )
    )

    # 콘솔 handler에 출력 수준과 형식을 연결한다.
    console_handler.setLevel(level)
    console_handler.setFormatter(
        formatter
    )

    # 서버의 터미널 연결이 끊겨도 실행 기록이 남도록
    # UTF-8 텍스트 파일에도 같은 내용을 기록한다.
    file_handler = logging.FileHandler(
        filename=log_path,
        mode="a",
        encoding="utf-8",
    )

    # 파일 handler에도 같은 출력 수준과 형식을 연결한다.
    file_handler.setLevel(level)
    file_handler.setFormatter(
        formatter
    )

    # 하나의 logger가 콘솔과 파일 두 곳으로
    # 메시지를 전달하도록 handler를 등록한다.
    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    # 학습 시작 직후 실제 로그 파일 위치를
    # 확인할 수 있도록 첫 메시지를 기록한다.
    logger.info(
        "로그 파일: %s",
        log_path,
    )

    # train.py가 추가 메시지를 기록할 logger와
    # 생성된 파일 경로를 함께 반환한다.
    return logger, log_path