"""이미지 파일, 디렉터리, glob 입력을 추론 소스로 확장한다."""

# Python 표준 glob과 Path만 사용하므로 source 탐색 단계에는 GPU 의존성이 없다.
import glob
from pathlib import Path


# Pillow/일반적인 Ultralytics 이미지 source에서 사용하는 확장자 목록이다.
IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _has_glob_magic(value):
    """*, ?, [ 문자가 포함된 glob 패턴인지 확인한다."""

    return any(character in value for character in "*?[")


def _is_supported_image(path):
    """실제 파일이며 지원 확장자인 경우에만 True를 반환한다."""

    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def resolve_image_sources(source):
    """하나 이상의 source를 중복 없는 절대 이미지 경로 목록으로 만든다."""

    # source 한 개와 source 목록을 같은 반복 구조로 처리한다.
    if isinstance(source, (str, Path)):
        source_items = [source]
    else:
        source_items = list(source)

    if not source_items:
        raise ValueError("source에는 이미지 경로가 하나 이상 필요합니다.")

    resolved = []

    for source_item in source_items:
        source_text = str(source_item).strip()

        if not source_text:
            continue

        # 원격 다운로드를 암묵적으로 수행하지 않아 예측 단계의 네트워크 부작용을 막는다.
        if source_text.startswith(("http://", "https://")):
            raise ValueError(
                "현재 predictor는 로컬 이미지, 디렉터리, glob을 지원합니다. "
                "URL 이미지는 먼저 로컬에 저장해 주세요."
            )

        if _has_glob_magic(source_text):
            # 예: images/**/*.jpg
            candidates = [Path(value) for value in glob.glob(source_text, recursive=True)]
        else:
            path = Path(source_text).expanduser()

            if path.is_dir():
                # 디렉터리는 하위 폴더까지 재귀적으로 이미지 파일을 찾는다.
                candidates = list(path.rglob("*"))
            else:
                candidates = [path]

        # 입력 순서가 실행마다 달라지지 않게 절대 경로로 변환 후 정렬한다.
        image_candidates = sorted(
            path.resolve()
            for path in candidates
            if _is_supported_image(path)
        )

        if not image_candidates:
            raise FileNotFoundError(f"지원되는 이미지를 찾을 수 없습니다: {source_text}")

        resolved.extend(image_candidates)

    # 여러 glob이 같은 이미지를 가리켜도 첫 번째 한 번만 추론한다.
    unique_paths = list(dict.fromkeys(resolved))

    if not unique_paths:
        raise FileNotFoundError("추론할 이미지가 없습니다.")

    return unique_paths


def iter_batches(items, batch_size):
    """리스트를 마지막의 작은 배치를 허용하며 순서대로 나눈다."""

    # 마지막 batch는 batch_size보다 작아도 그대로 처리한다.
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
