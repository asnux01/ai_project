# Python 표준 라이브러리
import logging
import ssl
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# PyTorch Tensor 처리
import torch

# PIL Mosaic 이미지를 직접 만들기 위해 사용한다.
from PIL import Image

# Mosaic에 들어갈 이미지를 resize하기 위해 사용한다.
from torchvision.transforms import functional as F

from torchvision.transforms.functional import (
    InterpolationMode
)

# Torchvision의 COCO JSON 파서와 이미지 로더
from torchvision.datasets import CocoDetection

# 이미지와 bbox를 함께 변환하는 프로젝트 내부 모듈
from .transforms import DetectionTransform


# --------------------------------------------------
# COCO2017 공식 다운로드 주소
# --------------------------------------------------

COCO2017_IMAGE_URLS = {
    "train2017": (
        "https://images.cocodataset.org/"
        "zips/train2017.zip"
    ),

    "val2017": (
        "https://images.cocodataset.org/"
        "zips/val2017.zip"
    )
}

COCO2017_ANNOTATION_URL = (
    "https://images.cocodataset.org/"
    "annotations/annotations_trainval2017.zip"
)

# SSL 인증서 검증 생략은
# 다른 주소가 아닌 COCO 공식 서버에만 허용한다.
COCO2017_DOWNLOAD_HOST = (
    "images.cocodataset.org"
)

# train.py의 setup_logger()가
# 같은 이름으로 logger를 구성한다.
LOGGER = logging.getLogger(
    "own_yolo11"
)


# --------------------------------------------------
# 로그 함수
# --------------------------------------------------

def _log(
    level,
    message,
    *args
):
    """
    logger가 준비된 경우에는
    콘솔과 로그 파일에 메시지를 기록한다.

    Dataset만 단독으로 사용하는 경우에는
    일반 print로 메시지를 출력한다.
    """

    # train.py에서 setup_logger()가 실행됐다면
    # LOGGER에 console/file handler가 존재한다.
    if LOGGER.handlers:
        getattr(
            LOGGER,
            level,
        )(
            message,
            *args,
        )

        return

    # Dataset만 별도로 사용할 때도
    # 다운로드 진행 상황을 확인할 수 있게 한다.
    formatted_message = (
        message % args
        if args
        else message
    )

    print(
        formatted_message
    )


# --------------------------------------------------
# 기존 COCO 데이터셋 탐색
# --------------------------------------------------

def _normalize_dataset_root(
    dataset_root
):
    """일반 디렉터리와 심볼릭 링크 경로를 보존해 절대 경로로 만든다."""

    return (
        Path(dataset_root)
        .expanduser()
        .absolute()
    )


def _coco2017_paths(
    dataset_root
):
    """COCO 루트를 train/val 이미지 및 annotation 경로로 확장한다."""

    dataset_root = _normalize_dataset_root(
        dataset_root
    )

    return {
        "train_images": (
            dataset_root
            / "images"
            / "train2017"
        ),
        "val_images": (
            dataset_root
            / "images"
            / "val2017"
        ),
        "train_annotation": (
            dataset_root
            / "annotations"
            / "instances_train2017.json"
        ),
        "val_annotation": (
            dataset_root
            / "annotations"
            / "instances_val2017.json"
        ),
    }


# --------------------------------------------------
# SSL 인증서 오류 확인
# --------------------------------------------------

def _is_ssl_certificate_error(
    error
):
    """
    중첩된 urllib 예외에서
    SSL 인증서 검증 오류를 찾는다.
    """

    # 현재 검사 중인 오류
    current_error = error

    # 같은 오류를 반복해서 확인하는
    # 순환 구조를 방지하기 위한 집합
    checked_error_ids = set()

    while current_error is not None:

        current_error_id = id(
            current_error
        )

        # 이미 검사한 오류를 다시 만났으면
        # 순환 구조이므로 반복을 중단한다.
        if (
            current_error_id
            in checked_error_ids
        ):
            break

        checked_error_ids.add(
            current_error_id
        )

        # 실제 SSL 인증서 검증 오류인지 확인한다.
        if isinstance(
            current_error,
            ssl.SSLCertVerificationError,
        ):
            return True

        # urllib.error.URLError는
        # 실제 오류 원인을 reason 속성에 저장한다.
        next_error = getattr(
            current_error,
            "reason",
            None,
        )

        # reason이 없는 경우에는
        # Python의 예외 연결 정보인 __cause__를 확인한다.
        if next_error is None:
            next_error = (
                current_error.__cause__
            )

        current_error = next_error

    # 중첩된 예외에서
    # SSL 인증서 검증 오류를 찾지 못했다.
    return False


def _open_download_response(
    request,
    timeout,
    allow_insecure_ssl_fallback
):
    """
    정상적인 SSL 검증을 먼저 사용한다.

    정상 SSL 연결이 인증서 검증 오류로 실패했고,
    설정에서 명시적으로 허용한 경우에만
    COCO 공식 호스트에 한해 검증 없이 한 번 재시도한다.
    """

    try:
        # 첫 번째 연결은 항상
        # 인증기관과 hostname을 정상적으로 검증한다.
        return urllib.request.urlopen(
            request,
            timeout=timeout,
        )

    except urllib.error.URLError as error:

        # DNS, timeout, 연결 거부처럼
        # 인증서와 관계없는 오류는 그대로 전달한다.
        if not _is_ssl_certificate_error(
            error
        ):
            raise

        # 사용자가 명시적으로 허용하지 않았다면
        # SSL 검증을 끄지 않는다.
        if not allow_insecure_ssl_fallback:
            raise

        # 요청 URL에서 scheme과 hostname을 분리한다.
        parsed_url = (
            urllib.parse.urlparse(
                request.full_url
            )
        )

        # SSL 검증 생략은
        # 정확한 COCO 공식 HTTPS 호스트에만 허용한다.
        if (
            parsed_url.scheme != "https"
            or parsed_url.hostname
            != COCO2017_DOWNLOAD_HOST
        ):
            raise RuntimeError(
                "SSL 검증 생략은 COCO 공식 "
                "다운로드 주소에만 허용됩니다: "
                f"{request.full_url}"
            ) from error

        _log(
            "warning",
            "[COCO2017] SSL 인증서 검증 실패. "
            "공식 COCO 호스트에 한해 "
            "검증 없이 한 번 재시도합니다.",
        )

        # 이 SSL context는 hostname과 인증기관을 검사하지 않는다.
        #
        # 보안상 위험할 수 있으므로
        # 위에서 제한한 공식 호스트의 재시도에만 사용한다.
        insecure_context = (
            ssl.create_default_context()
        )

        insecure_context.check_hostname = (
            False
        )

        insecure_context.verify_mode = (
            ssl.CERT_NONE
        )

        return urllib.request.urlopen(
            request,
            timeout=timeout,
            context=insecure_context,
        )


# --------------------------------------------------
# 파일 다운로드
# --------------------------------------------------

def _download_file(
    url,
    destination,
    chunk_size=1024 * 1024,
    allow_insecure_ssl_fallback=False
):
    """
    파일을 임시 경로에 다운로드한 뒤
    검사가 끝난 파일만 최종 경로로 이동한다.
    """

    # chunk 크기는 다운로드 반복에서 사용하므로
    # 양의 정수여야 한다.
    if (
        isinstance(chunk_size, bool)
        or not isinstance(
            chunk_size,
            int,
        )
    ):
        raise TypeError(
            "chunk_size는 정수여야 합니다."
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size는 1 이상이어야 합니다."
        )

    # 문자열로 전달된 경로를
    # Path 객체로 변환한다.
    destination = Path(
        destination
    )

    # 다운로드 파일의 상위 디렉터리가 없으면 생성한다.
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 다운로드 도중 실패한 파일을
    # 완성된 ZIP과 구분하기 위해 .part를 붙인다.
    partial_path = (
        destination.with_suffix(
            destination.suffix
            + ".part"
        )
    )

    # COCO 서버가 일반적인 HTTP 요청으로 인식하도록
    # User-Agent를 지정한다.
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "own-yolo11-"
                "coco-downloader/1.0"
            ),
        },
    )

    _log(
        "info",
        "[COCO2017] 다운로드 주소: %s",
        url,
    )

    try:
        # COCO 이미지 ZIP은 용량이 크므로
        # 연결 제한 시간을 60초로 설정한다.
        with _open_download_response(
            request=request,
            timeout=60,
            allow_insecure_ssl_fallback=(
                allow_insecure_ssl_fallback
            ),
        ) as response:

            # 서버가 Content-Length를 제공하면
            # 전체 파일 크기를 진행률 계산에 사용한다.
            total_size = int(
                response.headers.get(
                    "Content-Length",
                    0,
                )
            )

            # 현재까지 내려받은 byte 수
            downloaded_size = 0

            # 다음으로 출력할 진행률
            next_report_percent = 10

            # 전체 ZIP을 메모리에 한 번에 올리지 않고
            # 일정 크기의 chunk 단위로 파일에 기록한다.
            with partial_path.open(
                "wb"
            ) as output_file:

                while True:
                    chunk = response.read(
                        chunk_size
                    )

                    # 더 이상 읽을 데이터가 없으면
                    # 다운로드가 끝난 것이다.
                    if not chunk:
                        break

                    output_file.write(
                        chunk
                    )

                    downloaded_size += len(
                        chunk
                    )

                    # 전체 크기를 알 수 있을 때만
                    # 10% 단위 진행률을 기록한다.
                    if total_size > 0:
                        percent = int(
                            downloaded_size
                            * 100
                            / total_size
                        )

                        if (
                            percent
                            >= next_report_percent
                        ):
                            _log(
                                "info",
                                "[COCO2017] "
                                "다운로드 진행률: %d%%",
                                min(
                                    percent,
                                    100,
                                ),
                            )

                            next_report_percent += 10

        # HTML 오류 페이지나 프록시 응답을
        # ZIP으로 잘못 저장하지 않도록 검사한다.
        if not zipfile.is_zipfile(
            partial_path
        ):
            raise RuntimeError(
                "다운로드한 파일이 "
                "올바른 ZIP이 아닙니다: "
                f"{url}"
            )

        # 다운로드와 ZIP 검사가 모두 끝난 경우에만
        # 임시 파일을 실제 ZIP 이름으로 변경한다.
        partial_path.replace(
            destination
        )

    except Exception:

        # 실패한 .part 파일은 삭제한다.
        #
        # 다음 실행에서 불완전한 파일이
        # 정상 ZIP으로 인식되는 것을 방지한다.
        partial_path.unlink(
            missing_ok=True
        )

        raise


# --------------------------------------------------
# COCO 파일 존재 여부 확인
# --------------------------------------------------

def _has_coco_images(
    image_dir
):
    """폴더 안에 COCO JPG 이미지가 한 장 이상 있는지 확인한다."""

    image_dir = Path(
        image_dir
    )

    return (
        image_dir.is_dir()
        and next(
            image_dir.glob(
                "*.jpg"
            ),
            None,
        )
        is not None
    )


def _has_coco_annotation(
    annotation_file,
):
    """
    COCO annotation JSON 파일이
    존재하고 비어 있지 않은지 확인한다.
    """

    annotation_file = Path(
        annotation_file
    )

    return (
        annotation_file.is_file()
        and annotation_file.stat().st_size
        > 0
    )


# --------------------------------------------------
# ZIP 압축 해제
# --------------------------------------------------

def _extract_zip(
    archive_path,
    destination
):
    """
    ZIP 내부의 모든 경로를 검사한 뒤
    지정한 디렉터리에 압축을 해제한다.
    """

    archive_path = Path(
        archive_path
    )

    destination = Path(
        destination
    )

    # 압축 해제 디렉터리가 없으면 생성한다.
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 비교에 사용할 최종 절대 경로
    destination_resolved = (
        destination.resolve()
    )

    _log(
        "info",
        "[COCO2017] 압축 해제: "
        "%s -> %s",
        archive_path.name,
        destination,
    )

    with zipfile.ZipFile(
        archive_path
    ) as archive:

        # ZIP 내부에 ../ 같은 경로가 포함되면
        # 지정한 디렉터리 밖의 파일을 덮어쓸 수 있다.
        #
        # 따라서 압축 해제 전에 모든 경로를 검사한다.
        for member in archive.infolist():

            member_path = (
                destination
                / member.filename
            ).resolve()

            # 모든 member 경로는
            # destination 자체이거나 그 하위 경로여야 한다.
            if (
                member_path
                != destination_resolved
                and destination_resolved
                not in member_path.parents
            ):
                raise RuntimeError(
                    "ZIP에 안전하지 않은 경로가 "
                    "포함돼 있습니다: "
                    f"{member.filename}"
                )

        # 모든 경로 검사가 통과한 후에만
        # 실제 압축 해제를 실행한다.
        archive.extractall(
            destination
        )


# --------------------------------------------------
# COCO split 확인
# --------------------------------------------------

def _infer_coco_split(
    image_dir,
    annotation_file
):
    """
    입력 경로에서 train2017 또는
    val2017 split을 판별한다.
    """

    # 일반적인 COCO 이미지 폴더 이름은
    # train2017 또는 val2017이다.
    image_split = Path(
        image_dir
    ).name

    # annotation 파일 이름에서 확장자를 제거한다.
    #
    # instances_train2017.json
    #             ↓
    # instances_train2017
    annotation_name = Path(
        annotation_file
    ).stem

    # 이미지 폴더 이름이 COCO split 이름이 아니면
    # 이미지 경로에서 split을 찾지 못한 것으로 처리한다.
    if (
        image_split
        not in COCO2017_IMAGE_URLS
    ):
        image_split = None

    # annotation 파일 이름에서도 split을 찾는다.
    annotation_split = None

    for split in COCO2017_IMAGE_URLS:

        if (
            annotation_name
            == f"instances_{split}"
        ):
            annotation_split = split
            break

    # 이미지와 annotation이 서로 다른 split이라면
    # 잘못된 정답으로 학습할 수 있으므로 중단한다.
    #
    # 예:
    # image_dir       = train2017
    # annotation_file = instances_val2017.json
    if (
        image_split is not None
        and annotation_split is not None
        and image_split
        != annotation_split
    ):
        raise ValueError(
            "이미지와 annotation의 "
            "COCO split이 서로 다릅니다: "
            f"{image_split}, "
            f"{annotation_split}"
        )

    # 둘 중 하나에서 확인된 split을 사용한다.
    detected_split = (
        image_split
        or annotation_split
    )

    if detected_split is not None:
        return detected_split

    # 어느 경로에서도 split을 찾지 못하면
    # 어떤 파일을 다운로드해야 하는지 결정할 수 없다.
    raise ValueError(
        "COCO2017 split을 확인할 수 없습니다. "
        "image_dir은 'train2017' 또는 "
        "'val2017'로 끝나야 하며, "
        "annotation_file은 "
        "'instances_train2017.json' 또는 "
        "'instances_val2017.json'이어야 합니다."
    )


# --------------------------------------------------
# COCO 리소스 하나 다운로드 및 압축 해제
# --------------------------------------------------

def _download_and_extract_resource(
    resource_name,
    url,
    archive_path,
    extract_dir,
    allow_insecure_ssl_fallback=False,
):
    """
    이미지 또는 annotation 리소스 하나를
    다운로드하고 압축을 해제한다.
    """

    _log(
        "info",
        "[COCO2017] %s이 없어 "
        "다운로드합니다.",
        resource_name,
    )

    # 공식 COCO 주소에서 ZIP 파일을 다운로드한다.
    _download_file(
        url=url,
        destination=archive_path,
        allow_insecure_ssl_fallback=(
            allow_insecure_ssl_fallback
        ),
    )

    # 다운로드한 ZIP을
    # 필요한 데이터 디렉터리에 압축 해제한다.
    _extract_zip(
        archive_path=archive_path,
        destination=extract_dir,
    )

    # 압축 해제가 끝난 ZIP은 대용량이므로
    # 디스크 공간을 확보하기 위해 삭제한다.
    Path(
        archive_path
    ).unlink(
        missing_ok=True
    )

    _log(
        "info",
        "[COCO2017] %s 준비 완료",
        resource_name,
    )


# --------------------------------------------------
# COCO 데이터 확인 및 자동 다운로드
# --------------------------------------------------

def ensure_coco2017_available(
    image_dir,
    annotation_file,
    allow_insecure_ssl_fallback=False,
):
    """
    COCO2017 데이터가 존재하는지 확인하고
    누락된 파일만 자동으로 준비한다.
    """

    image_dir = Path(
        image_dir
    )

    annotation_file = Path(
        annotation_file
    )

    # --------------------------------------------------
    # 1. 데이터 존재 여부 확인
    # --------------------------------------------------

    # 이미지 디렉터리에 JPG 파일이 있는지 확인한다.
    images_available = (
        _has_coco_images(
            image_dir
        )
    )

    # annotation JSON이 존재하고 비어 있지 않은지 확인한다.
    annotation_available = (
        _has_coco_annotation(
            annotation_file
        )
    )

    # 이미지와 annotation이 모두 있다면
    # 다운로드하지 않고 기존 데이터를 사용한다.
    if (
        images_available
        and annotation_available
    ):
        _log(
            "info",
            "[COCO2017] 기존 데이터셋 사용: %s",
            image_dir,
        )

        return

    # --------------------------------------------------
    # 2. 사용할 split과 데이터 루트 확인
    # --------------------------------------------------

    split = _infer_coco_split(
        image_dir=image_dir,
        annotation_file=(
            annotation_file
        ),
    )

    # 예상되는 데이터 구조:
    #
    # coco/                       <- dataset_root
    # ├── images/
    # │   ├── train2017/
    # │   └── val2017/
    # └── annotations/
    dataset_root = (
        image_dir.parent.parent
    )

    # 다운로드 중인 ZIP은
    # 데이터 파일과 분리된 임시 디렉터리에 저장한다.
    download_dir = (
        dataset_root
        / ".downloads"
    )

    # --------------------------------------------------
    # 3. 누락된 항목 확인
    # --------------------------------------------------

    missing_items = []

    if not images_available:
        missing_items.append(
            f"{split} 이미지"
        )

    if not annotation_available:
        missing_items.append(
            "annotation"
        )

    _log(
        "info",
        "[COCO2017] 누락된 항목: %s",
        ", ".join(
            missing_items
        ),
    )

    # --------------------------------------------------
    # 4. 누락된 항목만 다운로드
    # --------------------------------------------------

    try:
        # 현재 split의 이미지가 없을 때만
        # 대용량 이미지 ZIP을 다운로드한다.
        if not images_available:

            image_archive = (
                download_dir
                / f"{split}.zip"
            )

            _download_and_extract_resource(
                resource_name=(
                    f"{split} 이미지"
                ),
                url=(
                    COCO2017_IMAGE_URLS[
                        split
                    ]
                ),
                archive_path=(
                    image_archive
                ),
                allow_insecure_ssl_fallback=(
                    allow_insecure_ssl_fallback
                ),

                # train2017.zip 내부에는
                # train2017 디렉터리가 들어 있다.
                #
                # images에 압축을 풀면
                # images/train2017 구조가 만들어진다.
                extract_dir=(
                    image_dir.parent
                ),
            )

        # annotation이 없을 때만
        # train/val 공통 annotation ZIP을 다운로드한다.
        if not annotation_available:

            annotation_archive = (
                download_dir
                / "annotations_"
                "trainval2017.zip"
            )

            _download_and_extract_resource(
                resource_name=(
                    "train/val annotation"
                ),
                url=(
                    COCO2017_ANNOTATION_URL
                ),
                archive_path=(
                    annotation_archive
                ),
                allow_insecure_ssl_fallback=(
                    allow_insecure_ssl_fallback
                ),

                # ZIP 내부에는 annotations 폴더가 들어 있다.
                #
                # coco에 압축을 풀면
                # coco/annotations 구조가 만들어진다.
                extract_dir=(
                    annotation_file
                    .parent
                    .parent
                ),
            )

    except Exception as error:
        # 회사나 학교 네트워크 정책, 프록시,
        # COCO 서버 상태, 저장 공간, 쓰기 권한 등에 따라
        # 자동 다운로드가 실패할 수 있다.
        #
        # 자동 다운로드가 실패한 경우
        # 수동 다운로드 주소와 압축 해제 위치를 알려준다.
        raise RuntimeError(
            "COCO2017 자동 다운로드에 실패했습니다. "
            "네트워크, 디스크 공간, "
            f"'{dataset_root}'의 쓰기 권한을 "
            "확인하세요.\n"
            "수동 이미지 다운로드: "
            f"{COCO2017_IMAGE_URLS[split]}\n"
            "이미지 ZIP 압축 해제 위치: "
            f"{image_dir.parent}\n"
            "수동 annotation 다운로드: "
            f"{COCO2017_ANNOTATION_URL}\n"
            "annotation ZIP 압축 해제 위치: "
            f"{annotation_file.parent.parent}"
        ) from error

    finally:
        # 다운로드 ZIP이 모두 삭제되어
        # 임시 디렉터리가 비어 있다면
        # .downloads 디렉터리도 제거한다.
        if (
            download_dir.is_dir()
            and not any(
                download_dir.iterdir()
            )
        ):
            download_dir.rmdir()

    # --------------------------------------------------
    # 5. 다운로드 및 압축 해제 결과 확인
    # --------------------------------------------------

    # 압축 해제 후 실제 이미지가 생성됐는지 확인한다.
    if not _has_coco_images(
        image_dir
    ):
        raise FileNotFoundError(
            "압축 해제 후에도 "
            "COCO2017 이미지를 찾을 수 없습니다: "
            f"{image_dir}"
        )

    # 압축 해제 후 annotation JSON이 생성됐는지 확인한다.
    if not _has_coco_annotation(
        annotation_file
    ):
        raise FileNotFoundError(
            "압축 해제 후에도 "
            "COCO2017 annotation을 "
            "찾을 수 없습니다: "
            f"{annotation_file}"
        )

    _log(
        "info",
        "[COCO2017] %s 데이터셋 준비 완료",
        split,
    )


# --------------------------------------------------
# 기존 데이터셋 선택 또는 공식 데이터셋 다운로드
# --------------------------------------------------

def is_complete_coco2017_dataset(
    dataset_root,
):
    """학습과 검증에 필요한 COCO2017 파일이 모두 있는지 확인한다."""

    dataset_root = _normalize_dataset_root(
        dataset_root
    )

    # Path의 파일 검사는 정상 심볼릭 링크의 대상을 자동으로 따라간다.
    # 따라서 일반 디렉터리와 기존 심볼릭 링크를 같은 방식으로 검사한다.
    if not dataset_root.is_dir():
        return False

    paths = _coco2017_paths(
        dataset_root
    )

    return all(
        (
            _has_coco_images(
                paths["train_images"]
            ),
            _has_coco_images(
                paths["val_images"]
            ),
            _has_coco_annotation(
                paths["train_annotation"]
            ),
            _has_coco_annotation(
                paths["val_annotation"]
            ),
        )
    )


def find_coco2017_dataset(
    candidate_dirs,
):
    """후보 경로 중 처음 발견한 완전한 COCO2017 데이터셋을 반환한다."""

    if not isinstance(
        candidate_dirs,
        (tuple, list),
    ):
        raise TypeError(
            "candidate_dirs는 경로의 "
            "tuple 또는 list여야 합니다."
        )

    if not candidate_dirs:
        raise ValueError(
            "candidate_dirs에는 경로가 "
            "하나 이상 필요합니다."
        )

    checked_paths = set()

    for candidate_dir in candidate_dirs:
        candidate_dir = _normalize_dataset_root(
            candidate_dir
        )

        # 같은 경로가 중복 설정된 경우에는 한 번만 검사한다.
        candidate_key = str(
            candidate_dir
        )

        if candidate_key in checked_paths:
            continue

        checked_paths.add(
            candidate_key
        )

        # 링크 대상이 사라진 깨진 심볼릭 링크는 사용할 수 없다.
        if (
            candidate_dir.is_symlink()
            and not candidate_dir.exists()
        ):
            _log(
                "warning",
                "[COCO2017] 깨진 심볼릭 링크를 건너뜁니다: %s",
                candidate_dir,
            )

            continue

        if is_complete_coco2017_dataset(
            candidate_dir
        ):
            if candidate_dir.is_symlink():
                _log(
                    "info",
                    "[COCO2017] 기존 심볼릭 링크의 "
                    "데이터셋 사용: %s -> %s",
                    candidate_dir,
                    candidate_dir.resolve(),
                )

            else:
                _log(
                    "info",
                    "[COCO2017] 기존 데이터셋 사용: %s",
                    candidate_dir,
                )

            return candidate_dir

        if candidate_dir.exists():
            _log(
                "warning",
                "[COCO2017] 경로는 존재하지만 필요한 파일이 "
                "모두 없어 건너뜁니다: %s",
                candidate_dir,
            )

        else:
            _log(
                "info",
                "[COCO2017] 후보 경로에 데이터셋이 없습니다: %s",
                candidate_dir,
            )

    return None


def prepare_coco2017_dataset(
    candidate_dirs,
    download_dir,
    auto_download=True,
    allow_insecure_ssl_fallback=False,
):
    """기존 COCO2017을 선택하고, 없을 때만 공식 서버에서 다운로드한다.

    이 함수는 심볼릭 링크를 새로 만들거나 기존 링크를 변경하지 않는다.
    후보 경로에 이미 존재하는 정상 심볼릭 링크는 일반 디렉터리처럼
    검사하며, 완전한 COCO2017 데이터가 있으면 그 경로를 그대로 반환한다.
    """

    if not isinstance(
        auto_download,
        bool,
    ):
        raise TypeError(
            "auto_download는 bool이어야 합니다."
        )

    if not isinstance(
        allow_insecure_ssl_fallback,
        bool,
    ):
        raise TypeError(
            "allow_insecure_ssl_fallback은 "
            "bool이어야 합니다."
        )

    if not isinstance(
        candidate_dirs,
        (tuple, list),
    ):
        raise TypeError(
            "candidate_dirs는 경로의 "
            "tuple 또는 list여야 합니다."
        )

    if not candidate_dirs:
        raise ValueError(
            "candidate_dirs에는 경로가 "
            "하나 이상 필요합니다."
        )

    download_dir = _normalize_dataset_root(
        download_dir
    )

    # 다운로드 경로도 기존 데이터 후보에 포함한다.
    # 사용자가 후보 목록에서 빠뜨려도 중복 다운로드하지 않기 위함이다.
    search_dirs = list(
        candidate_dirs
    )

    if download_dir not in search_dirs:
        search_dirs.append(
            download_dir
        )

    existing_dataset = find_coco2017_dataset(
        search_dirs
    )

    if existing_dataset is not None:
        return existing_dataset

    if not auto_download:
        checked = "\n".join(
            f"- {_normalize_dataset_root(path)}"
            for path in search_dirs
        )

        raise FileNotFoundError(
            "사용 가능한 COCO2017 데이터셋을 "
            "찾지 못했습니다.\n"
            f"검사한 경로:\n{checked}"
        )

    # 어느 후보에도 완전한 데이터셋이 없을 때만
    # 지정된 다운로드 경로를 준비한다.
    # 이 경로 자체가 정상 심볼릭 링크라면 링크 대상을 그대로 사용한다.
    if (
        download_dir.exists()
        and not download_dir.is_dir()
    ):
        raise NotADirectoryError(
            "COCO 다운로드 경로가 "
            "디렉터리가 아닙니다: "
            f"{download_dir}"
        )

    if (
        download_dir.is_symlink()
        and not download_dir.exists()
    ):
        raise FileNotFoundError(
            "COCO 다운로드 경로가 깨진 "
            "심볼릭 링크입니다: "
            f"{download_dir}"
        )

    download_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = _coco2017_paths(
        download_dir
    )

    _log(
        "info",
        "[COCO2017] 기존 데이터셋이 없어 "
        "공식 서버에서 다운로드합니다: %s",
        download_dir,
    )

    # ensure 함수는 이미 있는 항목은 유지하고 누락된 항목만 받는다.
    ensure_coco2017_available(
        image_dir=paths["train_images"],
        annotation_file=(
            paths["train_annotation"]
        ),
        allow_insecure_ssl_fallback=(
            allow_insecure_ssl_fallback
        ),
    )

    ensure_coco2017_available(
        image_dir=paths["val_images"],
        annotation_file=(
            paths["val_annotation"]
        ),
        allow_insecure_ssl_fallback=(
            allow_insecure_ssl_fallback
        ),
    )

    if not is_complete_coco2017_dataset(
        download_dir
    ):
        raise RuntimeError(
            "다운로드 후에도 COCO2017 데이터셋이 "
            "완전하지 않습니다: "
            f"{download_dir}"
        )

    _log(
        "info",
        "[COCO2017] 다운로드한 데이터셋 사용: %s",
        download_dir,
    )

    return download_dir


# --------------------------------------------------
# COCO2017 Dataset
# --------------------------------------------------

class Coco2017Dataset(
    CocoDetection
):
    """COCO2017 객체 탐지용 Dataset."""

    def __init__(
        self,
        image_dir,
        annotation_file,
        image_size=640,
        auto_download=True,
        allow_insecure_ssl_fallback=False,
        transform=None,
        mosaic_probability=0.0,
        total_epochs=1,
        close_mosaic_epochs=0,
    ):
        """
        Args:
            image_dir:
                COCO 이미지 디렉터리 경로

            annotation_file:
                COCO annotation JSON 파일 경로

            image_size:
                모델에 입력할 이미지의 높이와 너비

            auto_download:
                True이면 데이터가 없을 때
                COCO2017을 자동으로 다운로드

            allow_insecure_ssl_fallback:
                정상 SSL 검증이 실패한 경우
                공식 COCO 호스트에 한해
                검증 없이 한 번 재시도

            transform:
                PIL 이미지와 target을 함께 변환할 함수

                None이면 무작위 증강이 없는
                기본 Letterbox를 사용
        """

        # --------------------------------------------------
        # 1. 입력 설정 유효성 검사
        # --------------------------------------------------

        # 다운로드를 시작하기 전에
        # 잘못된 이미지 크기를 먼저 확인한다.
        if (
            isinstance(
                image_size,
                bool,
            )
            or not isinstance(
                image_size,
                int,
            )
        ):
            raise TypeError(
                "image_size는 정수여야 합니다."
            )

        if image_size <= 0:
            raise ValueError(
                "image_size는 0보다 커야 합니다."
            )

        if not isinstance(
            auto_download,
            bool,
        ):
            raise TypeError(
                "auto_download는 bool이어야 합니다."
            )

        if not isinstance(
            allow_insecure_ssl_fallback,
            bool,
        ):
            raise TypeError(
                "allow_insecure_ssl_fallback은 "
                "bool이어야 합니다."
            )

        # 문자열 경로도 처리할 수 있도록
        # Path 객체로 변환한다.
        image_dir = Path(
            image_dir
        )

        annotation_file = Path(
            annotation_file
        )

        # --------------------------------------------------
        # 2. COCO2017 데이터 준비
        # --------------------------------------------------

        # CocoDetection은 초기화하면서
        # annotation 파일을 바로 읽는다.
        #
        # 따라서 super().__init__()보다 먼저
        # 데이터가 존재하는지 확인해야 한다.
        if auto_download:

            ensure_coco2017_available(
                image_dir=image_dir,
                annotation_file=(
                    annotation_file
                ),
                allow_insecure_ssl_fallback=(
                    allow_insecure_ssl_fallback
                ),
            )

        else:
            # 자동 다운로드를 끈 경우에는
            # 부족한 파일을 명확하게 알려준다.
            if not _has_coco_images(
                image_dir
            ):
                raise FileNotFoundError(
                    "COCO 이미지 디렉터리가 없거나 "
                    "JPG 파일이 없습니다: "
                    f"{image_dir}"
                )

            if not _has_coco_annotation(
                annotation_file
            ):
                raise FileNotFoundError(
                    "COCO annotation 파일이 "
                    "없거나 비어 있습니다: "
                    f"{annotation_file}"
                )

        # --------------------------------------------------
        # 3. Torchvision CocoDetection 초기화
        # --------------------------------------------------

        # COCO JSON을 읽고
        # 이미지 id와 annotation을 연결한다.
        super().__init__(
            root=str(
                image_dir
            ),
            annFile=str(
                annotation_file
            ),
        )

        # Dataset이 반환해야 할 최종 이미지 크기
        self.image_size = image_size

        # transform을 전달하지 않은 기존 호출도 동작하도록
        # 무작위 증강이 없는 기본 Letterbox를 사용한다.
        if transform is None:
            transform = (
                DetectionTransform(
                    image_size=image_size,
                    training=False,
                )
            )

        # 클래스나 함수처럼 호출 가능한 객체여야 한다.
        if not callable(
            transform
        ):
            raise TypeError(
                "transform은 호출 가능한 "
                "객체여야 합니다."
            )

        # CocoDetection의 self.transform과 구분하기 위해
        # 별도의 이름으로 이미지·bbox 동시 변환을 저장한다.
        self.detection_transform = (
            transform
        )

        # --------------------------------------------------
        # Mosaic augmentation 설정
        # --------------------------------------------------

        if not (
           0.0
           <= mosaic_probability
            <= 1.0
        ):
            raise ValueError(
                "mosaic_probability는 "
                "0과 1 사이여야 합니다."
            )

        if (
            isinstance(
                total_epochs,
                bool,
            )
            or not isinstance(
                total_epochs,
                int,
            )
            or total_epochs <= 0
        ):
            raise ValueError(
                "total_epochs는 "
                "1 이상의 정수여야 합니다."
            )

        if (
            isinstance(
                close_mosaic_epochs,
                bool,
            )
            or not isinstance(
                close_mosaic_epochs,
                int,
            )
            or not (
                0
                <= close_mosaic_epochs
                <= total_epochs
            )
        ):
            raise ValueError(
                "close_mosaic_epochs는 "
                "0 이상 total_epochs 이하여야 합니다."
            )

        self.mosaic_probability = float(
            mosaic_probability
        )

        self.total_epochs = int(
            total_epochs
        )

        self.close_mosaic_epochs = int(
            close_mosaic_epochs
        )

        # persistent_workers=True인 경우에도
        # 메인 프로세스에서 변경한 epoch을
        # DataLoader worker가 볼 수 있도록           
        # 공유 메모리 Tensor로 관리한다.
        self.current_epoch = torch.zeros(
            (),
            dtype=torch.int64,
        )

        self.current_epoch.share_memory_()
        
        # --------------------------------------------------
        # 4. COCO category id를 연속 label로 변환
        # --------------------------------------------------

        # COCO category_id는 다음처럼
        # 중간 번호가 빠져 있어 연속적이지 않다.
        #
        # 1, 2, 3, ..., 11, 13, ..., 90
        category_ids = sorted(
            self.coco.getCatIds()
        )

        # COCO category_id를
        # 학습에 사용할 0부터 시작하는 label로 변환한다.
        #
        # category_id 1  → label 0
        # category_id 2  → label 1
        # category_id 90 → label 79
        self.category_id_to_label = {
            category_id: label
            for label, category_id
            in enumerate(
                category_ids
            )
        }

        # COCO2017 객체 탐지는 일반적으로 80개 클래스다.
        self.num_classes = len(
            category_ids
        )
        
    def set_epoch(
        self,
        epoch_index,
    ):
        """
        현재 학습 epoch을 Dataset에 전달한다.

        persistent_workers=True에서도
        worker들이 변경된 epoch을 확인할 수 있도록
        공유 Tensor 값을 변경한다.
        """

        if (
            isinstance(
                epoch_index,
                bool,
            )
            or not isinstance(
                epoch_index,
                int,
            )
        ):
            raise TypeError(
                "epoch_index는 정수여야 합니다."
            )

        if not (
            0
            <= epoch_index
            < self.total_epochs
        ):
            raise ValueError(
                "epoch_index가 "
                "전체 epoch 범위를 벗어났습니다."
            )

        self.current_epoch.fill_(
            epoch_index
        )

    def _load_raw_sample(
        self,
        index
    ):
        """index번째 이미지와 객체 탐지 정답을 반환한다."""

        # --------------------------------------------------
        # 1. 이미지와 annotation 읽기
        # --------------------------------------------------

        # 부모 CocoDetection을 이용해
        # PIL 이미지와 annotation 리스트를 읽는다.
        image, annotations = (
            super().__getitem__(
                index
            )
        )

        # 입력 채널을 항상 RGB 3채널로 통일한다.
        image = image.convert(
            "RGB"
        )

        # PIL image.size 순서는
        # (width, height)다.
        (
            original_width,
            original_height,
        ) = image.size

        # --------------------------------------------------
        # 2. Bounding box와 label 추출
        # --------------------------------------------------

        boxes = []
        labels = []

        for annotation in annotations:

            # iscrowd=1은 여러 객체가
            # 하나의 영역으로 묶인 annotation이다.
            #
            # 현재 구현에서는 일반 객체만 학습하기 위해 제외한다.
            if annotation.get(
                "iscrowd",
                0,
            ) == 1:
                continue

            # COCO bbox 형식:
            #
            # [x, y, width, height]
            (
                x,
                y,
                width,
                height,
            ) = annotation["bbox"]

            # 너비나 높이가 0 이하인
            # 잘못된 bbox는 제외한다.
            if (
                width <= 0
                or height <= 0
            ):
                continue

            # COCO xywh bbox를
            # loss가 사용하는 xyxy로 변환한다.
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height

            # bbox가 원본 이미지 범위를
            # 벗어나지 않게 제한한다.
            x1 = max(
                0.0,
                min(
                    float(x1),
                    original_width,
                ),
            )

            y1 = max(
                0.0,
                min(
                    float(y1),
                    original_height,
                ),
            )

            x2 = max(
                0.0,
                min(
                    float(x2),
                    original_width,
                ),
            )

            y2 = max(
                0.0,
                min(
                    float(y2),
                    original_height,
                ),
            )

            # 이미지 범위로 제한한 결과
            # 너비나 높이가 사라진 bbox는 제외한다.
            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

            # COCO category 번호를 가져온다.
            category_id = (
                annotation[
                    "category_id"
                ]
            )

            # 불연속 COCO category 번호를
            # 0부터 시작하는 연속 label로 바꾼다.
            label = (
                self.category_id_to_label[
                    category_id
                ]
            )

            labels.append(
                label
            )

        # --------------------------------------------------
        # 3. bbox와 label을 Tensor로 변환
        # --------------------------------------------------

        if boxes:
            # 객체가 있는 경우:
            #
            # boxes shape:  [N, 4]
            # labels shape: [N]
            boxes = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            labels = torch.tensor(
                labels,
                dtype=torch.int64,
            )

        else:
            # 객체가 없는 이미지도
            # 오류 없이 학습할 수 있도록 빈 Tensor를 만든다.
            boxes = torch.empty(
                (
                    0,
                    4,
                ),
                dtype=torch.float32,
            )

            labels = torch.empty(
                (0,),
                dtype=torch.int64,
            )

        # --------------------------------------------------
        # 4. 원본 좌표 기준 target 생성
        # --------------------------------------------------

        target = {
            # transform 적용 전 원본 이미지 기준
            # xyxy 픽셀 좌표
            "boxes": boxes,

            # 객체별 0~79 클래스 번호
            "labels": labels,

            # COCO JSON에 저장된
            # 이미지 고유 번호
            "image_id": torch.tensor(
                self.ids[index],
                dtype=torch.int64,
            ),
        }

        # --------------------------------------------------
        # 5. 원본 이미지와 target 반환
        # --------------------------------------------------
        #
        # 여기서는 아직 DetectionTransform을
        # 적용하지 않는다.
        #
        # 일반 이미지와 Mosaic 이미지 모두
        # 최종 __getitem__()에서 공통으로
        # DetectionTransform을 적용하기 때문이다.
        return image, target

    def _load_mosaic(
        self,
        index,
    ):
        """
        현재 이미지 1장과 무작위 이미지 3장을 이용해
        2x2 Mosaic 이미지를 만든다.

       최종 반환 크기는 image_size x image_size다.
       """

        # 최종 모델 입력 크기
        size = self.image_size

        # --------------------------------------------------
        # 1. 큰 Mosaic Canvas 생성
        # --------------------------------------------------
        #
        # 먼저 2S x 2S 크기의 큰 Canvas에서
        # 이미지 4장을 배치한다.
        mosaic_size = (
            size
            * 2
        )

        mosaic_image = Image.new(
            mode="RGB",
            size=(
                mosaic_size,
                mosaic_size,
            ),
            color=(
                114,
                114,
                114,
            ),
        )


        # --------------------------------------------------
        # 2. Mosaic 중심점 무작위 결정
        # --------------------------------------------------

        center_min = (
            size
            // 2
        )

        center_max = (
            size
            + size // 2
        )

        center_x = int(
            torch.randint(
                low=center_min,
                high=center_max + 1,
                size=(1,),
            ).item()
        )

        center_y = int(
            torch.randint(
                low=center_min,
                high=center_max + 1,
                size=(1,),
            ).item()
        )


        # --------------------------------------------------
        # 3. 사용할 이미지 4장 선택
        # --------------------------------------------------

        indices = [
            index
        ]

        for _ in range(3):

            random_index = int(
                torch.randint(
                    low=0,
                    high=len(self),
                    size=(1,),
                ).item()
            )

            indices.append(
                random_index
            )


        mosaic_boxes = []
        mosaic_labels = []


        # --------------------------------------------------
        # 4. 이미지 4장을 각각 배치
        # --------------------------------------------------

        for mosaic_index, sample_index in enumerate(
            indices
        ):

            # 원본 PIL 이미지와 bbox를 읽는다.
            image, target = (
                self._load_raw_sample(
                    sample_index
                )
            )

            boxes = (
                target["boxes"]
                .clone()
                .to(
                    dtype=torch.float32
                )
            )

            labels = (
                target["labels"]
                .clone()
                .to(
                    dtype=torch.int64
                )
            )


            original_width, original_height = (
                image.size
            )


            # --------------------------------------------------
            # 5. 원본 비율을 유지하며 resize
            # --------------------------------------------------

            resize_scale = min(
                size
                / float(
                    original_width
                ),
                size
                / float(
                    original_height
                ),
            )
            resized_width = max(
                1,
                int(
                    round(
                        original_width
                        * resize_scale
                    )
                ),
            )
            resized_height = max(
                1,
                int(
                    round(
                        original_height
                        * resize_scale
                    )
                ),
            )

            image = F.resize(
                image,
                [
                    resized_height,
                    resized_width,
                ],
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
                antialias=True,
            )

            # bbox에도 같은 resize 비율 적용
            if boxes.numel() > 0:

                boxes[:, [0, 2]] *= (
                    resize_scale
                )

                boxes[:, [1, 3]] *= (
                    resize_scale
                )


            width = resized_width
            height = resized_height


            # --------------------------------------------------
            # 6. Mosaic 위치 계산
            # --------------------------------------------------

            if mosaic_index == 0:
                # 왼쪽 위

                x1a = max(
                    center_x - width,
                    0,
                )

                y1a = max(
                    center_y - height,
                    0,
                )

                x2a = center_x
                y2a = center_y

                x1b = (
                    width
                    - (
                        x2a
                        - x1a
                    )
                )

                y1b = (
                    height
                    - (
                        y2a
                        - y1a
                    )
                )

                x2b = width
                y2b = height


            elif mosaic_index == 1:
                # 오른쪽 위

                x1a = center_x

                y1a = max(
                    center_y - height,
                    0,
                )

                x2a = min(
                    center_x + width,
                    mosaic_size,
                )

                y2a = center_y

                x1b = 0

                y1b = (
                    height
                    - (
                        y2a
                        - y1a
                    )
                )

                x2b = (
                    x2a
                    - x1a
                )

                y2b = height


            elif mosaic_index == 2:
                # 왼쪽 아래

                x1a = max(
                    center_x - width,
                    0,
                )

                y1a = center_y

                x2a = center_x

                y2a = min(
                    center_y + height,
                    mosaic_size,
                )

                x1b = (
                    width
                    - (
                        x2a
                        - x1a
                    )
                )

                y1b = 0

                x2b = width

                y2b = (
                    y2a
                    - y1a
                )


            else:
                # 오른쪽 아래

                x1a = center_x
                y1a = center_y

                x2a = min(
                    center_x + width,
                    mosaic_size,
                )

                y2a = min(
                    center_y + height,
                    mosaic_size,
                )

                x1b = 0
                y1b = 0

                x2b = (
                    x2a
                    - x1a
                )

                y2b = (
                    y2a
                    - y1a
                )


            # --------------------------------------------------
            # 7. 해당 이미지 영역을 Canvas에 붙인다.
            # --------------------------------------------------

            cropped_image = image.crop(
                (
                    x1b,
                    y1b,
                    x2b,
                    y2b,
                )
            )

            mosaic_image.paste(
                cropped_image,
                (
                    x1a,
                    y1a,
                ),
            )


            # --------------------------------------------------
            # 8. bbox 좌표를 Mosaic 위치로 이동
            # --------------------------------------------------

            if boxes.numel() > 0:
                
                offset_x = (
                    x1a
                    - x1b
                )

                offset_y = (
                    y1a
                    - y1b
                )

                boxes[:, [0, 2]] += (
                    float(
                        offset_x
                    )
                )

                boxes[:, [1, 3]] += (
                    float(
                        offset_y
                   )
                )


            mosaic_boxes.append(
                boxes
            )

            mosaic_labels.append(
                labels
            )


        # --------------------------------------------------
        # 9. 네 이미지의 bbox/label 연결
        # --------------------------------------------------

        boxes = torch.cat(
            mosaic_boxes,
            dim=0,
        )

        labels = torch.cat(
            mosaic_labels,
           dim=0,
        )


        # --------------------------------------------------
        # 10. 2S x 2S Canvas에서
        #     최종 S x S 영역 추출
        # --------------------------------------------------

        crop_left = (
            size
            // 2
        )

        crop_top = (
            size
            // 2
        )

        crop_right = (
            crop_left
            + size
        )

        crop_bottom = (
            crop_top
            + size
        )

        mosaic_image = (
            mosaic_image.crop(
                (
                    crop_left,
                    crop_top,
                    crop_right,
                    crop_bottom,
                )
            )
        )

        # --------------------------------------------------
        # 11. bbox도 Crop 기준으로 이동
        # --------------------------------------------------

        if boxes.numel() > 0:

            boxes[:, [0, 2]] -= (
                float(
                    crop_left
                )
            )

            boxes[:, [1, 3]] -= (
                float(
                    crop_top
                )
            )

            boxes[:, [0, 2]] = (
                boxes[:, [0, 2]].clamp_(
                    min=0.0,
                    max=float(
                        size
                    ),
                )
            )

            boxes[:, [1, 3]] = (
                boxes[:, [1, 3]].clamp_(
                    min=0.0,
                    max=float(
                        size
                    ),
                )
            )


            # --------------------------------------------------
            # 12. Crop 결과 너무 작아진 bbox 제거
            # --------------------------------------------------

            box_width = (
                boxes[:, 2]
                - boxes[:, 0]
            )
            
            box_height = (
                boxes[:, 3]
                - boxes[:, 1]
            )

            valid_mask = (
                (box_width > 2.0)
                & (box_height > 2.0)
            )

            boxes = boxes[
                valid_mask
            ]

            labels = labels[
                valid_mask
            ]
    
        
        # --------------------------------------------------
        # 13. Mosaic target 생성
        # --------------------------------------------------

        target = {
            "boxes": boxes,
            "labels": labels,

            # Mosaic은 여러 이미지로 만들어지지만
            # 학습 loss에서는 image_id를 사용하지 않으므로
            # 대표로 현재 index의 image id를 기록한다.
            "image_id": torch.tensor(
                self.ids[index],
                dtype=torch.int64,
            ),
        }

        return mosaic_image, target

    def __getitem__(
        self,
        index,
    ):
        """
        일반 이미지 또는 Mosaic 이미지를 선택하고,
        최종 DetectionTransform을 적용한다.
        """

        # 현재 epoch을 공유 Tensor에서 읽는다.
        epoch_index = int(
            self.current_epoch.item()
        )

        # 마지막 close_mosaic_epochs 구간이
        # 시작되는 epoch이다.
        mosaic_stop_epoch = max(
            self.total_epochs
            - self.close_mosaic_epochs,
            0,
        )

        # --------------------------------------------------
        # Mosaic 적용 여부 결정
        # --------------------------------------------------

        use_mosaic = (
            self.mosaic_probability
            > 0.0

            and epoch_index
            < mosaic_stop_epoch

            and torch.rand(
                1
            ).item()
            < self.mosaic_probability
        )


        if use_mosaic:

            # 4장의 이미지를 합친다.
            image, target = (
                self._load_mosaic(
                    index
                )
            )

        else:

            # 일반 COCO 이미지 1장을 사용한다.
            image, target = (
                self._load_raw_sample(
                    index
                )
            )


        # --------------------------------------------------
        # 최종 데이터 증강 + Letterbox
        # --------------------------------------------------

        image, target = (
            self.detection_transform(
                image,
                target,
            )
        )


        # --------------------------------------------------
        # 결과 유효성 검사
        # --------------------------------------------------

        if not isinstance(
            image,
            torch.Tensor,
        ):
            raise TypeError(
                "transform 결과 image는 "
                "Tensor여야 합니다."
            )

        if image.shape != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise RuntimeError(
                "Dataset 이미지 shape이 "
                "예상과 다릅니다: "
                f"{tuple(image.shape)}"
            )

        if not isinstance(
            target,
            dict,
        ):
            raise TypeError(
                "transform 결과 target은 "
                "dict여야 합니다."
            )

        return image, target

# --------------------------------------------------
# DataLoader용 collate 함수
# --------------------------------------------------

def detection_collate_fn(
    batch
):
    """
    이미지는 하나의 Tensor로 쌓고
    이미지마다 객체 수가 다른 target은 리스트로 유지한다.
    """

    # 빈 batch라면 torch.stack에서 발생하는
    # 이해하기 어려운 오류 대신 명확한 오류를 발생시킨다.
    if not batch:
        raise ValueError(
            "detection_collate_fn에 "
            "빈 batch가 전달됐습니다."
        )

    # 각 Dataset sample은
    # (image, target) 두 값으로 구성돼야 한다.
    if any(
        (
            not isinstance(
                sample,
                (tuple, list),
            )
            or len(sample) != 2
        )
        for sample in batch
    ):
        raise TypeError(
            "각 Dataset sample은 "
            "(image, target) 형식이어야 합니다."
        )

    # 모든 이미지는 transform에서 같은 크기로 변환된다.
    #
    # B개의 [3, H, W]
    #          ↓
    # [B, 3, H, W]
    images = torch.stack(
        [
            sample[0]
            for sample in batch
        ],
        dim=0,
    )

    # 이미지마다 객체 수가 다르므로
    # target은 하나의 Tensor로 쌓지 않는다.
    targets = [
        sample[1]
        for sample in batch
    ]

    return images, targets