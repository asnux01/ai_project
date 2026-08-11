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
    ),
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
    *args,
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
# COCO 저장소와 심볼릭 링크 준비
# --------------------------------------------------

def prepare_coco_dataset_link(
    storage_dir,
    link_dir,
):
    """
    실제 COCO 저장소와 프로젝트 내부의
    심볼릭 링크를 준비한다.

    Args:
        storage_dir:
            COCO 파일이 실제로 저장될 물리 경로

            이 프로젝트에서는:
                /home/jblee/datasets/coco

        link_dir:
            프로젝트에서 Dataset 경로로 사용할
            심볼릭 링크 경로

            이 프로젝트에서는:
                <project>/datasets/coco

    Returns:
        link_dir:
            검사 또는 생성이 끝난 심볼릭 링크 경로
    """

    # 실제 저장 경로는 링크의 대상이므로
    # 최종 절대 경로로 변환한다.
    storage_dir = (
        Path(storage_dir)
        .expanduser()
        .resolve(
            strict=False
        )
    )

    # 링크 경로에는 resolve()를 사용하지 않는다.
    #
    # 링크가 이미 존재하는 경우 resolve()를 사용하면
    # 링크 자체가 아니라 링크 대상 경로가 반환되기 때문이다.
    link_dir = (
        Path(link_dir)
        .expanduser()
        .absolute()
    )

    # 실제 저장소와 링크 경로가 같으면
    # 자기 자신을 가리키는 링크가 되므로 허용하지 않는다.
    if link_dir == storage_dir:
        raise ValueError(
            "COCO 실제 저장 경로와 "
            "심볼릭 링크 경로는 "
            "서로 달라야 합니다."
        )

    # 실제 저장 경로에 일반 파일이 있다면
    # COCO 데이터 디렉터리로 사용할 수 없다.
    if (
        storage_dir.exists()
        and not storage_dir.is_dir()
    ):
        raise NotADirectoryError(
            "COCO 실제 저장 경로가 "
            "디렉터리가 아닙니다: "
            f"{storage_dir}"
        )

    # 실제 COCO 데이터가 저장될 디렉터리를 만든다.
    storage_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 프로젝트의 datasets 폴더가 없으면 생성한다.
    link_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 기존 심볼릭 링크 확인
    # --------------------------------------------------

    # is_symlink()는 링크 대상이 사라진
    # 깨진 링크에 대해서도 True를 반환한다.
    if link_dir.is_symlink():

        # 기존 링크가 실제로 가리키는 경로를 확인한다.
        current_target = (
            link_dir.resolve(
                strict=False
            )
        )

        # 다른 데이터셋 경로를 가리키는 링크는
        # 사용자의 의도일 수 있으므로 자동으로 삭제하지 않는다.
        if current_target != storage_dir:
            raise RuntimeError(
                "기존 COCO 심볼릭 링크가 "
                "다른 경로를 가리키고 있습니다.\n"
                f"현재 링크: "
                f"{link_dir} -> {current_target}\n"
                f"필요한 링크: "
                f"{link_dir} -> {storage_dir}"
            )

        # 올바른 링크가 이미 있으면 그대로 사용한다.
        _log(
            "info",
            "[COCO2017] 기존 심볼릭 링크 사용: "
            "%s -> %s",
            link_dir,
            storage_dir,
        )

        return link_dir

    # --------------------------------------------------
    # 링크 위치의 일반 파일 또는 디렉터리 확인
    # --------------------------------------------------

    # 링크 위치에 일반 파일이 있으면
    # 자동으로 삭제하지 않고 오류를 발생시킨다.
    if (
        link_dir.exists()
        and not link_dir.is_dir()
    ):
        raise FileExistsError(
            "COCO 심볼릭 링크 위치에 "
            "일반 파일이 있습니다: "
            f"{link_dir}"
        )

    # 이전 실행에서 빈 일반 디렉터리만 남은 경우에는
    # 데이터 손실 없이 제거하고 링크로 교체할 수 있다.
    if link_dir.is_dir():

        # 내용이 있는 디렉터리는
        # 사용자 데이터가 들어 있을 수 있으므로 삭제하지 않는다.
        if any(
            link_dir.iterdir()
        ):
            raise RuntimeError(
                "COCO 링크 위치에 내용이 있는 "
                "일반 디렉터리가 있습니다. "
                "자동으로 삭제하지 않습니다: "
                f"{link_dir}"
            )

        # 비어 있는 디렉터리만 제거한다.
        link_dir.rmdir()

    # --------------------------------------------------
    # 심볼릭 링크 생성
    # --------------------------------------------------

    try:
        # 프로젝트의 datasets/coco가
        # 실제 저장소를 가리키게 한다.
        link_dir.symlink_to(
            storage_dir,
            target_is_directory=True,
        )

    except OSError as error:
        raise OSError(
            "COCO 데이터용 심볼릭 링크를 "
            "만들지 못했습니다. "
            "링크 생성 권한과 경로를 확인하세요.\n"
            f"생성할 링크: "
            f"{link_dir} -> {storage_dir}"
        ) from error

    _log(
        "info",
        "[COCO2017] 심볼릭 링크 생성: "
        "%s -> %s",
        link_dir,
        storage_dir,
    )

    return link_dir


# --------------------------------------------------
# SSL 인증서 오류 확인
# --------------------------------------------------

def _is_ssl_certificate_error(
    error,
):
    """
    중첩된 urllib 예외에서
    SSL 인증서 검증 오류를 찾는다.
    """

    # 현재 검사 중인 예외
    current_error = error

    # 같은 예외를 반복해서 확인하는
    # 순환 구조를 방지하기 위한 집합
    checked_error_ids = set()

    while current_error is not None:

        current_error_id = id(
            current_error
        )

        # 이미 검사한 예외를 다시 만났으면
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
    allow_insecure_ssl_fallback,
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
    allow_insecure_ssl_fallback=False,
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
    image_dir,
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
    destination,
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
    annotation_file,
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

    def __getitem__(
        self,
        index,
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
        # 5. 이미지와 target을 함께 변환
        # --------------------------------------------------

        # Letterbox, 좌우 반전, 색상 변화와
        # bbox 좌표 변경을 같은 변환에서 처리한다.
        image, target = (
            self.detection_transform(
                image,
                target,
            )
        )

        # 사용자 정의 transform이 잘못 연결된 경우
        # DataLoader에서 늦게 실패하지 않도록 검사한다.
        if not isinstance(
            image,
            torch.Tensor,
        ):
            raise TypeError(
                "transform 결과 image는 "
                "Tensor여야 합니다."
            )

        # 모든 이미지는 DataLoader에서 쌓을 수 있도록
        # 동일한 [3, image_size, image_size]여야 한다.
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
    batch,
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