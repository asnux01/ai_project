# 라이브러리
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


# COCO 공식 다운로드 주소
COCO_IMAGE_BASE = "https://images.cocodataset.org"

COCO_URLS = {
    "train2017": COCO_IMAGE_BASE + "/zips/train2017.zip",
    "val2017": COCO_IMAGE_BASE + "/zips/val2017.zip",
    "annotations": COCO_IMAGE_BASE + "/annotations/annotations_trainval2017.zip"
}


# 이미지 데이터 존재 여부 확인
def _image_split_exists(image_dir):

    # 이미지 디렉터리 확인
    if not image_dir.is_dir():
        return False

    # JPG 이미지 존재 여부 확인
    first_image = next(
        image_dir.glob("*.jpg"),
        None
    )

    return first_image is not None


# Annotation 데이터 존재 여부 확인
def _annotations_exist(annotation_dir):

    # Train annotation 경로 생성
    train_annotation = (
        annotation_dir
        / "instances_train2017.json"
    )

    # Validation annotation 경로 생성
    val_annotation = (
        annotation_dir
        / "instances_val2017.json"
    )

    # 두 Annotation 파일 존재 여부 반환
    return (
        train_annotation.is_file()
        and val_annotation.is_file()
    )


# 파일 다운로드
def _download_file(
    url,
    destination,
    chunk_size=8 * 1024 * 1024
):

    # 다운로드 경로 생성
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 이미 다운로드된 파일 확인
    if destination.is_file():
        print(
            f"Download file exists: {destination}"
        )
        return

    # 임시 다운로드 파일 경로 생성
    temporary_file = Path(
        str(destination) + ".part"
    )

    # 기존 다운로드 크기 확인
    downloaded_size = (
        temporary_file.stat().st_size
        if temporary_file.exists()
        else 0
    )

    # HTTP 요청 Header 생성
    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    # 중단된 다운로드 위치 설정
    if downloaded_size > 0:
        headers["Range"] = (
            f"bytes={downloaded_size}-"
        )

    # HTTP 요청 생성
    request = Request(
        url,
        headers=headers
    )

    print(
        f"Downloading: {destination.name}"
    )

    # COCO 서버 연결
    with urlopen(
        request,
        timeout=60,
    ) as response:

        # HTTP 응답 상태 가져오기
        status_code = getattr(
            response,
            "status",
            200
        )

        # 이어받기 가능 여부 확인
        resume_download = (
            downloaded_size > 0
            and status_code == 206
        )

        # 이어받기를 지원하지 않으면 처음부터 다운로드
        if downloaded_size > 0 and not resume_download:
            downloaded_size = 0

        # 파일 저장 방식 설정
        file_mode = (
            "ab"
            if resume_download
            else "wb"
        )

        # 남은 파일 크기 가져오기
        content_length = int(
            response.headers.get(
                "Content-Length",
                0
            )
        )

        # 전체 파일 크기 계산
        total_size = (
            downloaded_size
            + content_length
            if resume_download
            else content_length
        )

        # 다운로드 파일 열기
        with temporary_file.open(
            file_mode
        ) as file:

            current_size = downloaded_size

            while True:

                # 데이터를 일정 크기씩 다운로드
                chunk = response.read(
                    chunk_size
                )

                # 다운로드 완료 확인
                if not chunk:
                    break

                # 다운로드 데이터 저장
                file.write(chunk)

                # 다운로드 크기 갱신
                current_size += len(chunk)

                # 다운로드 진행률 출력
                if total_size > 0:
                    progress = (
                        current_size
                        / total_size
                        * 100
                    )

                    print(
                        (
                            "\r"
                            f"{destination.name}: "
                            f"{progress:6.2f}%"
                        ),
                        end="",
                        flush=True
                    )

    print()

    # 임시 파일을 최종 파일로 변경
    temporary_file.replace(
        destination
    )

    print(
        f"Download complete: {destination}"
    )


# ZIP 파일 압축 해제
def _extract_zip(
    archive_path,
    extract_dir
):

    # 압축 해제 경로 생성
    extract_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Extracting: {archive_path.name}"
    )

    try:

        # ZIP 파일 열기
        with ZipFile(
            archive_path,
            "r",
        ) as archive:

            # ZIP 파일 압축 해제
            archive.extractall(
                extract_dir
            )

    except BadZipFile as error:

        # 손상된 ZIP 파일 제거
        archive_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            (
                "Downloaded ZIP file is invalid: "
                f"{archive_path}"
            )
        ) from error

    print(
        f"Extraction complete: {extract_dir}"
    )


# COCO 이미지 데이터 준비
def _prepare_image_split(
    split,
    images_dir,
    archives_dir,
    remove_archive=True
):

    # 이미지 디렉터리 경로 생성
    image_dir = (
        images_dir
        / split
    )

    # 이미지 데이터 존재 여부 확인
    if _image_split_exists(
        image_dir
    ):
        print(
            f"{split} already exists: {image_dir}"
        )

        return image_dir

    # ZIP 파일 경로 생성
    archive_path = (
        archives_dir
        / f"{split}.zip"
    )

    # COCO 이미지 다운로드
    _download_file(
        url=COCO_URLS[split],
        destination=archive_path
    )

    # 이미지 압축 해제
    _extract_zip(
        archive_path=archive_path,
        extract_dir=images_dir
    )

    # 압축 해제 결과 확인
    if not _image_split_exists(
        image_dir
    ):
        raise RuntimeError(
            (
                "COCO image extraction failed: "
                f"{image_dir}"
            )
        )

    # ZIP 파일 제거
    if remove_archive:
        archive_path.unlink(
            missing_ok=True
        )

    return image_dir


# COCO Annotation 데이터 준비
def _prepare_annotations(
    root_dir,
    annotation_dir,
    archives_dir,
    remove_archive=True,
):

    # Annotation 존재 여부 확인
    if _annotations_exist(
        annotation_dir
    ):
        print(
            (
                "COCO annotations already exist: "
                f"{annotation_dir}"
            )
        )

        return annotation_dir

    # Annotation ZIP 경로 생성
    archive_path = (
        archives_dir
        / "annotations_trainval2017.zip"
    )

    # COCO Annotation 다운로드
    _download_file(
        url=COCO_URLS["annotations"],
        destination=archive_path,
    )

    # Annotation 압축 해제
    _extract_zip(
        archive_path=archive_path,
        extract_dir=root_dir,
    )

    # 압축 해제 결과 확인
    if not _annotations_exist(
        annotation_dir
    ):
        raise RuntimeError(
            (
                "COCO annotation extraction failed: "
                f"{annotation_dir}"
            )
        )

    # ZIP 파일 제거
    if remove_archive:
        archive_path.unlink(
            missing_ok=True
        )

    return annotation_dir


# COCO2017 Dataset 준비
def prepare_coco2017(
    root_dir,
    remove_archives=True,
):

    # Dataset 루트 경로 생성
    root_dir = Path(root_dir)

    # 이미지 디렉터리 경로 생성
    images_dir = (
        root_dir
        / "images"
    )

    # Annotation 디렉터리 경로 생성
    annotation_dir = (
        root_dir
        / "annotations"
    )

    # 다운로드 파일 저장 경로 생성
    archives_dir = (
        root_dir
        / "archives"
    )

    # 필요한 디렉터리 생성
    root_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    archives_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Train 이미지 준비
    train_image_dir = (
        _prepare_image_split(
            split="train2017",
            images_dir=images_dir,
            archives_dir=archives_dir,
            remove_archive=remove_archives
        )
    )

    # Validation 이미지 준비
    val_image_dir = (
        _prepare_image_split(
            split="val2017",
            images_dir=images_dir,
            archives_dir=archives_dir,
            remove_archive=remove_archives
        )
    )

    # Annotation 준비
    _prepare_annotations(
        root_dir=root_dir,
        annotation_dir=annotation_dir,
        archives_dir=archives_dir,
        remove_archive=remove_archives
    )

    # Annotation 파일 경로 생성
    train_annotation_file = (
        annotation_dir
        / "instances_train2017.json"
    )

    val_annotation_file = (
        annotation_dir
        / "instances_val2017.json"
    )

    # Dataset 경로 반환
    return {
        "train_images": train_image_dir,
        "val_images": val_image_dir,
        "train_annotations": train_annotation_file,
        "val_annotations": val_annotation_file
    }