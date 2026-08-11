# 라이브러리
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F


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


# --------------------------------------------------
# COCO2017 다운로드 및 확인 함수
# --------------------------------------------------

def _download_file(
    url,
    destination,
    chunk_size=1024 * 1024,
):
    """파일을 임시 경로에 다운로드한 뒤 최종 경로로 이동"""

    # 문자열 경로를 Path 객체로 변환
    destination = Path(destination)

    # 다운로드 파일을 저장할 폴더가 없으면 생성
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 다운로드 도중 실패한 파일을 완성된 ZIP과 구분하기 위해
    # 임시 파일에는 .part 확장자를 추가
    partial_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    # COCO 서버가 일반적인 HTTP 요청으로 인식하도록
    # User-Agent를 지정
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "own-yolo11-coco-downloader/1.0"
            )
        },
    )

    print(f"[COCO2017] 다운로드 주소: {url}")

    try:
        # COCO 이미지 ZIP은 용량이 크므로
        # 다운로드 제한 시간을 60초로 설정
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:

            # 서버가 전체 파일 크기를 알려주는 경우
            # 다운로드 진행률을 표시할 때 사용
            total_size = int(
                response.headers.get(
                    "Content-Length",
                    0,
                )
            )

            downloaded_size = 0
            next_report_percent = 10

            # 전체 ZIP을 메모리에 한 번에 올리지 않고
            # 일정 크기의 chunk 단위로 파일에 기록
            with partial_path.open("wb") as output_file:
                while True:
                    chunk = response.read(chunk_size)

                    # 더 이상 읽을 데이터가 없으면
                    # 다운로드 반복을 종료
                    if not chunk:
                        break

                    output_file.write(chunk)
                    downloaded_size += len(chunk)

                    # 전체 크기를 알 수 있는 경우
                    # 10% 단위로 진행률을 출력
                    if total_size > 0:
                        percent = int(
                            downloaded_size
                            * 100
                            / total_size
                        )

                        if percent >= next_report_percent:
                            print(
                                f"  {min(percent, 100)}%"
                            )
                            next_report_percent += 10

        # 다운로드가 완전히 끝난 경우에만
        # 임시 파일을 실제 ZIP 이름으로 변경
        partial_path.replace(destination)

    except Exception:
        # 다운로드 도중 실패하면 불완전한 파일을 삭제
        # 다음 실행에서 이 파일이 정상 ZIP으로 인식되는 것을 방지
        partial_path.unlink(
            missing_ok=True
        )
        raise


def _has_coco_images(image_dir):
    """폴더 안에 COCO JPG 이미지가 한 장 이상 있는지 확인"""

    image_dir = Path(image_dir)

    return (
        image_dir.is_dir()
        and next(
            image_dir.glob("*.jpg"),
            None,
        )
        is not None
    )


def _has_coco_annotation(annotation_file):
    """COCO annotation JSON 파일이 존재하고 비어 있지 않은지 확인"""

    annotation_file = Path(annotation_file)

    return (
        annotation_file.is_file()
        and annotation_file.stat().st_size > 0
    )


def _extract_zip(
    archive_path,
    destination,
):
    """ZIP 내부 경로를 검사한 뒤 지정한 폴더에 압축을 해제"""

    archive_path = Path(archive_path)
    destination = Path(destination)

    # 압축을 해제할 폴더가 없으면 생성
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_resolved = destination.resolve()

    print(
        f"[COCO2017] 압축 해제: "
        f"{archive_path.name} -> {destination}"
    )

    with zipfile.ZipFile(archive_path) as archive:

        # ZIP 내부에 ../ 등의 경로가 포함돼 있으면
        # 지정한 폴더 밖의 파일을 덮어쓸 수 있다.
        #
        # 따라서 모든 압축 파일 경로가 destination 내부인지
        # 먼저 확인한 후 압축을 해제
        for member in archive.infolist():
            member_path = (
                destination / member.filename
            ).resolve()

            if (
                member_path != destination_resolved
                and destination_resolved
                not in member_path.parents
            ):
                raise RuntimeError(
                    "ZIP에 안전하지 않은 경로가 "
                    f"포함돼 있습니다: {member.filename}"
                )

        archive.extractall(destination)


def _infer_coco_split(
    image_dir,
    annotation_file,
):
    """입력 경로에서 train2017 또는 val2017 split을 판별"""

    # 일반적인 이미지 폴더 이름은
    # train2017 또는 val2017
    image_split = Path(image_dir).name

    # annotation 파일 이름에서 확장자를 제거
    #
    # 예:
    # instances_train2017.json
    #             ↓
    # instances_train2017
    annotation_name = Path(
        annotation_file
    ).stem

    # 이미지 폴더 이름이 COCO split 이름이 아니면
    # 이미지 경로에서는 split을 찾지 못한 것으로 처리
    if image_split not in COCO2017_IMAGE_URLS:
        image_split = None

    # annotation 파일 이름에서 split을 찾는다.
    annotation_split = None

    for split in COCO2017_IMAGE_URLS:
        if annotation_name == f"instances_{split}":
            annotation_split = split
            break

    # 이미지 경로와 annotation 경로가 서로 다른 split을
    # 가리키면 잘못된 정답으로 학습할 수 있다.
    #
    # 예:
    # image_dir       = train2017
    # annotation_file = instances_val2017.json
    if (
        image_split is not None
        and annotation_split is not None
        and image_split != annotation_split
    ):
        raise ValueError(
            "이미지와 annotation의 COCO split이 "
            "서로 다릅니다: "
            f"{image_split}, {annotation_split}"
        )

    # 두 경로 중 하나에서 확인한 split을 사용
    detected_split = (
        image_split or annotation_split
    )

    if detected_split is not None:
        return detected_split

    # 두 경로에서 모두 split을 찾지 못하면
    # 어떤 COCO ZIP을 다운로드해야 하는지 결정할 수 없음
    raise ValueError(
        "COCO2017 split을 확인할 수 없습니다. "
        "image_dir은 'train2017' 또는 'val2017'로 "
        "끝나야 하며, annotation_file은 "
        "'instances_train2017.json' 또는 "
        "'instances_val2017.json'이어야 합니다."
    )


def _download_and_extract_resource(
    resource_name,
    url,
    archive_path,
    extract_dir,
):
    """COCO 리소스 하나를 다운로드하고 압축을 해제"""

    print(
        f"[COCO2017] {resource_name}이 없어 "
        "다운로드합니다."
    )

    # 공식 COCO 주소에서 ZIP 파일을 다운로드
    _download_file(
        url=url,
        destination=archive_path,
    )

    # 다운로드한 ZIP을 필요한 데이터 폴더에 압축 해제
    _extract_zip(
        archive_path=archive_path,
        destination=extract_dir,
    )

    # 압축 해제가 끝난 ZIP은 대용량이므로
    # 디스크 공간을 확보하기 위해 삭제
    Path(archive_path).unlink(
        missing_ok=True
    )

    print(
        f"[COCO2017] {resource_name} "
        "준비가 완료됐습니다."
    )


def ensure_coco2017_available(
    image_dir,
    annotation_file,
):
    """COCO2017을 확인하고 누락된 파일만 자동으로 준비"""

    image_dir = Path(image_dir)
    annotation_file = Path(annotation_file)

    # --------------------------------------------------
    # 1. 데이터 존재 여부 확인
    # --------------------------------------------------

    # 이미지 폴더 안에 JPG 이미지가 있는지 확인
    images_available = _has_coco_images(
        image_dir
    )

    # annotation JSON 파일이 존재하고
    # 비어 있지 않은지 확인
    annotation_available = (
        _has_coco_annotation(
            annotation_file
        )
    )

    # 이미지와 annotation이 모두 있다면
    # 네트워크 작업 없이 기존 데이터를 사용
    if images_available and annotation_available:
        print(
            "[COCO2017] 기존 데이터셋을 사용합니다: "
            f"{image_dir}"
        )
        return

    # --------------------------------------------------
    # 2. 사용할 COCO split 확인
    # --------------------------------------------------

    # 경로 이름을 이용해 train2017 또는
    # val2017 중 어떤 데이터인지 판별
    split = _infer_coco_split(
        image_dir=image_dir,
        annotation_file=annotation_file,
    )

    # 일반적인 프로젝트 구조:
    #
    # datasets/
    # └── coco/                 <- dataset_root
    #     ├── images/
    #     │   ├── train2017/
    #     │   └── val2017/
    #     └── annotations/
    dataset_root = image_dir.parent.parent

    # 다운로드 중인 ZIP은 데이터 폴더와 구분해
    # 임시 다운로드 폴더에 저장
    download_dir = (
        dataset_root / ".downloads"
    )

    # --------------------------------------------------
    # 3. 누락된 항목 출력
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

    print(
        "[COCO2017] 누락된 항목: "
        + ", ".join(missing_items)
    )

    # --------------------------------------------------
    # 4. 누락된 항목 다운로드
    # --------------------------------------------------

    try:
        # 이미지가 없을 때만 현재 split의
        # 대용량 이미지 ZIP을 다운로드
        if not images_available:
            image_archive = (
                download_dir / f"{split}.zip"
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
                archive_path=image_archive,

                # train2017.zip의 내부에는
                # train2017 폴더가 들어 있다.
                #
                # 따라서 images 폴더에 압축을 풀면
                # images/train2017 구조가 만들어진다.
                extract_dir=image_dir.parent,
            )

        # annotation이 없을 때만 train/val 공통
        # annotation ZIP을 다운로드
        if not annotation_available:
            annotation_archive = (
                download_dir
                / "annotations_trainval2017.zip"
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

                # annotation ZIP 내부에는
                # annotations 폴더가 들어 있음
                #
                # 따라서 coco 폴더에 압축을 풀면
                # coco/annotations 구조가 생성
                extract_dir=(
                    annotation_file.parent.parent
                ),
            )

    except Exception as error:
        # 회사나 학교의 네트워크 정책,
        # COCO 서버 상태, 디스크 공간 또는 폴더 권한 때문에
        # 자동 다운로드가 불가능할 수 있음
        #
        # 그런 경우 아래 오류 메시지에 표시되는 URL에서
        # ZIP을 직접 다운로드하고 안내된 위치에 압축을 풂
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
        # 모든 ZIP이 정상적으로 처리되어
        # 임시 다운로드 폴더가 비어 있으면 폴더도 삭제
        if (
            download_dir.is_dir()
            and not any(download_dir.iterdir())
        ):
            download_dir.rmdir()

    # --------------------------------------------------
    # 5. 다운로드 결과 최종 확인
    # --------------------------------------------------

    # 압축 해제 후 이미지가 실제로 생성됐는지 확인
    if not _has_coco_images(image_dir):
        raise FileNotFoundError(
            "압축 해제 후에도 COCO2017 이미지를 "
            "찾을 수 없습니다: "
            f"{image_dir}"
        )

    # 압축 해제 후 annotation 파일이
    # 실제로 생성됐는지 확인
    if not _has_coco_annotation(
        annotation_file
    ):
        raise FileNotFoundError(
            "압축 해제 후에도 COCO2017 annotation을 "
            "찾을 수 없습니다: "
            f"{annotation_file}"
        )

    print(
        f"[COCO2017] {split} 데이터셋 "
        "준비가 완료됐습니다."
    )


# --------------------------------------------------
# COCO2017 Dataset
# --------------------------------------------------

class Coco2017Dataset(CocoDetection):
    """COCO2017 객체 탐지용 Dataset."""

    def __init__(
        self,
        image_dir,
        annotation_file,
        image_size=640,
        auto_download=True,
    ):
        """
        Args:
            image_dir:
                COCO 이미지 폴더 경로

                예:
                    datasets/coco/images/train2017

            annotation_file:
                COCO annotation JSON 파일 경로

                예:
                    datasets/coco/annotations/
                    instances_train2017.json

            image_size:
                모델에 입력할 이미지의 높이와 너비

                640을 입력하면 모든 이미지를
                640×640으로 변환

            auto_download:
                True이면 데이터가 없는 경우
                COCO2017 이미지와 annotation을
                자동으로 다운로드

                False이면 자동 다운로드를 하지 않고
                전달받은 경로를 그대로 사용
        """

        # --------------------------------------------------
        # 1. COCO2017 데이터 준비
        # --------------------------------------------------

        # CocoDetection은 초기화 과정에서 annotation 파일을
        # 바로 읽으므로 super().__init__()보다 먼저
        # 데이터 존재 여부를 확인해야 함
        if auto_download:
            ensure_coco2017_available(
                image_dir=image_dir,
                annotation_file=annotation_file,
            )

        # --------------------------------------------------
        # 2. Torchvision CocoDetection 초기화
        # --------------------------------------------------

        # COCO JSON 파일을 읽고 이미지와
        # annotation을 연결
        super().__init__(
            root=image_dir,
            annFile=annotation_file,
        )

        # 모델 입력 크기로 사용할 값이므로
        # 1 이상의 값이어야 
        if image_size <= 0:
            raise ValueError(
                "image_size는 0보다 커야 합니다."
            )

        # 모델 입력 이미지 크기를 저장
        self.image_size = image_size

        # COCO category_id는 다음과 같이
        # 번호가 연속적이지 않음
        #
        # 1, 2, 3, ..., 11, 13, ...
        category_ids = sorted(
            self.coco.getCatIds()
        )

        # COCO category_id를 학습에 사용할
        # 0부터 시작하는 연속 label로 변환
        #
        # 예:
        # category_id 1  -> label 0
        # category_id 2  -> label 1
        # category_id 90 -> label 79
        self.category_id_to_label = {
            category_id: label
            for label, category_id
            in enumerate(category_ids)
        }

        # COCO2017 객체 탐지의 클래스 수는 일반적으로 80
        self.num_classes = len(
            category_ids
        )

    def __getitem__(self, index):
        """index번째 이미지와 객체 탐지 정답을 반환"""

        # --------------------------------------------------
        # 1. 이미지와 annotation 읽기
        # --------------------------------------------------

        # 부모 CocoDetection을 이용해
        # PIL 이미지와 annotation 리스트를 읽는다.
        image, annotations = (
            super().__getitem__(index)
        )

        # 입력 이미지 채널을 항상 RGB 3채널로 통일
        image = image.convert("RGB")

        # PIL image.size는
        # (width, height) 순서다.
        original_width, original_height = (
            image.size
        )

        # --------------------------------------------------
        # 2. Bounding box와 label 추출
        # --------------------------------------------------

        boxes = []
        labels = []

        for annotation in annotations:

            # iscrowd=1은 여러 객체가 하나의 영역으로
            # 묶여 있는 annotation
            #
            # 현재 구현에서는 일반 객체만 학습하기 위해 제외
            if annotation.get(
                "iscrowd",
                0,
            ) == 1:
                continue

            # COCO bbox 형식:
            #
            # [x, y, width, height]
            x, y, width, height = (
                annotation["bbox"]
            )

            # 너비나 높이가 0 이하인 잘못된 박스는 제외
            if width <= 0 or height <= 0:
                continue

            # COCO의 xywh 박스를 xyxy 박스로 변환
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height

            # 박스 좌표가 이미지 범위를 벗어나지 않게 제한
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
            # 너비나 높이가 사라진 박스는 제외
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

            # COCO category_id를 가져온다.
            category_id = (
                annotation["category_id"]
            )

            # COCO category_id를
            # 0부터 시작하는 학습 label로 변환
            label = (
                self.category_id_to_label[
                    category_id
                ]
            )

            labels.append(label)

        # --------------------------------------------------
        # 3. 박스와 label을 Tensor로 변환
        # --------------------------------------------------

        if boxes:
            # 객체가 한 개 이상 존재하는 경우:
            #
            # boxes shape:
            # [N, 4]
            boxes = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            # labels shape:
            # [N]
            labels = torch.tensor(
                labels,
                dtype=torch.int64,
            )

        else:
            # 객체가 없는 이미지도
            # 오류 없이 학습할 수 있도록 빈 Tensor를 만든다.
            boxes = torch.empty(
                (0, 4),
                dtype=torch.float32,
            )

            labels = torch.empty(
                (0,),
                dtype=torch.int64,
            )

        # --------------------------------------------------
        # 4. 이미지 resize
        # --------------------------------------------------

        # 현재 구현은 원본 비율을 유지하지 않고
        # image_size × image_size 크기로 직접 resize
        image = F.resize(
            image,
            [
                self.image_size,
                self.image_size,
            ],
        )

        # --------------------------------------------------
        # 5. 박스 좌표 resize
        # --------------------------------------------------

        if boxes.numel() > 0:

            # 원본 이미지에서 모델 입력 이미지로
            # 변경되는 가로 비율
            scale_x = (
                self.image_size
                / float(original_width)
            )

            # 원본 이미지에서 모델 입력 이미지로
            # 변경되는 세로 비율
            scale_y = (
                self.image_size
                / float(original_height)
            )

            # x1과 x2에 가로 비율을 적용
            boxes[:, [0, 2]] *= scale_x

            # y1과 y2에 세로 비율을 적용
            boxes[:, [1, 3]] *= scale_y

            # resize 이후에도 x좌표가
            # 이미지 범위를 벗어나지 않게 제한
            boxes[:, [0, 2]].clamp_(
                min=0.0,
                max=float(self.image_size),
            )

            # resize 이후에도 y좌표가
            # 이미지 범위를 벗어나지 않게 제한
            boxes[:, [1, 3]].clamp_(
                min=0.0,
                max=float(self.image_size),
            )

        # --------------------------------------------------
        # 6. PIL 이미지를 Tensor로 변환
        # --------------------------------------------------

        # 변환 결과:
        #
        # shape:
        # [3, H, W]
        #
        # dtype:
        # torch.uint8
        #
        # 값 범위:
        # 0~255
        image = F.pil_to_tensor(image)

        # 모델에 입력할 수 있도록 float32로 변경하고
        # 값의 범위를 0~1로 정규화
        image = image.to(
            dtype=torch.float32
        ) / 255.0

        # --------------------------------------------------
        # 7. 정답 딕셔너리 생성
        # --------------------------------------------------

        target = {
            # resize된 이미지 기준
            # xyxy 픽셀 좌표
            #
            # shape:
            # [N, 4]
            "boxes": boxes,

            # 객체별 학습 클래스 번호
            #
            # shape:
            # [N]
            #
            # 범위:
            # 0~79
            "labels": labels,

            # COCO JSON에 저장된
            # 이미지 고유 번호
            "image_id": torch.tensor(
                self.ids[index],
                dtype=torch.int64,
            ),
        }

        return image, target


# --------------------------------------------------
# DataLoader용 collate 함수
# --------------------------------------------------

def detection_collate_fn(batch):
    """이미지는 Tensor로 쌓고 target은 리스트로 유지"""

    # 모든 이미지는 Dataset에서 동일한 크기로 변환되므로
    # 하나의 배치 Tensor로 쌓을 수 있다.
    #
    # B개의 [3, H, W]
    #
    # ↓
    #
    # [B, 3, H, W]
    images = torch.stack(
        [
            sample[0]
            for sample in batch
        ],
        dim=0,
    )

    # 이미지마다 객체 수가 다르므로
    # target은 하나의 Tensor로 쌓지 않고 리스트로 유지
    targets = [
        sample[1]
        for sample in batch
    ]

    return images, targets


# --------------------------------------------------
# Dataset 동작 확인
# --------------------------------------------------

if __name__ == "__main__":

    # Dataset 생성 시 데이터가 없으면
    # train2017 이미지와 annotation을 자동으로 다운로드
    train_dataset = Coco2017Dataset(
        image_dir=(
            "datasets/coco/images/train2017"
        ),
        annotation_file=(
            "datasets/coco/annotations/"
            "instances_train2017.json"
        ),
        image_size=640,
        auto_download=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
        drop_last=False,
    )

    # --------------------------------------------------
    # Dataset 결과 확인
    # --------------------------------------------------

    image, target = train_dataset[0]

    print("===== Dataset 결과 =====")
    print(
        "전체 이미지 수:",
        len(train_dataset),
    )
    print(
        "클래스 수:",
        train_dataset.num_classes,
    )
    print(
        "image shape:",
        image.shape,
    )
    print(
        "image dtype:",
        image.dtype,
    )
    print(
        "boxes shape:",
        target["boxes"].shape,
    )
    print(
        "labels shape:",
        target["labels"].shape,
    )
    print(
        "image_id:",
        target["image_id"],
    )

    # --------------------------------------------------
    # DataLoader 결과 확인
    # --------------------------------------------------

    images, targets = next(
        iter(train_loader)
    )

    print()
    print("===== DataLoader 결과 =====")
    print(
        "images shape:",
        images.shape,
    )
    print(
        "images dtype:",
        images.dtype,
    )
    print(
        "target 개수:",
        len(targets),
    )
    print(
        "첫 번째 이미지 boxes:",
        targets[0]["boxes"].shape,
    )
    print(
        "두 번째 이미지 boxes:",
        targets[1]["boxes"].shape,
    )