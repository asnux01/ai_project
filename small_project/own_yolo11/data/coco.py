# --------------------------------------------------
# 라이브러리
# --------------------------------------------------

import torch

# 여러 샘플을 하나의 배치로 가져오는 PyTorch 클래스
from torch.utils.data import DataLoader

# COCO 이미지와 JSON annotation을 읽어주는 Torchvision Dataset
from torchvision.datasets import CocoDetection

# 이미지 resize와 Tensor 변환에 사용하는 Torchvision 함수
from torchvision.transforms import functional as F


# --------------------------------------------------
# COCO2017 Dataset
# --------------------------------------------------

class Coco2017Dataset(CocoDetection):
    """
    COCO2017 객체 탐지용 Dataset.

    CocoDetection을 상속하여 다음 기능을 그대로 활용한다.

    1. COCO JSON 파일 읽기
    2. 이미지 ID와 annotation 연결
    3. 이미지 파일 불러오기
    4. 전체 데이터 개수 관리

    이 클래스에서 추가로 처리하는 작업:

    1. COCO bbox를 xyxy 형식으로 변환
    2. COCO category_id를 0~79로 변환
    3. 이미지를 640x640으로 resize
    4. 박스 좌표도 resize
    5. 이미지를 float32 Tensor로 변환
    """

    def __init__(
        self,
        image_dir,
        annotation_file,
        image_size=640,
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
                640x640으로 변환한다.
        """

        # CocoDetection의 __init__을 실행한다.
        #
        # 여기서 COCO JSON 파일을 읽고,
        # 이미지와 annotation을 연결하는 작업이 수행된다.
        super().__init__(
            root=image_dir,
            annFile=annotation_file,
        )

        if image_size <= 0:
            raise ValueError(
                "image_size는 0보다 커야 합니다."
            )

        # 모델 입력 이미지 크기
        self.image_size = image_size

        # COCO의 실제 category_id 목록을 가져온다.
        #
        # COCO category_id는 다음처럼 번호가 연속적이지 않다.
        #
        # 1, 2, 3, ..., 11, 13, ...
        #
        # 하지만 모델 출력 클래스는 0~79처럼
        # 연속된 번호를 사용하는 것이 편리하다.
        category_ids = sorted(
            self.coco.getCatIds()
        )

        # COCO category_id를 학습용 클래스 번호로 변환하는 표
        #
        # 예:
        #   category_id 1  -> label 0
        #   category_id 2  -> label 1
        #   category_id 90 -> label 79
        self.category_id_to_label = {
            category_id: label
            for label, category_id in enumerate(category_ids)
        }

        # 전체 클래스 수
        # COCO2017 객체 탐지는 일반적으로 80개 클래스다.
        self.num_classes = len(category_ids)

    def __getitem__(self, index):
        """
        index번째 이미지와 정답을 반환한다.

        Returns:
            image:
                Tensor[3, image_size, image_size]
                dtype=torch.float32
                값 범위 0.0~1.0

            target:
                {
                    "boxes": Tensor[N, 4],
                    "labels": Tensor[N],
                    "image_id": Tensor[]
                }

        N은 현재 이미지에 포함된 객체 수다.
        """

        # --------------------------------------------------
        # 1. 이미지와 annotation 가져오기
        # --------------------------------------------------

        # CocoDetection의 __getitem__을 사용한다.
        #
        # image:
        #   PIL 이미지 한 장
        #
        # annotations:
        #   현재 이미지에 들어 있는 모든 객체의 annotation 리스트
        image, annotations = super().__getitem__(index)

        # 모든 이미지를 RGB 3채널로 통일한다.
        image = image.convert("RGB")

        # PIL의 image.size는 (width, height) 순서다.
        original_width, original_height = image.size

        # --------------------------------------------------
        # 2. annotation에서 박스와 클래스 추출
        # --------------------------------------------------

        boxes = []
        labels = []

        # 이미지에 포함된 객체들을 하나씩 처리한다.
        for annotation in annotations:

            # iscrowd=1은 여러 객체가 하나의 영역으로
            # 묶여 있는 annotation이다.
            #
            # 첫 번째 구현에서는 일반 객체만 학습하기 위해 제외한다.
            if annotation.get("iscrowd", 0) == 1:
                continue

            # COCO bbox 형식:
            #
            # [x, y, width, height]
            #
            # x, y는 박스의 좌상단 좌표다.
            x, y, width, height = annotation["bbox"]

            # 너비나 높이가 0 이하인 잘못된 박스는 사용하지 않는다.
            if width <= 0 or height <= 0:
                continue

            # COCO의 xywh 형식을 xyxy 형식으로 변환한다.
            #
            # 변환 전:
            #   [x, y, width, height]
            #
            # 변환 후:
            #   [x1, y1, x2, y2]
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height

            # 박스가 이미지 바깥으로 벗어나지 않도록 제한한다.
            x1 = max(0.0, min(float(x1), original_width))
            y1 = max(0.0, min(float(y1), original_height))
            x2 = max(0.0, min(float(x2), original_width))
            y2 = max(0.0, min(float(y2), original_height))

            # 이미지 범위로 제한한 뒤
            # 너비 또는 높이가 사라진 박스는 제외한다.
            if x2 <= x1 or y2 <= y1:
                continue

            # 변환한 박스를 리스트에 추가한다.
            boxes.append([
                x1,
                y1,
                x2,
                y2,
            ])

            # COCO category_id를 가져온다.
            category_id = annotation["category_id"]

            # category_id를 학습용 0~79 label로 변환한다.
            label = self.category_id_to_label[
                category_id
            ]

            labels.append(label)

        # --------------------------------------------------
        # 3. 박스와 클래스를 Tensor로 변환
        # --------------------------------------------------

        if len(boxes) > 0:
            # 객체가 한 개 이상 존재하는 경우
            #
            # boxes:
            #   [N, 4], float32
            boxes = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            # labels:
            #   [N], int64
            labels = torch.tensor(
                labels,
                dtype=torch.int64,
            )

        else:
            # 객체가 하나도 없는 이미지도 오류 없이 처리한다.
            #
            # boxes shape:
            #   [0, 4]
            boxes = torch.empty(
                (0, 4),
                dtype=torch.float32,
            )

            # labels shape:
            #   [0]
            labels = torch.empty(
                (0,),
                dtype=torch.int64,
            )

        # --------------------------------------------------
        # 4. 이미지 크기 변경
        # --------------------------------------------------

        # Torchvision의 resize 함수를 이용하여
        # 이미지를 image_size × image_size로 변환한다.
        #
        # 현재는 비율을 유지하지 않는 강제 resize 방식이다.
        image = F.resize(
            image,
            [
                self.image_size,
                self.image_size,
            ],
        )

        # --------------------------------------------------
        # 5. 박스 좌표 크기 변경
        # --------------------------------------------------

        # 이미지 크기가 변경됐으므로
        # 박스 좌표에도 같은 비율을 적용해야 한다.
        if boxes.numel() > 0:

            # 가로 방향 크기 변경 비율
            scale_x = (
                self.image_size
                / float(original_width)
            )

            # 세로 방향 크기 변경 비율
            scale_y = (
                self.image_size
                / float(original_height)
            )

            # boxes 형식:
            #
            # [x1, y1, x2, y2]
            #
            # 0번과 2번 열은 x좌표다.
            boxes[:, [0, 2]] *= scale_x

            # 1번과 3번 열은 y좌표다.
            boxes[:, [1, 3]] *= scale_y

            # resize 이후에도 좌표가 이미지 범위를
            # 벗어나지 않도록 제한한다.
            boxes[:, [0, 2]].clamp_(
                min=0.0,
                max=float(self.image_size),
            )

            boxes[:, [1, 3]].clamp_(
                min=0.0,
                max=float(self.image_size),
            )

        # --------------------------------------------------
        # 6. PIL 이미지를 Tensor로 변환
        # --------------------------------------------------

        # PIL Image를 Tensor로 변환한다.
        #
        # 변환 결과:
        #   shape = [3, H, W]
        #   dtype = torch.uint8
        #   값 범위 = 0~255
        image = F.pil_to_tensor(image)

        # uint8을 float32로 변환하고
        # 255로 나누어 값의 범위를 0~1로 만든다.
        image = image.to(
            dtype=torch.float32
        ) / 255.0

        # --------------------------------------------------
        # 7. 정답 딕셔너리 구성
        # --------------------------------------------------

        target = {
            # resize된 이미지 기준 xyxy 픽셀 좌표
            #
            # shape:
            #   [N, 4]
            "boxes": boxes,

            # 객체별 클래스 번호
            #
            # shape:
            #   [N]
            #
            # 값 범위:
            #   0~79
            "labels": labels,

            # COCO JSON에 저장된 이미지 고유 번호
            #
            # 나중에 COCO 평가를 할 때 사용할 수 있다.
            "image_id": torch.tensor(
                self.ids[index],
                dtype=torch.int64,
            ),
        }

        return image, target


# --------------------------------------------------
# 객체 탐지용 collate 함수
# --------------------------------------------------

def detection_collate_fn(batch):
    """
    Dataset에서 가져온 여러 샘플을 하나의 배치로 결합한다.

    batch_size=2일 때 batch는 다음 형태다.

    [
        (image_0, target_0),
        (image_1, target_1)
    ]

    이미지는 모두 같은 크기이므로 하나의 Tensor로 합친다.

    각 이미지의 객체 수는 다르므로
    target은 리스트 형태로 유지한다.
    """

    # 각 샘플의 첫 번째 값인 이미지를 가져와 쌓는다.
    #
    # [3, 640, 640] 이미지 B개
    #       ↓
    # [B, 3, 640, 640]
    images = torch.stack(
        [
            sample[0]
            for sample in batch
        ],
        dim=0,
    )

    # 각 이미지의 target은 객체 수가 다르므로
    # Tensor로 쌓지 않고 리스트로 유지한다.
    targets = [
        sample[1]
        for sample in batch
    ]

    return images, targets


# --------------------------------------------------
# Dataset과 DataLoader 실행 예제
# --------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------
    # 학습 Dataset
    # --------------------------------------------------

    train_dataset = Coco2017Dataset(
        image_dir=(
            "datasets/coco/images/train2017"
        ),
        annotation_file=(
            "datasets/coco/annotations/"
            "instances_train2017.json"
        ),
        image_size=640,
    )

    # --------------------------------------------------
    # 학습 DataLoader
    # --------------------------------------------------

    train_loader = DataLoader(
        dataset=train_dataset,

        # 한 번에 가져올 이미지 개수
        batch_size=2,

        # 학습 데이터 순서를 섞는다.
        shuffle=True,

        # 처음에는 오류 확인이 쉬운 0으로 설정한다.
        num_workers=0,

        # GPU 사용 시 메모리 전송을 도울 수 있다.
        pin_memory=torch.cuda.is_available(),

        # 객체 탐지용 배치 결합 함수
        collate_fn=detection_collate_fn,

        # 마지막 배치가 2장보다 작아도 사용한다.
        drop_last=False,
    )

    # --------------------------------------------------
    # Dataset 결과 확인
    # --------------------------------------------------

    image, target = train_dataset[0]

    print("===== Dataset 결과 =====")

    print("전체 이미지 수:", len(train_dataset))
    print("클래스 수:", train_dataset.num_classes)

    print("image shape:", image.shape)
    print("image dtype:", image.dtype)

    print("boxes shape:", target["boxes"].shape)
    print("labels shape:", target["labels"].shape)

    print("image_id:", target["image_id"])

    # --------------------------------------------------
    # DataLoader 결과 확인
    # --------------------------------------------------

    images, targets = next(
        iter(train_loader)
    )

    print()
    print("===== DataLoader 결과 =====")

    print("images shape:", images.shape)
    print("images dtype:", images.dtype)

    print("target 개수:", len(targets))

    print(
        "첫 번째 이미지 boxes:",
        targets[0]["boxes"].shape,
    )

    print(
        "두 번째 이미지 boxes:",
        targets[1]["boxes"].shape,
    )