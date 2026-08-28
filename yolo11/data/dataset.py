# 라이브러리
from pathlib import Path

import torch

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCODetectionDataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_file,
        transforms=None,
    ):
        super().__init__()

        # 데이터 경로 저장
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)

        # Transform 저장
        self.transforms = transforms

        # 이미지 디렉터리 확인
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        # Annotation 파일 확인
        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}"
            )

        # COCO annotation 로드
        self.coco = COCO(str(self.annotation_file))

        # 이미지 ID 목록 생성
        self.image_ids = sorted(self.coco.getImgIds())

        # COCO category ID를 class index로 변환
        self.category_id_to_class_index = self._build_category_mapping()


    def _build_category_mapping(self):

        # COCO category ID 목록 가져오기
        category_ids = sorted(self.coco.getCatIds())

        # Category ID를 0부터 시작하는 class index로 변환
        category_mapping = {
            category_id: class_index
            for class_index, category_id
            in enumerate(category_ids)
        }

        return category_mapping


    def _get_image_info(self, index):

        # Dataset index에 해당하는 이미지 ID 가져오기
        image_id = self.image_ids[index]

        # 이미지 정보 가져오기
        image_info = self.coco.loadImgs(image_id)[0]

        return image_id, image_info


    def _load_image(self, image_info):

        # 이미지 파일 경로 생성
        image_path = self.image_dir / image_info["file_name"]

        # 이미지 파일 확인
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # 이미지를 RGB 형식으로 로드
        image = Image.open(image_path).convert("RGB")

        return image, image_path


    def _load_annotations(self, image_id):

        # 이미지에 해당하는 annotation ID 가져오기
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id])

        # Annotation 정보 가져오기
        annotations = self.coco.loadAnns(annotation_ids)

        return annotations


    def _parse_annotations(
        self,
        annotations,
        image_width,
        image_height,
    ):

        # Bbox와 class 저장 공간 생성
        boxes = []
        classes = []

        for annotation in annotations:

            # Crowd annotation 제외
            if annotation.get("iscrowd", 0):
                continue

            # COCO bbox 정보 가져오기
            x, y, width, height = annotation["bbox"]

            # 잘못된 bbox 제거
            if width <= 0 or height <= 0:
                continue

            # xywh를 xyxy 형식으로 변환
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height

            # Bbox를 이미지 범위 내부로 제한
            x1 = max(0.0, min(x1, image_width))
            y1 = max(0.0, min(y1, image_height))
            x2 = max(0.0, min(x2, image_width))
            y2 = max(0.0, min(y2, image_height))

            # 변환 후 잘못된 bbox 제거
            if x2 <= x1 or y2 <= y1:
                continue

            # COCO category ID 가져오기
            category_id = annotation["category_id"]

            # Category ID를 class index로 변환
            class_index = self.category_id_to_class_index[category_id]

            # Bbox 정보 저장
            boxes.append([x1, y1, x2, y2])

            # Class 정보 저장
            classes.append(class_index)

        # Bbox를 Tensor로 변환
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)

        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # Class를 Tensor로 변환
        if classes:
            classes = torch.tensor(classes, dtype=torch.int64)

        else:
            classes = torch.zeros((0,), dtype=torch.int64)

        return boxes, classes


    def get_image_and_label(self, index):

        # 이미지 정보 가져오기
        image_id, image_info = self._get_image_info(index)
        

        # 이미지 로드
        image, image_path = self._load_image(image_info)

        # Annotation 로드
        annotations = self._load_annotations(image_id)

        # Annotation을 학습용 정보로 변환
        boxes, classes = (
            self._parse_annotations(
                annotations=annotations,
                image_width=image_info["width"],
                image_height=image_info["height"]
            )
        )

        # 이미지와 관련 정보를 하나의 sample로 구성
        sample = {
            "img": image,
            "bboxes": boxes,
            "cls": classes,
            "image_id": image_id,
            "im_file": str(image_path),
            "ori_shape": (
                image_info["height"],
                image_info["width"]
            )
        }

        return sample


    def __getitem__(self, index):

        # 이미지와 label 정보 가져오기
        sample = self.get_image_and_label(index)

        # 이미지와 label에 Transform 적용
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def __len__(self):

        # 전체 이미지 개수 반환
        return len(self.image_ids)