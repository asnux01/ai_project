# 라이브러리
from pathlib import Path

import random

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
        image_size=640,
        mosaic_prob=0.0,
        close_mosaic=0
    ):
        
        super().__init__()

        # 데이터 경로 저장
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)

        # Transform 저장
        self.transforms = transforms
        
        # Mosaic 설정 저장
        self.image_size = int(image_size)
        self.mosaic_prob = float(mosaic_prob)
        self.close_mosaic = int(close_mosaic)
        
        # 현재 Epoch 정보
        self.current_epoch = 0
        self.total_epochs = None
        
        # Mosaic 확률 검사
        if not 0.0 <= self.mosaic_prob <= 1.0:
            raise ValueError(
                "mosaic_prob은 0과 1 사이여야 합니다."
            )

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

    
    def set_epoch(
        self,
        epoch,
        total_epochs
    ):
        
        # 현재 Epoch 저장
        self.current_epoch = int(epoch)
        
        # 전체 Epoch 저장
        self.total_epochs = int(total_epochs)
    
    
    def _mosaic_enabled(self):
        
        # Mosaic 확률이 0이면 사용하지 않음
        if self.mosaic_prob <= 0.0:
            return False
            
        # 전체 Epoch 정보가 없으면 Mosaic 사용
        if self.total_epochs is None:
            return True
        
        # close_mosaic가 0이면 마지막까지 Mosaic 사용
        if self.close_mosaic <= 0:
            return True
        
        # 마지막 close_mosaic Epoch에서는 Mosaic 비활성화
        return (
            self.current_epoch
            < self.total_epochs - self.close_mosaic
        )
        
    
    def _load_mosaic(self, index):
        
        # Mosaic 기준 이미지 크기
        image_size = self.image_size
        
        # 첫 번째 이미지는 현재 Dataset Index 사용
        mosaic_indices = [index]
        
        # 나머지 3개의 이미지 Index를 무작위 선택
        mosaic_indices.extend(
            random.choices(
                range(len(self.image_ids)),
                k=3
            )
        )        
        
        # 2배 크기의 Mosaic Canvas 생성
        mosaic_image = Image.new(
            "RGB",
            (self.image_size * 2, image_size * 2),
            color=(114, 114, 114)
        )
        
        # Mosaic 중심 위치 무작위 설정
        mosaic_center_x = random.randint(
            image_size // 2,
            image_size * 3 // 2
        )
        
        mosaic_center_y = random.randint(
            image_size // 2,
            image_size * 3 // 2
        )
        
        # 전체 Bbox와 Class 저장
        mosaic_boxes = []
        mosaic_classes = []
        
        # Mosaic에 사용할 이미지 4장 처리
        for mosaic_index, dataset_index in enumerate(mosaic_indices):
            
            # Transform 적용 전 원본 이미지와 label 가져오기
            sample = self.get_image_and_label(dataset_index)
            
            image = sample["img"]
            boxes = sample["bboxes"].clone()
            classes = sample["cls"].clone()
            
            # 원본 이미지 크기
            image_width, image_height = image.size
            
            # 원본 비유을 유지하며 image_size 기준으로 Resize
            resize_ratio = (
                image_size
                / max(image_width, image_height)
            )
            
            resized_width = max(
                int(round(image_width * resize_ratio)), 1
            )
            
            resized_height = max(
                int(round(image_height * resize_ratio)), 1
            )
            
            # 이미지 Resize
            image = image.resize(
                (resized_width, resized_height),
                Image.Resampling.BILINEAR
            )
            
            # Bbox도 동일한 비율로 Resize
            if boxes.numel() > 0:
                
                boxes[:, [0, 2]] *= resize_ratio
                boxes[:, [1, 3]] *= resize_ratio
            
            # Resize 이후 크기
            image_width = resized_width
            image_height = resized_height
            
            # 왼쪽 위 미지
            if mosaic_index == 0:
                
                x1_canvas = max(
                    mosaic_center_x - image_width, 0
                )
                
                y1_canvas = max(
                    mosaic_center_y - image_height, 0
                )
                
                x2_canvas = mosaic_center_x
                y2_canvas = mosaic_center_y
                
                x1_image = image_width - (x2_canvas - x1_canvas)
                y1_image = image_height - (y2_canvas - y1_canvas)
                x2_image = image_width
                y2_image = image_height
            
            # 오른쪽 위 이미지
            elif mosaic_index == 1:
                
                x1_canvas = mosaic_center_x
                y1_canvas = max(mosaic_center_y - image_height, 0)
                x2_canvas = min(
                    mosaic_center_x + image_width,
                    image_size * 2
                )
                y2_canvas = mosaic_center_y
                
                x1_image = 0
                y1_image = image_height - (y2_canvas - y1_canvas)
                x2_image = min(image_width, x2_canvas - x1_canvas)
                y2_image = image_height
            
            # 왼쪽 아래 이미지
            elif mosaic_index == 2:
                
                x1_canvas = max(mosaic_center_x - image_width, 0)
                y1_canvas = mosaic_center_y
                x2_canvas = mosaic_center_x
                y2_canvas = min(
                    mosaic_center_y + image_height,
                    image_size * 2
                )
                                
                x1_image = image_width - (x2_canvas - x1_canvas)
                y1_image = 0
                x2_image = image_width
                y2_image = min(image_height, y2_canvas - y1_canvas)
                
            # 오른쪽 아래 이미지
            else:
                            
                x1_canvas = mosaic_center_x
                y1_canvas = mosaic_center_y
                x2_canvas = min(
                    mosaic_center_x + image_width,
                    image_size * 2
                )
                y2_canvas = min(
                    mosaic_center_y + image_height,
                    image_size * 2
                )
                                            
                x1_image = 0
                y1_image = 0
                x2_image = min(image_width, x2_canvas - x1_canvas)
                y2_image = min(image_height, y2_canvas - y1_canvas)
            
            # 실제 Mosaic에 들어갈 이미지 영역 자르기
            image_crop = image.crop(
                (x1_image, y1_image, x2_image, y2_image)
            )
            
            # Mosaic Canvas에 이미지 붙이기
            mosaic_image.paste(
                image_crop, (x1_canvas, y1_canvas)
            )
            
            # Bbox가 존재하는 경우 Mosaic 좌표로 이동
            if boxes.numel() > 0:
                
                # 원본 Image 좌표애서 Mosaic Canvas 좌표로 이동량 계산
                pad_x = x1_canvas - x1_image
                pad_y = y1_canvas - y1_image
                
                # Bbox를 Mosaic Canvas 좌표로 이동
                boxes[:, [0, 2]] += pad_x
                boxes[:, [1, 3]] += pad_y
                
                # Bbox 저장
                mosaic_boxes.append(boxes)
                
                # Class 저장
                mosaic_classes.append(classes)
        
        # Bbox가 하나 이상 존재하는 경우
        if len(mosaic_boxes) > 0:
            
            # 네 이미지으 Bbox 결합
            boxes = torch.cat(
                mosaic_boxes,
                dim=0
            )
            
            # 네 이미지의 Class 결합
            classes = torch.cat(
                mosaic_classes,
                dim=0
            )
            
            # Crop 전 Bbox 크기 저장
            original_widths = (
                boxes[:, 2] - boxes[:, 0]
            ).clamp(min=1e-6)
            
            original_heights = (
                boxes[:, 3] - boxes[:, 1]
            ).clamp(min=1e-6)
            
            original_areas = original_widths * original_heights
            
            # 2x Canvas의 중앙 image_size 영역만 사용할 것이므로 Offset 계산
            crop_offset = image_size // 2
            
            # Crop 위치만큼 Bbox 좌표 이동
            boxes[:, [0, 2]] -= crop_offset
            boxes[:, [1, 3]] -= crop_offset
            
            # 최종 Mosaic 이미지 범위 내부로 Bbox 제한
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_size)

            # Crop 이후 Bbox 크기 계산
            new_widths = (
                boxes[:, 2] - boxes[:, 0]
            ).clamp(min=0.0)
            
            new_heights = (
                boxes[:, 3] - boxes[:, 1]
            ).clamp(min=0.0)
            
            # Crop 이후 Bbox 면적 계산
            new_areas = new_widths * new_heights
            
            # 기존 Bbox 면적 대비 유지된 비율 계산
            retained_area_ratio = new_areas / (original_areas + 1e-6)
            
            # Bbox 가로셀로 비율 계산
            aspect_ratio = torch.maximum(
                new_widths / (new_heights + 1e-6),
                new_heights / (new_widths + 1e-6)
            )
            
            # 지나치게 작거나 심하게 잘린 Bbox 제거
            keep_mask = (
                (new_widths > 2.0)
                & (new_heights > 2.0)
                & (retained_area_ratio > 0.10)
                & (aspect_ratio < 100.0)
            )
            
            # 유효한 Bbox와 Class만 유지
            boxes = boxes[keep_mask]
            classes = classes[keep_mask]
            
        else:
            
            # Bbox가 하나도 없는 경우 빈 Tensor 생성
            boxes = torch.zeros(
                (0, 4), dtype=torch.float32
            )
            
            # Class가 하나도 없는 경우 빈 Tensor 생성
            classes = torch.zeros(
                (0,), dtype=torch.int64
            )
        
        # 2x Canvas의 중앙 640x640 영역 Crop
        crop_offset = image_size // 2
        
        mosaic_image = mosaic_image.crop(
            (
                crop_offset,
                crop_offset,
                crop_offset + image_size,
                crop_offset + image_size
            )
        )
        
        # Training영 Mosaic Sample 구성
        sample = {
            "img": mosaic_image,
            "bboxes": boxes,
            "cls": classes,
            
            # Training에서는 실제 COCO 평가에 사용되지 않으므로
            # 기준 이미지의 ID를 그대로 저장
            "image_id": self.image_ids[index],
            
            # Mosaic Image임을 표시
            "im_file": "mosaic",
            
            # Mosaic 결과 자체가 image_size x image_size이므로 해당 크기 저장
            "ori_shape": (image_size, image_size)
        }
        
        return sample
    
    
    def __getitem__(self, index):

        # Mosaic 사용 여부 결정
        use_mosaic = (
            self._mosaic_enabled()
            and random.random() < self.mosaic_prob
        )
        
        # Mosaic이 활성화된 경우 이미지 4장을 결합
        if use_mosaic:
            sample = self._load_mosaic(index)
        
        # Mosaic을 사용하지 않는 경우 일반 이미지 사용
        else:
            sample = self.get_image_and_label(index)

        # 이미지와 label에 Transform 적용
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def __len__(self):

        # 전체 이미지 개수 반환
        return len(self.image_ids)