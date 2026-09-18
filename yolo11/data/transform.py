# 라이브러리
import math
import random

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision.transforms import functional as TF


class DetectionTransform:

    def __init__(
        self,
        image_size=640,
        training=True,
        hflip_prob=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        degrees=0.0,
        shear=0.0,
        perspective=0.0
    ):

        # 입력 이미지 크기 저장
        self.image_size = image_size

        # 학습 여부 저장
        self.training = training

        # 좌우 반전 확률 저장
        self.hflip_prob = hflip_prob

        # HSV 색상 증강 설정
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        
        # Random Perspective 설정
        self.translate = translate
        self.scale = scale
        self.degrees = degrees
        self.shear = shear
        self.perspective = perspective

    
    def _hsv_augment(self, image):
        
        # Hue 변화량 무작위 Sampling
        hue_factor = (
            (torch.rand(1).item() * 2.0 - 1.0)
            * self.hsv_h
        )
        
        # Saturation 변화 비율 무작위 Sampling
        saturation_factor = (
            1.0 + (torch.rand(1).item() * 2.0 - 1.0)
            * self.hsv_s
        )
        
        # Brightness(Value) 변화 비율 무작위 Sampling
        value_factor = (
            1.0 + (torch.rand(1).item() * 2.0 - 1.0)
            * self.hsv_v
        )
        
        # 음수 배율 방지
        saturation_factor = max(saturation_factor, 0.0)
        value_factor = max(value_factor, 0.0)
        
        # Hue 변환
        image = TF.adjust_hue(image, hue_factor)
        
        # Saturation 변환
        image = TF.adjust_saturation(image, saturation_factor)
        
        # Brightness(Value) 변환
        image = TF.adjust_brightness(image, value_factor)
        
        return image
    
    
    def _random_perspective(
        self,
        image,
        boxes,
        classes,
        border=(0, 0)
    ):
        
        # 원본 이미지 크기
        image_width, image_height = image.size
        
        # Border
        border_y, border_x = border
        
        # 출력 이미지 크기
        output_width = image_width + border_x * 2
        output_height = image_height + border_y * 2
        
        # Center Matrix
        center_matrix = np.eye(
            3,
            dtype=np.float32
        )
        
        center_matrix[0, 2] = -image_width / 2
        center_matrix[1, 2] = -image_height / 2
        
        # Perspective Matrix
        perspective_matrix = np.eye(
            3,
            dtype=np.float32
        )
        
        perspective_matrix[2, 0] = random.uniform(
            -self.perspective,
            self.perspective
        )
        
        perspective_matrix[2, 1] = random.uniform(
            -self.perspective,
            self.perspective
        )
        
        # Rotation / Scale Matrix
        rotation_matrix = np.eye(
            3,
            dtype=np.float32
        )
        
        angle = random.uniform(
            -self.degrees,
            self.degrees
        )
        
        scale_factor = random.uniform(
            1.0 - self.scale,
            1.0 + self.scale
        )
        
        rotation_matrix[:2] = (
            cv2.getRotationMatrix2D(
                center=(0, 0),
                angle=angle,
                scale=scale_factor
            )
        )
        
        # Shear Matrix
        shear_matrix = np.eye(
            3,
            dtype=np.float32
        )
        
        shear_matrix[0, 1] = math.tan(
            random.uniform(
                -self.shear,
                self.shear
            )
            * math.pi / 180
        )
        
        shear_matrix[1, 0] = math.tan(
            random.uniform(
                -self.shear,
                self.shear
            )
            * math.pi /180
        )
        
        # Translation Matrix
        translation_matrix = np.eye(
            3,
            dtype=np.float32
        )
        
        translation_matrix[0, 2] = random.uniform(
            0.5 - self.translate,
            0.5 + self.translate
        ) * output_width
        
        translation_matrix[1, 2] = random.uniform(
            0.5 - self.translate,
            0.5 + self.translate
        ) * output_height
        
        # Transform Matrix
        transform_matrix = (
            translation_matrix
            @ shear_matrix
            @ rotation_matrix
            @ perspective_matrix
            @ center_matrix
        )
        
        # PIL Image를 Numpy로 변환
        image_array = np.asarray(image)
        
        # Perspective Transform
        if self.perspective:
            
            image_array = cv2.warpPerspective(
                image_array,
                transform_matrix,
                dsize=(output_width, output_height),
                borderValue=(114, 114, 114)
            )
        
        # Affine Transform
        else:
            image_array = cv2.warpAffine(
                image_array,
                transform_matrix[:2],
                dsize=(output_width, output_height),
                borderValue=(114, 114, 114)
            )
        
        # Numpy를 PIL Image로 변환
        image = Image.fromarray(image_array)
        
        # Bbox가 없으면 바로 반환
        if boxes.numel() == 0:
            return image, boxes, classes
        
        # 기존 Bbox 저장
        original_boxes = boxes.clone()
        
        # Bbox 개수
        num_boxes = boxes.shape[0]
        
        # Bbox 네 모서리 생성
        corners = torch.ones(
            (num_boxes * 4, 3),
            dtype=boxes.dtype,
            device=boxes.device
        )
        
        corners[:, :2] = boxes[
            :,
            [
                0, 1,
                2, 3,
                0, 3,
                2, 1
            ]
        ].reshape(num_boxes * 4, 2)
        
        # Transfrom Matrix를 Tensor로 변환
        matrix = torch.from_numpy(
            transform_matrix
        ).to(
            device=boxes.device,
            dtype=boxes.dtype
        )
        
        # Bbox 좌표 변환
        corners = corners @ matrix.T
        
        # Perspective 좌표 변환
        if self.perspective:
            
            corners = (
                corners[:, :2]
                / corners[:, 2:3]
            )
        
        else:
            
            corners = corners[:, :2]
        
        # Bbox별 네 모서리 복원
        corners = corners.reshape(
            num_boxes,
            8
        )
        
        # X 좌표
        x = corners[
            :,
            [0, 2, 4, 6]
        ]
        
        # Y 좌표
        y = corners[
            :,
            [1, 3, 5, 7]
        ]
        
        # 새로운 Bbox 생성
        boxes = torch.stack(
            (
                x.min(dim=1).values,
                y.min(dim=1).values,
                x.max(dim=1).values,
                y.max(dim=1).values
            ),
            dim=1
        )
        
        # Bbox 좌표 제한
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
            0,
            output_width
        )
        
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
            0,
            output_height
        )
        
        # 기존 Bbox 크기
        old_width = (
            original_boxes[:, 2]
            - original_boxes[:, 0]
        ).clamp(min=1e-6)
        
        old_height = (
            original_boxes[:, 3]
            - original_boxes[:, 1]
        ).clamp(min=1e-6)
        
        # Scale 적용
        old_width *= scale_factor
        old_height *= scale_factor
        
        # 변환 후 Bbox 크기
        new_width = (
            boxes[:, 2]
            - boxes[:, 0]
        ).clamp(min=0.0)
        
        new_height = (
            boxes[:, 3]
            - boxes[:, 1]
        ).clamp(min=0.0)
        
        # Bbox 면적 비율
        area_ratio = (
            new_width
            * new_height
            / (old_width * old_height + 1e-6)
        )
        
        # Bbox 가로세로 비율
        aspect_ratio = torch.maximum(
            new_width / (new_height + 1e-6),
            new_height / (new_width + 1e-6)
        )
        
        # 유효한 Bbox 선택
        keep_mask = (
            (new_width > 2.0)
            & (new_height > 2.0)
            & (area_ratio > 0.10)
            & (aspect_ratio < 100.0)
        )
        
        # 유효한 Bbox와 Class만 유지
        boxes = boxes[keep_mask]
        classes = classes[keep_mask]
        
        return image, boxes, classes
                
        
    def _horizontal_flip(
        self,
        image,
        boxes
    ):

        # 이미지 너비 가져오기
        image_width = image.width

        # 이미지 좌우 반전
        image = TF.hflip(image)

        # Bbox가 없는 경우 바로 반환
        if boxes.numel() == 0:
            return image, boxes

        # 기존 x 좌표 저장
        x1 = boxes[:, 0].clone()
        x2 = boxes[:, 2].clone()

        # Bbox x 좌표 좌우 반전
        boxes[:, 0] = image_width - x2
        boxes[:, 2] = image_width - x1

        return image, boxes


    def _letterbox(
        self,
        image,
        boxes
    ):

        # 원본 이미지 크기 가져오기
        original_width, original_height = image.size

        # Resize 비율 계산
        scale = min(
            self.image_size / original_width,
            self.image_size / original_height
        )

        # Resize 후 이미지 크기 계산
        resized_width = round(original_width * scale)
        resized_height = round(original_height * scale)

        # 이미지 비율을 유지하며 Resize
        image = TF.resize(
            image,
            [resized_height, resized_width]
        )

        # Padding 크기 계산
        pad_width = self.image_size - resized_width
        pad_height = self.image_size - resized_height

        # Padding 위치 계산
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        # 이미지에 Padding 적용
        image = TF.pad(
            image,
            [pad_left, pad_top, pad_right, pad_bottom],
            fill=114
        )

        # Bbox가 없는 경우 바로 반환
        if boxes.numel() == 0:
            return image, boxes

        # Bbox를 Resize 비율에 맞게 변환
        boxes[:, [0, 2]] *= scale
        boxes[:, [1, 3]] *= scale

        # Padding 위치만큼 Bbox 이동
        boxes[:, [0, 2]] += pad_left
        boxes[:, [1, 3]] += pad_top

        # Bbox를 이미지 범위 내부로 제한
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
            0,
            self.image_size
        )

        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
            0,
            self.image_size
        )

        return image, boxes


    def _remove_invalid_boxes(
        self,
        boxes,
        classes
    ):

        # Bbox가 없는 경우 바로 반환
        if boxes.numel() == 0:
            return boxes, classes

        # Bbox 너비와 높이 계산
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        # 정상적인 Bbox만 선택
        valid_mask = (widths > 0) & (heights > 0)

        boxes = boxes[valid_mask]
        classes = classes[valid_mask]

        return boxes, classes


    def _to_tensor(self, image):

        # PIL 이미지를 uint8 Tensor로 변환
        image = TF.pil_to_tensor(image)

        # 이미지를 float32로 변환
        image = image.to(dtype=torch.float32)

        # 픽셀 값을 0~1 범위로 변환
        image = image / 255.0

        return image


    def __call__(self, sample):

        # Sample 데이터 가져오기
        image = sample["img"]
        boxes = sample["bboxes"].clone()
        classes = sample["cls"].clone()

        # Mosaic Border 가져오기
        mosaic_border = sample.pop(
            "mosaic_border",
            None
        )
        
        # 학습 Transform
        if self.training:
            
            # 일반 이미지이면 LetterBox 적용
            if mosaic_border is None:
                
                image, boxes = self._letterbox(
                    image,
                    boxes
                )
                
                border = (0, 0)
            
            # Mosaic 이미지이면 Border 사용
            else:
                
                border = mosaic_border
            
            # Random Perspective 적용
            image, boxes, classes = (
                self._random_perspective(
                    image=image,
                    boxes=boxes,
                    classes=classes,
                    border=border
                )
            )
            
            # HSV 색상 증강 적용
            image = self._hsv_augment(image)
            
            # 확률적으로 좌우 반전
            if (
                torch.rand(1).item()
                < self.hflip_prob
            ):
                
                image, boxes = (
                    self._horizontal_flip(
                        image,
                        boxes
                    )
                )
        
        # Validation LetterBox
        else:
            
            image, boxes = self._letterbox(
                image,
                boxes
            )
            
        # 잘못된 Bbox 제거
        boxes, classes = (
            self._remove_invalid_boxes(
                boxes,
                classes
            )
        )

        # 이미지를 Tensor로 변환
        image = self._to_tensor(image)

        # 변환된 데이터 저장
        sample["img"] = image
        sample["bboxes"] = boxes
        sample["cls"] = classes

        # Resize 후 이미지 크기 저장
        sample["resized_shape"] = (
            self.image_size,
            self.image_size
        )

        return sample