# 라이브러리
import torch

from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


class DetectionTransform:

    def __init__(
        self,
        image_size=640,
        training=True,
        hflip_prob=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02
    ):

        # 입력 이미지 크기 저장
        self.image_size = image_size

        # 학습 여부 저장
        self.training = training

        # 좌우 반전 확률 저장
        self.hflip_prob = hflip_prob

        # 색상 증강 설정
        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )


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

        # 학습 시 색상 증강 적용
        if self.training:
            image = self.color_jitter(image)

        # 학습 시 확률적으로 좌우 반전
        if (
            self.training
            and torch.rand(1).item()
            < self.hflip_prob
        ):
            image, boxes = (
                self._horizontal_flip(
                    image,
                    boxes
                )
            )

        # 이미지와 Bbox에 LetterBox 적용
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