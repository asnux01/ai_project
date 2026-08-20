"""학습과 동일한 LetterBox를 적용해 이미지 배치를 만든다."""

# 이미지별 좌표 복원 정보를 명시적으로 묶는다.
from dataclasses import dataclass
from pathlib import Path

# Ultralytics Results는 원본 이미지를 NumPy BGR 배열로 사용한다.
import numpy as np

# 전처리 Tensor와 빈 detection target을 만든다.
import torch

# 다양한 이미지 포맷 로딩 및 EXIF 회전을 처리한다.
from PIL import Image, ImageOps

# 학습/검증에서 검증된 동일 LetterBox 구현을 재사용한다.
from data.transforms import DetectionTransform


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """후처리에서 원본 좌표와 이미지를 복원할 메타데이터."""

    # 원본 파일과 Results 시각화에 사용할 BGR 이미지
    path: Path
    original_bgr: np.ndarray

    # 원본 (height, width), resize 비율, (left, top, right, bottom) padding
    original_shape: tuple[int, int]
    scale: float
    padding: tuple[int, int, int, int]

    @property
    def ratio_pad(self):
        """Ultralytics ops.scale_boxes가 사용하는 ratio_pad 형식."""

        # scale_boxes는 좌표 복원에 왼쪽과 위쪽 padding만 직접 사용한다.
        left, top, _, _ = self.padding
        return ((self.scale, self.scale), (left, top))


class InferencePreprocessor:
    """PIL 로딩, RGB 변환, LetterBox, BCHW 배치 생성을 담당한다."""

    def __init__(self, image_size):
        self.image_size = int(image_size)

        # training=False이므로 색상 증강과 좌우 반전 없이 LetterBox만 적용된다.
        self.transform = DetectionTransform(
            image_size=self.image_size,
            training=False,
        )

    @staticmethod
    def _empty_target():
        """DetectionTransform 인터페이스를 재사용하기 위한 객체 0개의 target."""

        # 추론 이미지에는 정답 box/label이 없지만 transform은 두 key를 요구한다.
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
        }

    def prepare_image(self, image_path):
        """이미지 하나를 Tensor와 좌표 복원 메타데이터로 변환한다."""

        image_path = Path(image_path).expanduser().resolve()

        with Image.open(image_path) as opened_image:
            # 휴대폰 사진 등의 EXIF orientation을 픽셀에 실제 반영한다.
            image = ImageOps.exif_transpose(opened_image).convert("RGB")

            # PIL 파일이 닫힌 뒤에도 Results가 사용할 수 있도록 메모리를 복사한다.
            original_rgb = np.asarray(image, dtype=np.uint8).copy()

            # 반환 image_tensor: [3, image_size, image_size], float32, 0~1
            image_tensor, transformed_target = self.transform(
                image,
                self._empty_target(),
            )

        # Ultralytics Results.plot/save가 기대하는 OpenCV BGR 순서로 변환한다.
        original_bgr = np.ascontiguousarray(original_rgb[:, :, ::-1])
        padding_values = transformed_target["padding"].tolist()

        # scale/padding은 NMS 박스를 원본 좌표로 되돌릴 때 사용한다.
        metadata = ImageMetadata(
            path=image_path,
            original_bgr=original_bgr,
            original_shape=tuple(int(value) for value in original_bgr.shape[:2]),
            scale=float(transformed_target["scale"].item()),
            padding=tuple(int(value) for value in padding_values),
        )

        return image_tensor.contiguous(), metadata

    def prepare_batch(self, image_paths):
        """고정 크기 CHW 이미지들을 BCHW float32 Tensor로 쌓는다."""

        tensors = []
        metadata = []

        for image_path in image_paths:
            image_tensor, image_metadata = self.prepare_image(image_path)
            tensors.append(image_tensor)
            metadata.append(image_metadata)

        if not tensors:
            raise ValueError("빈 이미지 배치는 전처리할 수 없습니다.")

        # 모든 입력은 정사각 LetterBox이므로 shape가 같고 batch stack이 가능하다.
        return torch.stack(tensors, dim=0), metadata
