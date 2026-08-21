"""객체 탐지 이미지와 bounding box를 함께 변환한다."""
# 라이브러리
import torch
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


class DetectionTransform:
    """
    Letterbox와 기본 데이터 증강을
    이미지와 bbox에 함께 적용한다.
    """

    def __init__(
        self,
        image_size,
        training=False,
        horizontal_flip_probability=0.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        translate_fraction=0.0,
        scale_gain=0.0,
        fill_value=114,
    ):
        """
        Args:
            image_size:
                최종 모델 입력 이미지 크기

            training:
                True이면 무작위 학습 증강을 적용

            horizontal_flip_probability:
                좌우 반전을 적용할 확률

            brightness:
                밝기 변화 범위

            contrast:
                대비 변화 범위

            saturation:
                채도 변화 범위

            hue:
                색조 변화 범위

            fill_value:
                Letterbox의 빈 영역을 채울 픽셀값
        """

        # 모델 입력 크기는 Tensor shape을 결정하므로
        # 양의 정수여야 한다.
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
        ):
            raise TypeError(
                "image_size는 정수여야 합니다."
            )

        if image_size <= 0:
            raise ValueError(
                "image_size는 0보다 커야 합니다."
            )

        # training=False인 검증 데이터에는
        # 무작위 증강을 적용하지 않는다.
        if not isinstance(training, bool):
            raise TypeError(
                "training은 bool이어야 합니다."
            )

        if not (0.0 <= horizontal_flip_probability <= 1.0):
            raise ValueError(
                "horizontal_flip_probability는 "
                "0과 1 사이여야 합니다."
            )
        
        if not (
            0.0
            <= translate_fraction
            <= 1.0
        ):
            raise ValueError(
                "translate_fraction은 "
                "0과 1 사이여야 합니다."
            )

        if not (
            0.0
           <= scale_gain
            < 1.0
        ):
           raise ValueError(
                "scale_gain은 "
                "0 이상 1 미만이어야 합니다."
            )

        # RGB 픽셀값은 0~255 범위여야 한다.
        if not (0 <= fill_value <= 255):
            raise ValueError(
                "fill_value는 0과 255 사이여야 합니다."
            )

        # 검사된 설정을 실제 변환에서 사용할 수 있도록 저장한다.
        self.image_size = image_size
        self.training = training

        self.horizontal_flip_probability = (
            horizontal_flip_probability
        )
        
        # Random Translation에 사용할
        # 최대 이동 비율을 저장한다.
        self.translate_fraction = float(
            translate_fraction
        )

        # Random Scale에 사용할
        # 확대/축소 변화 범위를 저장한다.
        self.scale_gain = float(
            scale_gain
        )

        # float 등이 들어오더라도 실제 padding에는
        # 정수 픽셀값을 사용한다.
        self.fill_value = int(
            fill_value
        )
        
        # ColorJitter는 이미지 색만 변경하므로
        # bbox 좌표는 변경할 필요가 없다.
        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

        # 모든 색상 변화량이 0이면
        # ColorJitter 호출 자체를 생략한다.
        self.use_color_jitter = any(
            value > 0
            for value in (
                brightness,
                contrast,
                saturation,
                abs(hue)
            )
        )

    def _horizontal_flip(
        self,
        image,
        boxes,
        image_width,
    ):
        """
        이미지와 xyxy 형식 bbox를
        같은 방향으로 좌우 반전한다.
        """

        # 먼저 PIL 이미지를 좌우 반전한다.
        image = F.hflip(image)

        # 객체가 있을 때만 bbox 좌표를 계산한다.
        if boxes.numel() > 0:

            # 같은 Tensor에 값을 덮어쓰기 전에
            # 기존 x좌표를 복사한다.
            old_x1 = (
                boxes[:, 0].clone()
            )

            old_x2 = (
                boxes[:, 2].clone()
            )

            # 좌우 반전된 xyxy 좌표:
            #
            # new_x1 = image_width - old_x2
            # new_x2 = image_width - old_x1
            boxes[:, 0] = float(image_width) - old_x2
            boxes[:, 2] = float(image_width) - old_x1

        # 반전된 이미지와 bbox를 함께 반환한다.
        return image, boxes
    
    def _random_affine(
        self,
        image,
        boxes,
        image_width,
        image_height,
    ):
        """
        이미지와 xyxy bounding box에
        동일한 Random Scale과
        Random Translation을 적용한다.

        이미지에 기하학적 변환을 적용하면
        정답 bbox에도 정확히 같은 변환을
        적용해야 한다.
        """

        # --------------------------------------------------
        # 1. Random Scale 결정
        # --------------------------------------------------

        if self.scale_gain > 0.0:

            # 예:
            #
            # scale_gain = 0.5
            #
            # 최소 배율 = 0.5
            # 최대 배율 = 1.5
            scale_min = (
                1.0
                - self.scale_gain
            )

            scale_max = (
                1.0
                + self.scale_gain
            )

            random_scale = (
                scale_min
                + torch.rand(
                    1
                ).item()
                * (
                    scale_max
                    - scale_min
                )
            )

        else:
            # Scale 증강을 사용하지 않으면
            # 원본 크기를 유지한다.
            random_scale = 1.0


        # --------------------------------------------------
        # 2. Random Translation 결정
        # --------------------------------------------------

        if (
            self.translate_fraction
            > 0.0
        ):

            # 가로 방향 최대 이동 픽셀 수
            max_translate_x = (
                float(
                    image_width
                )
                * self.translate_fraction
            )

            # 세로 방향 최대 이동 픽셀 수
            max_translate_y = (
                float(
                    image_height
                )
                * self.translate_fraction
            )

            # -max ~ +max 사이에서
            # 무작위 이동량을 결정한다.
            translate_x = (
                (
                    torch.rand(
                        1
                    ).item()
                    * 2.0
                    - 1.0
                )
                * max_translate_x
            )

            translate_y = (
                (
                    torch.rand(
                        1
                    ).item()
                    * 2.0
                    - 1.0
                )
                * max_translate_y
            )

        else:

            translate_x = 0.0
            translate_y = 0.0


        # torchvision affine의 translate는
        # 정수 픽셀 단위로 전달한다.
        translate_x = int(
            round(
                translate_x
            )
        )

        translate_y = int(
            round(
                translate_y
            )
        )


        # --------------------------------------------------
        # 3. 이미지에 Affine 적용
        # --------------------------------------------------

        image = F.affine(
            image,

            # 이번 구현에서는 회전하지 않는다.
            angle=0.0,

            # x, y 방향 이동량
            translate=[
                translate_x,
                translate_y,
            ],

            # 확대/축소 비율
            scale=random_scale,

            # 기울이기는 사용하지 않는다.
            shear=[
                0.0,
                0.0,
            ],

            interpolation=(
                InterpolationMode.BILINEAR
            ),

            # 변환으로 새로 생긴 영역은
            # YOLO에서 사용하는 회색값 114로 채운다.
            fill=(
                self.fill_value,
                self.fill_value,
                self.fill_value,
            ),
        )


        # --------------------------------------------------
        # 4. bbox에 동일한 Scale 적용
        # --------------------------------------------------

        if boxes.numel() > 0:

            # torchvision affine의 Scale은
            # 이미지 중심을 기준으로 수행된다.
            center_x = (
                float(
                    image_width
                )
                / 2.0
            )

            center_y = (
                float(
                    image_height
                )
                / 2.0
            )

            # x1, x2를 이미지 중심 기준으로
            # 확대/축소한다.
            boxes[:, [0, 2]] = (
                (
                    boxes[:, [0, 2]]
                    - center_x
                )
                * random_scale
                + center_x
            )

            # y1, y2도 동일하게 변환한다.
            boxes[:, [1, 3]] = (
                (
                    boxes[:, [1, 3]]
                    - center_y
                )
                * random_scale
                + center_y
            )


            # --------------------------------------------------
            # 5. bbox Translation 적용
            # --------------------------------------------------

            boxes[:, [0, 2]] += (
                float(
                    translate_x
                )
            )

            boxes[:, [1, 3]] += (
                float(
                    translate_y
                )
            )


            # --------------------------------------------------
            # 6. 이미지 범위로 bbox 제한
            # --------------------------------------------------
            boxes[:, [0, 2]] = (
                boxes[:, [0, 2]].clamp_(
                    min=0.0,
                    max=float(
                        image_width
                    ),
                )
            )

            boxes[:, [1, 3]] = (
                boxes[:, [1, 3]].clamp_(
                    min=0.0,
                    max=float(
                        image_height
                    ),
                )
            )


        return image, boxes

    def _letterbox(
        self,
        image,
        boxes,
        original_width,
        original_height,
    ):
        """
        원본 비율을 유지하면서
        정사각형 모델 입력 크기로 맞춘다.
        """

        # 가로와 세로 중 더 많이 축소해야 하는 비율을 선택하면
        # 원본 비율을 유지한 채 전체 이미지가 입력 영역에 들어간다.
        scale = min(
            self.image_size / float(original_width),
            self.image_size / float(original_height)
        )

        # 반올림 결과가 0이 되지 않도록
        # 최소 크기를 1로 제한한다.
        resized_width = max(
            1, 
            int(round(original_width * scale))
        )

        resized_height = max(
            1,
            int(round(original_height * scale))
        )

        # 계산한 비율로 이미지를 resize한다.
        image = F.resize(
            image,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )

        # resize 이후 입력 크기까지 남아 있는
        # 전체 padding 크기를 계산한다.
        horizontal_padding = self.image_size - resized_width
        vertical_padding = self.image_size - resized_height

        # padding을 양쪽에 최대한 균등하게 나눈다.
        #
        # 전체 padding이 홀수이면
        # 오른쪽 또는 아래쪽에 한 픽셀 더 들어간다.
        left = horizontal_padding // 2
        right = horizontal_padding - left
        top = vertical_padding // 2
        bottom = vertical_padding - top

        # YOLO 계열에서 일반적으로 사용하는
        # 회색 픽셀값 114로 빈 영역을 채운다.
        image = F.pad(
            image,
            [left, top, right, bottom],
            fill=(
                self.fill_value,
                self.fill_value,
                self.fill_value
            )
        )

        # 이미지에 객체가 있을 때 bbox에도
        # 동일한 scale과 padding을 적용한다.
        if boxes.numel() > 0:

            # x1과 x2에 resize 비율 및
            # 왼쪽 padding을 적용한다.
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + left

            # y1과 y2에 resize 비율 및
            # 위쪽 padding을 적용한다.
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + top

            # 계산 오차로 bbox가 모델 입력 범위를
            # 벗어나지 않도록 좌표를 제한한다.
            boxes[:, [0, 2]] = (
                boxes[:, [0, 2]].clamp_(
                    min=0.0,
                    max=float(self.image_size)
                )
            )
            
            boxes[:, [1, 3]] = (
                boxes[:, [1, 3]].clamp_(
                    min=0.0,
                    max=float(self.image_size)
                )
            )

        # 나중에 시각화나 원본 좌표 복원에 사용할 수 있도록
        # 네 방향 padding 값을 저장한다.
        padding = (left, top, right, bottom)

        return (image, boxes, scale, padding)

    def __call__(
        self,
        image,
        target,
    ):
        """PIL 이미지와 target을 함께 변환한다."""

        # target은 boxes, labels 등을 담는 딕셔너리여야 한다.
        if not isinstance(target, dict):
            raise TypeError(
                "target은 dict여야 합니다."
            )

        # 탐지 학습에는 bbox와 클래스 label이 모두 필요하다.
        if (
            "boxes" not in target
            or "labels" not in target
        ):
            raise KeyError(
                "target에는 boxes와 labels가 필요합니다."
            )

        # PIL image.size의 순서는
        # Tensor shape과 달리 (width, height)다.
        (original_width, original_height) = image.size

        if (
            original_width <= 0
            or original_height <= 0
        ):
            raise ValueError(
                "이미지의 너비와 높이는 "
                "0보다 커야 합니다."
            )

        # 원본 target을 직접 변경하지 않도록
        # 새로운 딕셔너리를 만든다.
        transformed_target = dict(target)

        # bbox 계산은 float32로 수행한다.
        boxes = target["boxes"].clone().to(dtype=torch.float32)

        # 클래스 번호는 정수형으로 유지한다.
        labels = target["labels"].clone().to(dtype=torch.int64)

        # xyxy bbox는 객체마다 좌표 4개가 필요하다.
        if (
            boxes.ndim != 2
            or boxes.shape[-1] != 4
        ):
            raise ValueError(
                "boxes shape은 [N, 4]여야 합니다."
            )

        # bbox 하나마다 label도 정확히 하나씩 있어야 한다.
        if (
            labels.ndim != 1
            or labels.shape[0]
            != boxes.shape[0]
        ):
            raise ValueError(
                "labels shape은 [N]이고 "
                "boxes 개수와 같아야 합니다."
            )

        # 색상 증강은 학습 데이터에만 적용한다.
        if (self.training and self.use_color_jitter):
            image = self.color_jitter(image)
            
        # 학습 중에만 적용한다.
        #
        # 이미지 크기와 위치를 변화시켜
        # 객체의 크기와 위치 변화에 대한
        # 모델의 일반화 성능을 높인다.
        if (
            self.training
            and (
                self.translate_fraction > 0.0
                or self.scale_gain > 0.0
            )
        ):
            image, boxes = (
                self._random_affine(
                    image=image,
                    boxes=boxes,
                    image_width=(
                       original_width
                    ),
                    image_height=(
                        original_height
                    ),
                )
            )

        # target metadata에 실제 반전 여부를
        # 기록하기 위한 초기값이다.
        was_flipped = False

        # 좌우 반전 여부도 PyTorch seed의 영향을 받으므로
        # DataLoader worker별 재현 가능한 결과를 만들 수 있다.
        if (
            self.training
            and torch.rand(1).item()
            < self.horizontal_flip_probability
        ):
            image, boxes = (
                self._horizontal_flip(
                    image=image,
                    boxes=boxes,
                    image_width=(
                        original_width
                    ),
                )
            )

            was_flipped = True

        # 학습과 검증 모두 마지막에는
        # 동일한 입력 크기의 Letterbox를 적용한다.
        (
            image,
            boxes,
            scale,
            padding
        ) = self._letterbox(
            image=image,
            boxes=boxes,
            original_width=(
                original_width
            ),
            original_height=(
                original_height
            ),
        )

        # 변환 후 너비 또는 높이가 사라진 bbox와
        # 해당 bbox의 label을 함께 제거한다.
        if boxes.numel() > 0:

            valid_mask = (
                (boxes[:, 2] > boxes[:, 0])
                & 
                (boxes[:, 3] > boxes[:, 1])
            )
            boxes = boxes[valid_mask]
            labels = labels[valid_mask]

        # PIL 이미지를 다음 형식으로 변환한다.
        #
        # shape:
        # [3, H, W]
        #
        # dtype:
        # float32
        #
        # 값 범위:
        # 0~1
        image = F.pil_to_tensor(image).to(dtype=torch.float32) / 255.0
        

        # DataLoader에서 이미지를 쌓기 전에
        # 예상한 크기로 변환됐는지 확인한다.
        if image.shape != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise RuntimeError(
                "변환된 이미지 크기가 "
                "예상과 다릅니다: "
                f"{tuple(image.shape)}"
            )

        # 변환이 완료된 bbox와 label을 target에 저장한다.
        transformed_target["boxes"] = boxes

        transformed_target["labels"] = labels

        # 검증 및 시각화에서 좌표 변환을 확인할 수 있도록
        # 원본 이미지 크기를 [height, width] 순서로 저장한다.
        transformed_target["original_size"] = torch.tensor(
            [original_height, original_width], dtype=torch.int64
        )

        # 모델에 들어가는 최종 입력 크기를 저장한다.
        transformed_target["input_size"] = torch.tensor(
            [self.image_size, self.image_size], dtype=torch.int64
        )

        # Letterbox resize에 사용한 비율을 저장한다.
        transformed_target["scale"] = torch.tensor(
            scale, dtype=torch.float32
        )

        # padding 순서는
        # [left, top, right, bottom]이다.
        transformed_target["padding"] = torch.tensor(
            padding, dtype=torch.int64
        )

        # 학습 증강에서 좌우 반전됐는지 저장한다.
        transformed_target["was_flipped"] = torch.tensor(
            was_flipped, dtype=torch.bool
        )

        return (image, transformed_target)


def build_train_transform(
    config
):
    """
    학습 데이터용 Letterbox와
    기본 증강을 생성한다.
    """

    return DetectionTransform(
        image_size=config.image_size,
        training=True,
        horizontal_flip_probability=config.horizontal_flip_probability,
        brightness=config.brightness,
        contrast=config.contrast,
        saturation=config.saturation,
        hue=config.hue,
        translate_fraction=config.translate_fraction,
        scale_gain=config.scale_gain,
    )


def build_val_transform(
    config,
):
    """
    무작위 증강이 없는
    검증 데이터용 Letterbox를 생성한다.
    """

    return DetectionTransform(
        image_size=config.image_size,
        training=False
    )