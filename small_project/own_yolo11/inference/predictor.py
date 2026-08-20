"""Ultralytics DetectionPredictor 흐름을 자체 YOLO11 모델에 연결한다."""

# 단계별 소요 시간을 Ultralytics Results.speed에 기록한다.
import time

# 추론 모드, autocast, Tensor 처리를 담당한다.
import torch

# 공식 Ultralytics 결과 객체와 좌표 변환 함수를 그대로 사용한다.
from ultralytics.engine.results import Results
from ultralytics.utils import ops

try:
    # 최신 Ultralytics는 NMS를 별도 모듈에 둔다.
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    # 기존 설치 버전과도 호환한다.
    from ultralytics.utils.ops import non_max_suppression

from .checkpoint import load_inference_model
from .config import InferenceConfig
from .names import build_class_names
from .preprocess import InferencePreprocessor
from .sources import iter_batches, resolve_image_sources


class YOLO11Predictor:
    """preprocess → inference → NMS → Results 순서로 이미지 추론을 수행한다."""

    def __init__(self, config, names=None):
        # --------------------------------------------------
        # 1. 설정과 checkpoint 준비
        # --------------------------------------------------
        if not isinstance(config, InferenceConfig):
            raise TypeError("config는 InferenceConfig여야 합니다.")

        self.config = config

        # best.pt/last.pt에서 모델 구조와 가중치를 한 번만 복원한다.
        self.loaded_model = load_inference_model(
            checkpoint_path=config.weights,
            device=config.device,
            prefer_ema=config.prefer_ema,
        )
        self.model = self.loaded_model.model
        self.device = self.loaded_model.device
        self.num_classes = self.loaded_model.num_classes

        # CLI --imgsz가 없으면 학습 checkpoint의 입력 크기를 그대로 사용한다.
        self.image_size = config.image_size or self.loaded_model.image_size

        # P3/P4/P5 feature map 크기가 정확히 나뉘도록 최대 stride 배수만 허용한다.
        if self.image_size % max(self.loaded_model.strides) != 0:
            raise ValueError("image_size는 모델의 최대 stride 배수여야 합니다.")

        # 존재하지 않는 클래스 ID를 NMS에 넘기기 전에 명확한 오류를 낸다.
        if config.classes is not None and any(
            class_id >= self.num_classes for class_id in config.classes
        ):
            raise ValueError(
                f"classes는 0~{self.num_classes - 1} 범위여야 합니다: {config.classes}"
            )

        # --------------------------------------------------
        # 2. Results 메타데이터와 전처리기 준비
        # --------------------------------------------------
        self.names = self._normalize_names(names)

        # Results 및 외부 코드가 model.names를 참조할 수 있도록 공식 모델과 비슷하게 노출한다.
        self.model.names = self.names

        # 학습/검증의 DetectionTransform을 재사용하는 전처리기다.
        self.preprocessor = InferencePreprocessor(self.image_size)
        self._warmed_up = False

    def _normalize_names(self, names):
        """list/dict/None 클래스 이름을 {class_id: name} 형식으로 통일한다."""

        # COCO 80클래스 checkpoint면 공식 COCO 이름을 자동으로 사용한다.
        if names is None:
            return build_class_names(self.num_classes)

        # Results는 정수 class ID로 이름을 조회하므로 dict 형태가 가장 명확하다.
        if isinstance(names, (list, tuple)):
            names = {index: str(name) for index, name in enumerate(names)}
        elif isinstance(names, dict):
            names = {int(index): str(name) for index, name in names.items()}
        else:
            raise TypeError("names는 dict, list, tuple 또는 None이어야 합니다.")

        # 이름이 빠지면 시각화와 verbose 출력에서 잘못된 class가 표시될 수 있다.
        if set(names) != set(range(self.num_classes)):
            raise ValueError(f"names에는 0~{self.num_classes - 1}의 이름이 모두 필요합니다.")

        return names

    @property
    def amp_enabled(self):
        """현재 코드에서는 CUDA일 때만 선택적으로 FP16 autocast를 사용한다."""

        return self.config.use_amp and self.device.type == "cuda"

    def _synchronize(self):
        """비동기 CUDA 연산을 완료해 오류 위치와 시간 측정을 정확하게 한다."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def preprocess(self, image_paths):
        """이미지를 학습과 동일한 LetterBox BCHW Tensor로 만든다."""

        # CPU에서 이미지 파일을 읽고 [B, 3, H, W] float32 Tensor로 쌓는다.
        images, metadata = self.preprocessor.prepare_batch(image_paths)

        # 모델이 있는 장치로 한 번에 이동한다.
        images = images.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=self.device.type == "cuda",
        )
        return images, metadata

    @torch.inference_mode()
    def inference(self, images):
        """eval 모드 forward를 수행하고 [B, 4+C, N] decoded output을 반환한다."""

        # 기본은 FP32다. --amp가 지정되고 CUDA일 때만 Conv/MatMul을 FP16으로 autocast한다.
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            model_output = self.model(images)

        # 자체 Head의 eval 출력 형식:
        #   model_output[0] = decoded [B, 4 + C, N]
        #   model_output[1] = loss/debug용 raw output dict
        if not isinstance(model_output, (tuple, list)) or len(model_output) < 1:
            raise TypeError("eval 모드 모델 출력은 (decoded_output, raw_outputs)여야 합니다.")

        decoded_output = model_output[0]

        if not isinstance(decoded_output, torch.Tensor) or decoded_output.ndim != 3:
            raise ValueError("decoded_output shape은 [B, 4 + C, N]이어야 합니다.")

        # 앞 4채널은 xywh이고 나머지는 sigmoid가 적용된 클래스 확률이다.
        expected_channels = 4 + self.num_classes
        if decoded_output.shape[1] != expected_channels:
            raise ValueError(
                f"decoded_output 채널 수가 {decoded_output.shape[1]}입니다. "
                f"예상값은 {expected_channels}입니다."
            )

        # NMS와 좌표 복원은 수치 안정성을 위해 FP32로 수행한다.
        return decoded_output.float()

    def warmup(self):
        """공식 predictor처럼 첫 실입력 전에 CUDA 모델을 한 번 준비한다."""

        # CPU에서는 warmup 비용이 이득보다 클 수 있어 CUDA에서만 수행한다.
        if self._warmed_up or self.device.type != "cuda":
            return

        # 실제 입력과 같은 shape의 빈 Tensor로 CUDA/cuDNN 커널 선택과 초기화를 끝낸다.
        dummy = torch.zeros(
            (1, 3, self.image_size, self.image_size),
            device=self.device,
            dtype=torch.float32,
        )
        self.inference(dummy)
        self._synchronize()
        self._warmed_up = True

    def _apply_nms(self, decoded_output):
        """설치된 Ultralytics 버전에 맞춰 공식 NMS를 호출한다."""

        # decoded_output shape: [B, 4 + C, N]
        # 반환되는 이미지별 Tensor shape: [D, 6] = xyxy, conf, class_id
        common_arguments = {
            "conf_thres": self.config.confidence_threshold,
            "iou_thres": self.config.nms_iou_threshold,
            "classes": (
                list(self.config.classes)
                if self.config.classes is not None
                else None
            ),
            "agnostic": self.config.agnostic_nms,
            "max_det": self.config.max_detections,
        }

        try:
            # 최신 API에서는 nc를 명시해 클래스 뒤의 채널을 mask로 오해하지 않게 한다.
            return non_max_suppression(
                decoded_output,
                nc=self.num_classes,
                **common_arguments,
            )
        except TypeError as error:
            # nc 인자가 없던 구버전에서만 한 번 호환 호출한다.
            if "nc" not in str(error):
                raise
            return non_max_suppression(decoded_output, **common_arguments)

    def postprocess(self, decoded_output, metadata):
        """NMS 후 박스를 원본 이미지 좌표로 복원하고 Results를 만든다."""

        # --------------------------------------------------
        # 1. Confidence filtering 및 class-aware NMS
        # --------------------------------------------------
        detections = self._apply_nms(decoded_output)

        if len(detections) != len(metadata):
            raise RuntimeError("NMS 결과 수와 입력 이미지 수가 다릅니다.")

        results = []

        for detection, image_metadata in zip(detections, metadata):
            # NMS 출력 Tensor를 직접 바꾸지 않도록 이미지별 복사본을 만든다.
            detection = detection.clone()

            if detection.numel() > 0:
                # --------------------------------------------------
                # 2. LetterBox 입력 좌표 → 원본 이미지 좌표
                # --------------------------------------------------
                # scale과 left/top padding을 제거하고 원본 너비·높이 안으로 clip한다.
                detection[:, :4] = ops.scale_boxes(
                    (self.image_size, self.image_size),
                    detection[:, :4],
                    image_metadata.original_shape,
                    ratio_pad=image_metadata.ratio_pad,
                    padding=True,
                )

            # --------------------------------------------------
            # 3. 공식 Ultralytics Results 생성
            # --------------------------------------------------
            # boxes의 각 행은 [x1, y1, x2, y2, confidence, class_id]다.
            result = Results(
                image_metadata.original_bgr,
                path=str(image_metadata.path),
                names=self.names,
                boxes=detection[:, :6],
            )
            results.append(result)

        return results

    def stream(self, source):
        """많은 이미지도 메모리에 누적하지 않도록 Results를 순차적으로 yield한다."""

        # 파일/디렉터리/glob을 정렬된 실제 이미지 경로로 확장한다.
        image_paths = resolve_image_sources(source)

        # CUDA일 경우 첫 실제 입력의 초기화 지연을 별도 warmup으로 분리한다.
        self.warmup()

        for image_batch in iter_batches(image_paths, self.config.batch_size):
            # --------------------------------------------------
            # 1. 전처리 시간
            # --------------------------------------------------
            preprocess_start = time.perf_counter()
            images, metadata = self.preprocess(image_batch)
            self._synchronize()
            preprocess_end = time.perf_counter()

            # --------------------------------------------------
            # 2. 모델 forward 시간
            # --------------------------------------------------
            decoded_output = self.inference(images)
            self._synchronize()
            inference_end = time.perf_counter()

            # --------------------------------------------------
            # 3. NMS 및 좌표 복원 시간
            # --------------------------------------------------
            results = self.postprocess(decoded_output, metadata)
            self._synchronize()
            postprocess_end = time.perf_counter()

            # 공식 Results.speed와 동일하게 이미지 한 장당 millisecond를 기록한다.
            batch_count = len(results)
            speed = {
                "preprocess": (preprocess_end - preprocess_start) * 1000.0 / batch_count,
                "inference": (inference_end - preprocess_end) * 1000.0 / batch_count,
                "postprocess": (postprocess_end - inference_end) * 1000.0 / batch_count,
            }

            for result in results:
                # generator는 한 장씩 반환하므로 호출자가 즉시 저장하거나 분석할 수 있다.
                result.speed = dict(speed)
                yield result

    def __call__(self, source, stream=False):
        """Ultralytics처럼 stream=False면 list, True면 generator를 반환한다."""

        generator = self.stream(source)
        return generator if stream else list(generator)
