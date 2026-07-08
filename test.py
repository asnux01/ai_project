# test.py
# ------------------------------------------------------------
# 목적:
#   QAT로 학습한 YOLO11 모델을 이용해 동영상 추론을 수행하는 코드
#
# 이 코드가 하는 일:
#   1. QAT 학습 결과인 qat_state_dict.pt를 불러온다.
#   2. 학습 때와 같은 QAT 구조를 YOLO11 모델에 다시 적용한다.
#   3. 동영상 파일을 입력으로 받아 YOLO 객체 탐지를 수행한다.
#   4. 탐지된 객체에 박스를 그린다.
#   5. 오른쪽 상단에 FPS를 표시한다.
#   6. 결과 동영상을 mp4 파일로 저장한다.
#
# 중요한 점:
#   여기서 FPS는 원본 동영상 FPS가 아니다.
#   이 FPS는 모델이 초당 몇 프레임을 처리하는지 나타내는 추론 처리 FPS다.
#
# 최적화 포인트:
#   기존 방식:
#       프레임 하나 읽기 → yolo.predict() 호출 → 반복
#
#   이 코드:
#       yolo.predict(source=동영상경로, stream=True)
#
#   즉, 동영상 전체를 Ultralytics에 넘겨서 stream 방식으로 처리한다.
#   그래서 매 프레임마다 predict()를 새로 호출하는 오버헤드를 줄일 수 있다.
#
# 주의:
#   이 코드는 진짜 INT8 엔진 추론 코드가 아니다.
#   현재 qat_state_dict.pt는 fake quant QAT 모델이다.
#   실제 INT8 배포 속도는 TensorRT, ONNX Runtime, Vitis AI 등으로
#   변환한 뒤 따로 측정해야 한다.
# ------------------------------------------------------------

import argparse
import time
import types
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from ultralytics import YOLO


# ------------------------------------------------------------
# 1. QAT qconfig 선택 함수
# ------------------------------------------------------------

def choose_qat_qconfig(backend: str):
    """
    QAT에 사용할 quantization 설정을 선택하는 함수.

    backend 의미:
        x86     : 일반 PC/서버 CPU 계열에서 사용 가능
        fbgemm  : x86 서버/PC에서 많이 사용
        qnnpack : ARM 계열에서 많이 사용

    여기서는 기존 학습 코드와 맞추기 위해 기본값을 x86으로 사용한다.
    """

    # PyTorch quantization backend 설정
    try:
        torch.backends.quantized.engine = backend
    except Exception:
        pass

    # 사용자가 지정한 backend로 qconfig 생성 시도
    try:
        qconfig = tq.get_default_qat_qconfig(backend)
        print(f"[QAT] backend={backend} qconfig 사용")
        return qconfig

    # 실패하면 fallback backend를 사용
    except Exception as e:
        print(f"[QAT] backend={backend} qconfig 생성 실패: {e}")

        fallback = "fbgemm" if backend in ["x86", "fbgemm"] else "qnnpack"

        try:
            torch.backends.quantized.engine = fallback
        except Exception:
            pass

        qconfig = tq.get_default_qat_qconfig(fallback)
        print(f"[QAT] fallback backend={fallback} qconfig 사용")
        return qconfig


# ------------------------------------------------------------
# 2. YOLO Conv + BN fuse 함수
# ------------------------------------------------------------

def fuse_yolo_conv_bn_for_qat(model: nn.Module):
    """
    YOLO 모델 내부의 Conv + BatchNorm 구조를 QAT용으로 fuse한다.

    일반적으로 YOLO의 Conv 블록은 내부에:
        Conv2d
        BatchNorm2d
        Activation

    형태를 가진다.

    QAT에서는 Conv와 BatchNorm을 합쳐서 학습/추론 구조를 맞추는 경우가 많다.
    """

    fused_count = 0
    failed_count = 0

    # 모델 안의 모든 모듈을 순회
    for name, module in model.named_modules():

        # Ultralytics Conv 블록은 보통 module.conv, module.bn을 가진다.
        has_conv_bn = (
            hasattr(module, "conv")
            and hasattr(module, "bn")
            and isinstance(module.conv, nn.Conv2d)
            and isinstance(module.bn, nn.BatchNorm2d)
        )

        if not has_conv_bn:
            continue

        # Conv + BN fuse 시도
        try:
            tq.fuse_modules_qat(module, ["conv", "bn"], inplace=True)
            fused_count += 1

        except Exception:
            failed_count += 1

    print(f"[QAT] Conv+BN fuse 성공: {fused_count}")
    print(f"[QAT] Conv+BN fuse 실패 또는 skip: {failed_count}")


# ------------------------------------------------------------
# 3. QAT 제외 layer 설정 함수
# ------------------------------------------------------------

def disable_qat_for_unsupported_or_sensitive_layers(model: nn.Module):
    """
    QAT를 적용하면 문제가 생기거나 민감할 수 있는 layer를 QAT 대상에서 제외한다.

    여기서는:
        1. DFL layer
        2. Upsample layer

    를 QAT 대상에서 제외한다.
    """

    disabled = []

    for name, module in model.named_modules():
        lname = name.lower()
        should_disable = False

        # YOLO Detect head 안의 DFL 부분은 민감할 수 있어서 제외
        if "dfl" in lname:
            should_disable = True

        # Upsample은 quantization 대상에서 제외
        if isinstance(module, nn.Upsample):
            should_disable = True

        if should_disable:
            module.qconfig = None
            disabled.append(name)

    if disabled:
        print("[QAT] QAT 제외 모듈:")
        for name in disabled:
            print(f"  - {name}")


# ------------------------------------------------------------
# 4. QAT YOLO 모델 로드 함수
# ------------------------------------------------------------

def load_qat_yolo(args):
    """
    QAT 학습으로 저장된 qat_state_dict.pt를 불러와서
    추론 가능한 YOLO 객체를 만드는 함수.

    중요한 이유:
        qat_state_dict.pt는 일반적인 Ultralytics best.pt가 아니다.
        따라서 YOLO(qat_state_dict.pt)처럼 바로 로드하면 안 된다.

    처리 순서:
        1. base FP32 모델 yolo11n.pt 로드
        2. 학습 때와 같은 QAT 구조 재구성
        3. qat_state_dict.pt의 state_dict 로드
        4. observer 비활성화
        5. fake quant 활성화
        6. GPU로 이동
    """

    weights_path = Path(args.weights).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"QAT weight 파일을 찾을 수 없습니다: {weights_path}")

    print(f"[LOAD] QAT checkpoint 로드: {weights_path}")

    # QAT checkpoint 로드
    checkpoint = torch.load(weights_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint 안에 'model_state_dict'가 없습니다.")

    # checkpoint 안에 저장된 시작 FP32 모델 이름을 사용
    fp32_weight = args.fp32

    if fp32_weight is None:
        fp32_weight = checkpoint.get("fp32_start_weight", "yolo11n.pt")

    print(f"[LOAD] base FP32 모델 로드: {fp32_weight}")

    # Ultralytics YOLO 모델 생성
    yolo = YOLO(fp32_weight)
    model = yolo.model

    # QAT prepare 전에는 train 모드가 필요
    model.train()

    print("[QAT] inference용 QAT 구조 재구성 시작")

    # 학습 때와 동일하게 Conv + BN fuse
    fuse_yolo_conv_bn_for_qat(model)

    # qconfig 설정
    qconfig = choose_qat_qconfig(args.qat_backend)
    model.qconfig = qconfig

    # 일부 layer는 QAT 제외
    disable_qat_for_unsupported_or_sensitive_layers(model)

    # QAT observer/fake quant module 삽입
    tq.prepare_qat(model, inplace=True)

    # QAT 학습으로 저장된 가중치 로드
    state_dict = checkpoint["model_state_dict"]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # strict=False를 사용했기 때문에 key 차이가 있으면 경고만 출력
    if missing_keys:
        print(f"[WARN] missing keys 개수: {len(missing_keys)}")
        print("[WARN] 일부 missing keys 예시:")
        for key in missing_keys[:10]:
            print(f"  - {key}")

    if unexpected_keys:
        print(f"[WARN] unexpected keys 개수: {len(unexpected_keys)}")
        print("[WARN] 일부 unexpected keys 예시:")
        for key in unexpected_keys[:10]:
            print(f"  - {key}")

    # 추론 시에는 observer 업데이트가 필요 없다.
    # observer는 학습 중 min/max 통계를 모으는 역할을 한다.
    model.apply(tq.disable_observer)

    # fake quant는 켜둔다.
    # 즉 INT8 상황을 흉내 내는 fake quant 연산은 유지한다.
    model.apply(tq.enable_fake_quant)

    # 추론 모드로 변경
    model.eval()

    # device 설정
    device = args.device

    if device != "cpu":
        device = f"cuda:{device}" if not str(device).startswith("cuda") else device

    model.to(device)

    # ------------------------------------------------------------
    # 중요:
    #
    # 위에서 QAT 구조를 만들면서 이미 Conv + BN fuse를 수행했다.
    # 이때 BatchNorm은 Identity로 바뀐다.
    #
    # 그런데 Ultralytics의 yolo.predict()는 내부적으로 model.fuse()를
    # 다시 호출하려고 한다.
    #
    # 이미 BN이 Identity가 된 상태에서 다시 fuse하면:
    #
    # AttributeError: 'Identity' object has no attribute 'weight'
    #
    # 오류가 발생한다.
    #
    # 그래서 predict 단계에서 추가 fuse를 하지 않도록
    # model.fuse 함수를 아무것도 하지 않는 함수로 바꾼다.
    # ------------------------------------------------------------

    def _skip_fuse(self, verbose=True):
        print("[QAT] predict 단계의 추가 fuse를 건너뜁니다.")
        return self

    model.fuse = types.MethodType(_skip_fuse, model)

    print(f"[LOAD] QAT model 준비 완료. device={device}")

    return yolo


# ------------------------------------------------------------
# 5. FPS 표시 함수
# ------------------------------------------------------------

def draw_fps_top_right(frame, fps_value):
    """
    프레임 오른쪽 상단에 FPS 텍스트를 그리는 함수.

    입력:
        frame     : OpenCV 이미지 프레임
        fps_value : 표시할 FPS 값

    출력:
        FPS가 그려진 frame
    """

    text = f"FPS: {fps_value:.1f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # 텍스트 크기 계산
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    h, w = frame.shape[:2]

    # 오른쪽 위 위치 계산
    margin = 12
    x1 = w - text_w - margin * 2
    y1 = margin
    x2 = w - margin
    y2 = margin + text_h + baseline + margin

    # FPS 글자가 잘 보이도록 검은 배경 박스 그림
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    text_x = x1 + margin
    text_y = y1 + text_h + 2

    # 흰색 FPS 텍스트 출력
    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return frame


# ------------------------------------------------------------
# 6. 최적화된 동영상 추론 함수
# ------------------------------------------------------------

def run_video_inference(args):
    """
    동영상 추론 전체를 수행하는 함수.

    기존 방식:
        OpenCV로 프레임 하나 읽기
        yolo.predict(source=frame) 호출
        반복

    현재 방식:
        yolo.predict(source=동영상경로, stream=True) 사용

    장점:
        매 프레임마다 predict를 새로 호출하지 않아서
        Python/Ultralytics 호출 오버헤드가 줄어들 수 있다.
    """

    # 입력 크기가 고정되어 있을 때 cuDNN이 더 빠른 알고리즘을 찾게 한다.
    torch.backends.cudnn.benchmark = True

    # 행렬 연산 precision 설정
    # 지원되지 않는 환경에서는 그냥 무시된다.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    source = args.source
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # QAT 모델 로드
    yolo = load_qat_yolo(args)

    # ------------------------------------------------------------
    # 결과 영상을 저장하려면 원본 영상의 크기와 FPS를 알아야 한다.
    # 그래서 OpenCV로 메타데이터만 읽는다.
    # 실제 추론용 프레임 읽기는 Ultralytics stream=True가 담당한다.
    # ------------------------------------------------------------

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"동영상을 열 수 없습니다: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if src_fps is None or src_fps <= 0:
        src_fps = 30.0

    # vid_stride가 2이면 2프레임마다 1장만 처리하므로 출력 FPS도 반으로 설정
    out_fps = src_fps / max(1, args.vid_stride)

    print(f"[VIDEO] source: {source}")
    print(f"[VIDEO] size: {width}x{height}")
    print(f"[VIDEO] input fps: {src_fps:.2f}")
    print(f"[VIDEO] output fps: {out_fps:.2f}")
    print(f"[VIDEO] total frames: {total_frames}")
    print(f"[VIDEO] vid_stride: {args.vid_stride}")
    print(f"[VIDEO] output: {output_path}")

    # 결과 동영상 writer 생성
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"결과 동영상 writer를 열 수 없습니다: {output_path}")

    print("[INFER] stream=True 최적화 추론 시작")

    frame_idx = 0
    measured_frames = 0
    total_time = 0.0
    fps_smooth = None

    # inference_mode는 gradient 계산을 완전히 끄므로 추론에 적합하다.
    with torch.inference_mode():

        # ------------------------------------------------------------
        # 핵심 최적화 부분
        #
        # source에 frame이 아니라 동영상 경로를 넣는다.
        # stream=True를 주면 결과가 generator 형태로 하나씩 나온다.
        # ------------------------------------------------------------

        results_iter = yolo.predict(
            source=source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            stream=True,
            vid_stride=args.vid_stride,
        )

        iterator = iter(results_iter)

        while True:
            # 한 프레임 처리 시간 측정 시작
            start = time.perf_counter()

            try:
                result = next(iterator)
            except StopIteration:
                break

            # YOLO 결과에 박스, 클래스, confidence를 그린 이미지 생성
            annotated = result.plot()

            # 한 프레임 처리 시간 측정 종료
            elapsed = time.perf_counter() - start

            frame_idx += 1

            # 초기 몇 프레임은 CUDA warmup 때문에 느릴 수 있으므로
            # 평균 FPS 계산에서 제외한다.
            if frame_idx > args.warmup_frames:
                total_time += elapsed
                measured_frames += 1

            # 순간 FPS 계산
            instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0

            # FPS가 너무 튀지 않도록 지수 평균으로 부드럽게 표시
            if fps_smooth is None:
                fps_smooth = instant_fps
            else:
                fps_smooth = (
                    args.fps_smooth * fps_smooth
                    + (1.0 - args.fps_smooth) * instant_fps
                )

            # 오른쪽 상단에 FPS 표시
            annotated = draw_fps_top_right(annotated, fps_smooth)

            # 결과 영상 writer 크기와 프레임 크기가 다르면 맞춰준다.
            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))

            # 결과 프레임 저장
            writer.write(annotated)

            # 일정 프레임마다 진행 상황 출력
            if frame_idx % args.print_every == 0:
                print(f"[INFER] output frame {frame_idx}, FPS={fps_smooth:.2f}")

    writer.release()

    # 평균 pipeline FPS 계산
    # 이 값은 순수 모델 forward FPS가 아니라,
    # 전처리 + 추론 + NMS + 박스 그리기 + 저장이 포함된 처리 FPS다.
    avg_fps = measured_frames / total_time if total_time > 0 else 0.0

    print("[DONE] 동영상 추론 완료")
    print(f"[DONE] output frames: {frame_idx}")
    print(f"[DONE] measured frames: {measured_frames}")
    print(f"[DONE] average pipeline FPS: {avg_fps:.2f}")
    print(f"[DONE] saved video: {output_path}")


# ------------------------------------------------------------
# 7. 명령행 옵션 parser
# ------------------------------------------------------------

def parse_args():
    """
    터미널에서 입력하는 옵션들을 받아오는 함수.

    예:
        python -u test.py --source videos/test1.mp4 --output outputs/result.mp4
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/runs_qat_e20/yolo11_qat_coco_e20/weights/qat_state_dict.pt",
        help="QAT 학습으로 저장된 qat_state_dict.pt 경로",
    )

    parser.add_argument(
        "--fp32",
        type=str,
        default=None,
        help="base FP32 모델. 지정하지 않으면 checkpoint의 fp32_start_weight를 사용.",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="입력 동영상 경로. 예: videos/test1.mp4",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/qat_video_result.mp4",
        help="박스와 FPS가 그려진 결과 동영상 저장 경로",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO 입력 이미지 크기. 작게 하면 빨라지지만 정확도가 떨어질 수 있음.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="confidence threshold. 이 값보다 낮은 박스는 제거.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS에서 사용하는 IoU threshold.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="사용할 장치. 0은 첫 번째 GPU, cpu는 CPU.",
    )

    parser.add_argument(
        "--qat-backend",
        type=str,
        default="x86",
        choices=["x86", "fbgemm", "qnnpack"],
        help="QAT backend. PC/서버는 x86 또는 fbgemm, ARM은 qnnpack 권장.",
    )

    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="1이면 모든 프레임 처리, 2이면 2프레임마다 1장 처리.",
    )

    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="평균 FPS 계산에서 제외할 초기 프레임 수.",
    )

    parser.add_argument(
        "--fps-smooth",
        type=float,
        default=0.90,
        help="FPS 표시 smoothing 값. 0에 가까우면 순간 FPS, 1에 가까우면 부드러운 FPS.",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="몇 프레임마다 진행 상황을 출력할지 설정.",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# 8. 실행 시작점
# ------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_video_inference(args)