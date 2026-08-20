"""자체 YOLO11 체크포인트로 이미지 추론을 실행하는 CLI."""

# 명령행 옵션과 정상적인 종료 코드를 처리한다.
import argparse
import sys
from pathlib import Path

# inference 패키지의 공개 API만 가져온다.
# 세부 구현은 inference/ 내부에 숨겨 CLI 파일이 복잡해지지 않게 한다.
from inference import (
    InferenceConfig,
    YOLO11Predictor,
    create_save_dir,
    save_results,
)


def build_argument_parser():
    """Ultralytics predict 명령과 비슷한 추론 옵션을 정의한다."""

    parser = argparse.ArgumentParser(
        description="자체 구현 YOLO11 이미지 추론",
    )

    # --------------------------------------------------
    # 모델과 입력 소스
    # --------------------------------------------------
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("checkpoints/best.pt"),
        help="best.pt 또는 last.pt 경로",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="이미지 파일, 디렉터리 또는 glob (여러 개 가능)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="입력 크기. 기본값은 체크포인트의 image_size",
    )

    # --------------------------------------------------
    # Confidence filtering 및 NMS 설정
    # --------------------------------------------------
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="이미지당 최대 탐지 수")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0 등")
    parser.add_argument("--batch", type=int, default=1, help="이미지 추론 batch size")
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=None,
        help="추론할 클래스 번호만 지정",
    )
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="클래스와 무관하게 NMS 적용",
    )

    # --------------------------------------------------
    # 실행 장치와 가중치 선택
    # --------------------------------------------------
    parser.add_argument(
        "--amp",
        action="store_true",
        help="CUDA FP16 autocast 사용 (현재 Blackwell 환경에서는 기본 비활성화)",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="EMA 대신 checkpoint의 model_state_dict 사용",
    )

    # --------------------------------------------------
    # 결과 저장 설정
    # --------------------------------------------------
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="박스가 그려진 결과 이미지 저장 안 함",
    )
    parser.add_argument("--save-txt", action="store_true", help="YOLO 형식 txt 저장")
    parser.add_argument("--save-conf", action="store_true", help="txt에 confidence 포함")
    parser.add_argument("--project", type=Path, default=Path("runs/predict"))
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true", help="기존 출력 폴더 사용 허용")
    return parser


def main(arguments=None):
    """설정을 만들고 모든 source를 스트리밍 방식으로 추론한다."""

    # 문자열로 받은 CLI 인자를 Python 값으로 변환한다.
    args = build_argument_parser().parse_args(arguments)

    # InferenceConfig가 범위, 경로, batch size 등을 실행 전에 검사한다.
    config = InferenceConfig(
        weights=args.weights,
        image_size=args.imgsz,
        confidence_threshold=args.conf,
        nms_iou_threshold=args.iou,
        max_detections=args.max_det,
        device=args.device,
        batch_size=args.batch,
        use_amp=args.amp,
        prefer_ema=not args.no_ema,
        classes=(tuple(args.classes) if args.classes is not None else None),
        agnostic_nms=args.agnostic_nms,
    )

    # 체크포인트 로딩과 모델 복원은 Predictor 생성 시 한 번만 수행한다.
    predictor = YOLO11Predictor(config)

    # --no-save가 없으면 시각화 이미지를 기본으로 저장한다.
    # TXT만 저장하는 경우에도 출력 폴더가 필요하다.
    save_images = not args.no_save
    needs_save_dir = save_images or args.save_txt
    save_dir = None

    if needs_save_dir:
        # 기본값은 runs/predict/exp이며 이미 있으면 exp2, exp3 순으로 증가한다.
        save_dir = create_save_dir(
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            save_txt=args.save_txt,
        )

    print(
        f"모델: {config.weights} | epoch: {predictor.loaded_model.completed_epoch} | "
        f"weights: {predictor.loaded_model.weight_source}"
    )
    print(
        f"장치: {predictor.device} | 입력 크기: {predictor.image_size} | "
        f"AMP: {predictor.amp_enabled}"
    )

    seen = 0

    # stream=True는 이미지가 많아도 모든 Results를 메모리에 쌓지 않는다.
    for result in predictor(args.source, stream=True):
        seen += 1

        # Ultralytics Results가 제공하는 탐지 개수와 클래스별 요약을 출력한다.
        box_count = 0 if result.boxes is None else len(result.boxes)
        description = result.verbose().strip()
        print(f"{result.path}: {box_count} detections | {description}")

        if save_dir is not None:
            # result.save()/save_txt()를 통해 이미지 및 YOLO 형식 라벨을 기록한다.
            save_results(
                [result],
                save_dir=save_dir,
                save_images=save_images,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
            )

    print(f"추론 완료: {seen} images")

    if save_dir is not None:
        print(f"결과 저장: {save_dir}")

    return 0


if __name__ == "__main__":
    # import 시에는 실행하지 않고 predict.py를 직접 실행한 경우에만 시작한다.
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Ctrl+C 종료를 셸에서 식별 가능한 표준 종료 코드 130으로 반환한다.
        print("사용자에 의해 추론이 중단됐습니다.", file=sys.stderr)
        raise SystemExit(130)
