"""COCO2017 데이터로 자체 구현 YOLO11 모델을 학습하는 실행 파일."""

# 학습 설정을 사람이 읽기 쉬운
# JSON 형식으로 로그에 기록한다.
import json

# 모델, Loss, AMP와 DataLoader를 사용한다.
import torch
from torch.utils.data import DataLoader

# 프로젝트 전체 학습 설정
from config import get_config

# COCO Dataset, 심볼릭 링크,
# collate 및 이미지·bbox 변환
from data import (
    Coco2017Dataset,
    build_train_transform,
    build_val_transform,
    detection_collate_fn,
    prepare_coco_dataset_link,
)

# 기존 YOLO11 Detection Loss
from loss import YOLO11DetectionLoss

# 기존 YOLO11 모델
from model import Yolov11

# Optimizer, Scheduler, EMA,
# checkpoint 및 epoch 학습 함수
from training import (
    DetectionMAP,
    ModelEMA,
    build_checkpoint,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    train_epoch,
    validate_epoch,
)

# 로그와 난수 seed 설정
from utils import (
    seed_worker,
    set_seed,
    setup_logger,
)


def select_device():
    """
    CUDA 사용 가능 여부를 확인해
    학습 장치를 선택한다.
    """

    # CUDA를 사용할 수 있으면
    # 첫 번째 CUDA GPU를 선택한다.
    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    # CUDA를 사용할 수 없으면
    # CPU로 학습한다.
    return torch.device(
        "cpu"
    )


def build_grad_scaler(
    amp_enabled,
):
    """
    설치된 PyTorch 버전에 맞는
    CUDA GradScaler를 생성한다.
    """

    if not isinstance(
        amp_enabled,
        bool,
    ):
        raise TypeError(
            "amp_enabled는 bool이어야 합니다."
        )

    try:
        # 새로운 PyTorch에서 권장하는 API다.
        return torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )

    except (
        AttributeError,
        TypeError,
    ):
        # 이전 PyTorch에는 torch.amp.GradScaler가 없거나
        # 첫 번째 device 인자를 지원하지 않을 수 있다.
        return torch.cuda.amp.GradScaler(
            enabled=amp_enabled,
        )


def build_detection_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    seed,
):
    """
    객체 탐지 Dataset에 맞는
    DataLoader를 생성한다.

    Args:
        dataset:
            Coco2017Dataset

        batch_size:
            한 batch의 이미지 수

        shuffle:
            Dataset 순서를 섞을지 결정

        num_workers:
            데이터를 준비할 worker process 수

        pin_memory:
            CUDA 전송에 page-locked memory를 사용할지 결정

        persistent_workers:
            epoch이 끝난 뒤에도 worker를 유지할지 결정

        prefetch_factor:
            worker마다 미리 준비할 batch 수

        seed:
            DataLoader 난수 seed

    Returns:
        data_loader:
            객체 탐지용 DataLoader
    """

    # 빈 Dataset은 학습할 수 없으므로
    # DataLoader를 만들기 전에 중단한다.
    if len(dataset) == 0:
        raise ValueError(
            "DataLoader에 전달된 "
            "Dataset이 비어 있습니다."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size는 1 이상이어야 합니다."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers는 0 이상이어야 합니다."
        )

    # shuffle 순서를 다시 실행해도 비교할 수 있도록
    # DataLoader 전용 PyTorch 난수 생성기를 만든다.
    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    # 모든 DataLoader에서 공통으로 사용할 설정
    loader_options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": (
            detection_collate_fn
        ),
        "drop_last": False,
        "worker_init_fn": (
            seed_worker
        ),
        "generator": generator,
    }

    # persistent_workers와 prefetch_factor는
    # worker process가 존재할 때만 사용할 수 있다.
    #
    # num_workers=0일 때 이 값을 전달하면
    # PyTorch에서 오류가 발생할 수 있다.
    if num_workers > 0:
        loader_options[
            "persistent_workers"
        ] = persistent_workers

        loader_options[
            "prefetch_factor"
        ] = prefetch_factor

    # 딕셔너리에 저장한 설정으로
    # 실제 DataLoader를 생성한다.
    return DataLoader(
        **loader_options
    )


def build_model_config(
    config,
    model,
):
    """
    checkpoint에서 모델 구조를 확인할
    최소 설정을 만든다.
    """

    return {
        # Detection Head 클래스 수
        "num_classes": int(
            config.num_classes
        ),

        # n, s, m, l, x 중 선택된 모델 scale
        "scale": str(
            model.scale
        ),

        # DFL 분포 구간 수
        "reg_max": int(
            config.reg_max
        ),

        # P3, P4, P5 stride
        "strides": tuple(
            config.strides
        ),

        # 모델 입력 이미지 크기
        "image_size": int(
            config.image_size
        ),
    }


def log_epoch_result(
    logger,
    epoch,
    epochs,
    train_loss,
    val_result,
):
    """
    한 epoch의 train 및 validation 결과를
    콘솔과 로그 파일에 기록한다.
    """

    # 학습 Loss 기록
    logger.info(
        "Epoch %d/%d 완료 | "
        "Train Box %.4f | "
        "Cls %.4f | "
        "DFL %.4f | "
        "Total %.4f",
        epoch,
        epochs,
        train_loss[
            "box_loss"
        ],
        train_loss[
            "cls_loss"
        ],
        train_loss[
            "dfl_loss"
        ],
        train_loss[
            "total_loss"
        ],
    )

    # 검증 Loss 기록
    logger.info(
        "Epoch %d/%d 완료 | "
        "Val Box %.4f | "
        "Cls %.4f | "
        "DFL %.4f | "
        "Total %.4f",
        epoch,
        epochs,
        val_result[
            "box_loss"
        ],
        val_result[
            "cls_loss"
        ],
        val_result[
            "dfl_loss"
        ],
        val_result[
            "total_loss"
        ],
    )

    # mAP를 사용한 경우에만
    # detection 평가 지표를 기록한다.
    if "map" in val_result:
        logger.info(
            "Epoch %d/%d | "
            "mAP50-95 %.4f | "
            "mAP50 %.4f | "
            "mAP75 %.4f | "
            "mAR100 %.4f",
            epoch,
            epochs,
            val_result[
                "map"
            ],
            val_result[
                "map_50"
            ],
            val_result[
                "map_75"
            ],
            val_result[
                "mar_100"
            ],
        )


def main():
    """
    데이터 준비부터 checkpoint 저장까지
    전체 학습을 실행한다.
    """

    # --------------------------------------------------
    # 1. 설정과 로그 준비
    # --------------------------------------------------

    # config.py의 설정을 하나의 객체로 가져온다.
    config = get_config()

    # 콘솔과 파일에 동시에 기록하는 logger를 만든다.
    logger, log_path = (
        setup_logger(
            log_dir=(
                config.log_dir
            )
        )
    )

    try:
        # 실행 설정을 로그에 남기면
        # 나중에 실험 조건을 다시 확인할 수 있다.
        logger.info(
            "학습 설정:\n%s",
            json.dumps(
                config.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
        )

        # Python, NumPy, PyTorch 및
        # CUDA에서 사용할 난수 seed를 설정한다.
        set_seed(
            seed=config.seed,
            deterministic=(
                config.deterministic
            ),
        )

        # --------------------------------------------------
        # 2. 학습 장치 선택
        # --------------------------------------------------

        device = select_device()

        logger.info(
            "학습 장치: %s",
            device,
        )

        # CUDA가 선택된 경우에는
        # 실제 GPU 이름도 기록한다.
        if device.type == "cuda":
            logger.info(
                "CUDA 장치 이름: %s",
                torch.cuda.get_device_name(
                    0
                ),
            )

        # 실제 CUDA를 사용할 때만
        # float16 AMP를 활성화한다.
        amp_enabled = bool(
            config.use_amp
            and device.type == "cuda"
        )

        # pinned memory는 CPU에서 CUDA로
        # 데이터를 전송할 때 사용한다.
        pin_memory_enabled = bool(
            config.pin_memory
            and device.type == "cuda"
        )

        logger.info(
            "AMP: %s | "
            "pin_memory: %s | "
            "num_workers: %d",
            amp_enabled,
            pin_memory_enabled,
            config.num_workers,
        )

        # --------------------------------------------------
        # 3. COCO 저장소와 심볼릭 링크 준비
        # --------------------------------------------------

        # 실제 데이터 저장 위치:
        # /home/jblee/datasets/coco
        #
        # 프로젝트에서 접근할 위치:
        # <project>/datasets/coco
        prepare_coco_dataset_link(
            storage_dir=(
                config.dataset_storage_dir
            ),
            link_dir=(
                config.dataset_link_dir
            ),
        )

        logger.info(
            "COCO 실제 저장소: %s",
            config.dataset_storage_dir,
        )

        logger.info(
            "COCO 프로젝트 링크: %s",
            config.dataset_link_dir,
        )

        # --------------------------------------------------
        # 4. 학습 및 검증 Transform 생성
        # --------------------------------------------------

        # 학습 데이터에는 다음 변환을 적용한다.
        #
        # Letterbox
        # Random horizontal flip
        # ColorJitter
        train_transform = (
            build_train_transform(
                config
            )
        )

        # 검증 데이터에는 무작위 증강 없이
        # Letterbox만 적용한다.
        val_transform = (
            build_val_transform(
                config
            )
        )

        # --------------------------------------------------
        # 5. COCO Dataset 생성
        # --------------------------------------------------

        # 데이터가 없다면
        # /home/jblee/datasets/coco 아래에 다운로드한다.
        train_dataset = (
            Coco2017Dataset(
                image_dir=(
                    config.train_image_dir
                ),
                annotation_file=(
                    config.train_annotation_file
                ),
                image_size=(
                    config.image_size
                ),
                auto_download=(
                    config.auto_download
                ),
                allow_insecure_ssl_fallback=(
                    config
                    .allow_insecure_ssl_fallback
                ),
                transform=(
                    train_transform
                ),
            )
        )

        # Validation Dataset을 생성한다.
        #
        # train Dataset 생성 과정에서 annotation이 다운로드됐다면
        # 기존 annotation을 재사용하고 val 이미지만 확인한다.
        val_dataset = (
            Coco2017Dataset(
                image_dir=(
                    config.val_image_dir
                ),
                annotation_file=(
                    config.val_annotation_file
                ),
                image_size=(
                    config.image_size
                ),
                auto_download=(
                    config.auto_download
                ),
                allow_insecure_ssl_fallback=(
                    config
                    .allow_insecure_ssl_fallback
                ),
                transform=(
                    val_transform
                ),
            )
        )

        # Dataset 클래스 수와 모델 클래스 수가 다르면
        # class target이 Head 범위를 벗어날 수 있다.
        if (
            train_dataset.num_classes
            != config.num_classes
        ):
            raise ValueError(
                "Train Dataset 클래스 수와 "
                "모델 설정이 다릅니다: "
                f"{train_dataset.num_classes}, "
                f"{config.num_classes}"
            )

        if (
            val_dataset.num_classes
            != config.num_classes
        ):
            raise ValueError(
                "Validation Dataset 클래스 수와 "
                "모델 설정이 다릅니다: "
                f"{val_dataset.num_classes}, "
                f"{config.num_classes}"
            )

        logger.info(
            "Train 이미지 수: %d | "
            "Validation 이미지 수: %d",
            len(
                train_dataset
            ),
            len(
                val_dataset
            ),
        )

        # --------------------------------------------------
        # 6. DataLoader 생성
        # --------------------------------------------------

        # 학습 Dataset은 매 epoch 순서를 섞는다.
        train_loader = (
            build_detection_loader(
                dataset=(
                    train_dataset
                ),
                batch_size=(
                    config.batch_size
                ),
                shuffle=True,
                num_workers=(
                    config.num_workers
                ),
                pin_memory=(
                    pin_memory_enabled
                ),
                persistent_workers=(
                    config.persistent_workers
                ),
                prefetch_factor=(
                    config.prefetch_factor
                ),
                seed=(
                    config.seed
                ),
            )
        )

        # 검증 Dataset은 결과 비교를 위해
        # 순서를 섞지 않는다.
        val_loader = (
            build_detection_loader(
                dataset=(
                    val_dataset
                ),
                batch_size=(
                    config.batch_size
                ),
                shuffle=False,
                num_workers=(
                    config.num_workers
                ),
                pin_memory=(
                    pin_memory_enabled
                ),
                persistent_workers=(
                    config.persistent_workers
                ),
                prefetch_factor=(
                    config.prefetch_factor
                ),
                seed=(
                    config.seed + 1
                ),
            )
        )

        # --------------------------------------------------
        # 7. YOLO11 모델과 Loss 생성
        # --------------------------------------------------

        # 모델 구현은 수정하지 않고
        # 기존 Yolov11 클래스를 그대로 사용한다.
        model = Yolov11(
            num_classes=(
                config.num_classes
            ),
            scale=(
                config.scale
            ),
            reg_max=(
                config.reg_max
            ),
            strides=(
                config.strides
            ),
        ).to(
            device
        )

        # 기존 YOLO11 Detection Loss를 생성한다.
        criterion = (
            YOLO11DetectionLoss(
                num_classes=(
                    config.num_classes
                ),
                reg_max=(
                    config.reg_max
                ),
                strides=(
                    config.strides
                ),
                box_gain=(
                    config.box_gain
                ),
                cls_gain=(
                    config.cls_gain
                ),
                dfl_gain=(
                    config.dfl_gain
                ),
                tal_topk=(
                    config.tal_topk
                ),
            )
            .to(
                device
            )
        )

        # checkpoint에서
        # 모델 구조를 확인할 설정을 만든다.
        model_config = (
            build_model_config(
                config=config,
                model=model,
            )
        )

        logger.info(
            "모델 설정: %s",
            model_config,
        )

        # --------------------------------------------------
        # 8. Optimizer, Scheduler, AMP, EMA 생성
        # --------------------------------------------------

        optimizer = (
            build_optimizer(
                model=model,
                learning_rate=(
                    config.learning_rate
                ),
                weight_decay=(
                    config.weight_decay
                ),
            )
        )

        # Scheduler는 batch마다 갱신된다.
        #
        # 따라서 한 epoch의 실제 batch 수를 전달한다.
        scheduler = (
            build_scheduler(
                optimizer=optimizer,
                epochs=(
                    config.epochs
                ),
                steps_per_epoch=len(
                    train_loader
                ),
                warmup_epochs=(
                    config.warmup_epochs
                ),
                min_lr_ratio=(
                    config.min_lr_ratio
                ),
            )
        )

        # CUDA AMP에 사용할 GradScaler를 만든다.
        #
        # CPU에서는 disabled 상태의 Scaler가 생성된다.
        scaler = (
            build_grad_scaler(
                amp_enabled=(
                    amp_enabled
                )
            )
        )

        # EMA를 사용하지 않는 경우를 위한 초기값
        ema = None

        if config.use_ema:
            ema = ModelEMA(
                model=model,
                decay=(
                    config.ema_decay
                ),
                tau=(
                    config.ema_tau
                ),
            )

        # mAP를 사용하지 않는 경우를 위한 초기값
        metric = None

        # mAP는 torchmetrics 및 NMS 연산이 필요하므로
        # config.calculate_map=True일 때만 생성한다.
        if config.calculate_map:
            metric = DetectionMAP(
                num_classes=(
                    config.num_classes
                ),
                image_size=(
                    config.image_size
                ),
                confidence_threshold=(
                    config
                    .confidence_threshold
                ),
                nms_iou_threshold=(
                    config
                    .nms_iou_threshold
                ),
                max_detections=(
                    config.max_detections
                ),
            )

        # --------------------------------------------------
        # 9. 학습 재개 상태 초기화
        # --------------------------------------------------

        # 새 학습은 0번째 epoch부터 시작한다.
        start_epoch = 0

        # 아직 Optimizer가 실행되지 않았다.
        global_step = 0

        # 첫 validation 결과는 항상
        # 무한대보다 작으므로 best.pt로 저장된다.
        best_val_loss = float(
            "inf"
        )

        # --------------------------------------------------
        # 10. Checkpoint 복원
        # --------------------------------------------------

        if (
            config.resume_checkpoint
            is not None
        ):
            resume_state = (
                load_checkpoint(
                    checkpoint_path=(
                        config.resume_checkpoint
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    map_location=device,
                )
            )

            # 다른 scale, reg_max, image_size 등의 checkpoint를
            # 실수로 이어서 학습하는 것을 방지한다.
            if (
                resume_state[
                    "model_config"
                ]
                != model_config
            ):
                raise ValueError(
                    "checkpoint 모델 설정과 "
                    "현재 설정이 다릅니다.\n"
                    "checkpoint: "
                    f"{resume_state['model_config']}\n"
                    "현재 설정: "
                    f"{model_config}"
                )

            # 완료된 다음 epoch부터 이어서 학습한다.
            start_epoch = (
                resume_state[
                    "start_epoch"
                ]
            )

            global_step = (
                resume_state[
                    "global_step"
                ]
            )

            best_val_loss = (
                resume_state[
                    "best_val_loss"
                ]
            )

            logger.info(
                "checkpoint에서 학습 재개: %s | "
                "완료 epoch %d | "
                "global step %d",
                config.resume_checkpoint,
                start_epoch,
                global_step,
            )

        # checkpoint가 이미 전체 epoch를 완료했다면
        # 추가 학습 없이 정상적으로 종료한다.
        if (
            start_epoch
            >= config.epochs
        ):
            logger.info(
                "checkpoint가 이미 설정된 "
                "전체 epoch를 완료했습니다: "
                "%d/%d",
                start_epoch,
                config.epochs,
            )

            return

        # --------------------------------------------------
        # 11. 전체 Epoch 학습 반복
        # --------------------------------------------------

        for epoch_index in range(
            start_epoch,
            config.epochs,
        ):
            logger.info(
                "========== "
                "Epoch %d/%d 시작 "
                "==========",
                epoch_index + 1,
                config.epochs,
            )

            # 한 epoch의 forward, backward,
            # Optimizer, Scheduler, EMA 갱신을 수행한다.
            (
                train_loss,
                global_step,
            ) = train_epoch(
                model=model,
                data_loader=(
                    train_loader
                ),
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                scheduler=scheduler,
                ema=ema,
                use_amp=(
                    config.use_amp
                ),
                max_grad_norm=(
                    config.max_grad_norm
                ),
                epoch_index=(
                    epoch_index
                ),
                total_epochs=(
                    config.epochs
                ),
                global_step=(
                    global_step
                ),
                logger=logger,
                log_interval=(
                    config.log_interval
                ),
            )

            # EMA가 활성화됐다면
            # validation은 이동평균 모델로 수행한다.
            validation_model = (
                ema.ema
                if ema is not None
                else model
            )

            # Validation loss와
            # 선택적인 mAP를 계산한다.
            val_result = (
                validate_epoch(
                    model=(
                        validation_model
                    ),
                    data_loader=(
                        val_loader
                    ),
                    criterion=criterion,
                    device=device,
                    use_amp=(
                        config.use_amp
                    ),
                    metric=metric,
                    logger=logger,
                )
            )

            # 한 epoch 결과를 로그에 기록한다.
            log_epoch_result(
                logger=logger,
                epoch=(
                    epoch_index + 1
                ),
                epochs=(
                    config.epochs
                ),
                train_loss=(
                    train_loss
                ),
                val_result=(
                    val_result
                ),
            )

            # 현재 validation total loss가
            # 이전 best보다 작은지 확인한다.
            is_best = (
                val_result[
                    "total_loss"
                ]
                < best_val_loss
            )

            if is_best:
                best_val_loss = (
                    val_result[
                        "total_loss"
                    ]
                )

            # --------------------------------------------------
            # 12. Checkpoint 생성 및 저장
            # --------------------------------------------------

            checkpoint = (
                build_checkpoint(
                    # 지금까지 완료된 epoch 수
                    epoch=(
                        epoch_index + 1
                    ),

                    # 기본 학습 모델
                    model=model,

                    # Optimizer와 Scheduler 상태
                    optimizer=optimizer,
                    scheduler=scheduler,

                    # AMP와 EMA 상태
                    scaler=scaler,
                    ema=ema,

                    # 학습 반복 상태
                    best_val_loss=(
                        best_val_loss
                    ),
                    global_step=(
                        global_step
                    ),

                    # 모델 및 전체 학습 설정
                    model_config=(
                        model_config
                    ),
                    train_config=(
                        config.to_dict()
                    ),

                    # 사람이 확인할 Loss 결과
                    train_loss=(
                        train_loss
                    ),
                    val_loss=(
                        val_result
                    ),
                )
            )

            # 매 epoch이 완료될 때마다
            # last.pt를 갱신한다.
            last_path = (
                save_checkpoint(
                    checkpoint=(
                        checkpoint
                    ),
                    checkpoint_path=(
                        config.checkpoint_dir
                        / "last.pt"
                    ),
                )
            )

            logger.info(
                "최근 checkpoint 저장: %s",
                last_path,
            )

            # Validation total loss가 좋아진 경우에만
            # best.pt를 갱신한다.
            if is_best:
                best_path = (
                    save_checkpoint(
                        checkpoint=(
                            checkpoint
                        ),
                        checkpoint_path=(
                            config.checkpoint_dir
                            / "best.pt"
                        ),
                    )
                )

                logger.info(
                    "최적 checkpoint 저장: %s | "
                    "Val total %.4f",
                    best_path,
                    best_val_loss,
                )

        # --------------------------------------------------
        # 13. 학습 완료
        # --------------------------------------------------

        logger.info(
            "학습 완료 | "
            "best validation total loss: %.4f",
            best_val_loss,
        )

        logger.info(
            "전체 로그 파일: %s",
            log_path,
        )

    except KeyboardInterrupt:
        # Ctrl+C 또는 작업 중단이 발생해도
        # 마지막으로 완료된 epoch의 last.pt는 이미 저장돼 있다.
        logger.warning(
            "사용자에 의해 학습이 중단됐습니다. "
            "마지막으로 완료된 epoch의 "
            "last.pt를 사용할 수 있습니다."
        )

        raise

    except Exception:
        # 다운로드, Dataset, 모델, Loss, CUDA 오류 등의
        # 전체 traceback을 콘솔과 로그 파일에 기록한다.
        logger.exception(
            "학습 중 오류가 발생했습니다."
        )

        raise


# 이 파일을 직접 실행했을 때만 학습을 시작한다.
#
# 다른 Python 파일에서 import할 때는
# 자동으로 학습하지 않는다.
if __name__ == "__main__":
    main()