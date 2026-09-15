# 라이브러리
import random

import torch

from config import Config

from data import (
    COCODetectionDataset,
    build_dataloader,
    DetectionTransform
)

from model import Yolov11

from loss import YOLO11DetectionLoss

from training import (
    build_optimizer,
    build_scheduler,
    ModelEMA,
    load_checkpoint,
    DetectionTrainer,
    Trainer
)

from validation import (
    Validator,
    COCOEvaluator
)

# Postprocessor 구현 후 사용
from utils import DetectionPostprocessor


def set_seed(seed):

    # Python Seed 설정
    random.seed(seed)

    # PyTorch Seed 설정
    torch.manual_seed(seed)

    # CUDA Seed 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    # Config
    config = Config()

    # Random Seed 설정
    set_seed(config.seed)

    # Device 설정
    device = torch.device(config.device)

    # Training Transform
    train_transform = DetectionTransform(
        image_size=config.image_size,
        training=True,
        hflip_prob=config.horizontal_flip,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        translate=config.translate,
        scale=config.scale
    )

    # Validation Transform
    val_transform = DetectionTransform(
        image_size=config.image_size,
        training=False
    )

    # Training Dataset
    train_dataset = COCODetectionDataset(
        image_dir=config.train_image_dir,
        annotation_file=config.train_annotation_file,
        transforms=train_transform,
        image_size=config.image_size,
        mosaic_prob=config.mosaic_prob,
        close_mosaic=config.close_mosaic
    )

    # Validation Dataset
    val_dataset = COCODetectionDataset(
        image_dir=config.val_image_dir,
        annotation_file=config.val_annotation_file,
        transforms=val_transform
    )

    # Class Index를 원래 COCO Category ID로 복원하는 Mapping 생성
    class_index_to_category_id = {
        class_index: category_id
        for category_id, class_index
        in val_dataset.category_id_to_class_index.items()
    }

    # COCO 공식 Evaluator 생성
    coco_evaluator = COCOEvaluator(
        coco_gt=val_dataset.coco,
        class_index_to_category_id=class_index_to_category_id,
        image_size=config.image_size,
        max_detections=config.max_detections
    )

    # Training DataLoader
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=False
    )

    # Validation DataLoader
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=True
    )

    # Model
    model = Yolov11(
        num_classes=config.num_classes,
        scale=config.model_scale,
        reg_max=config.reg_max,
        strides=config.strides
    )

    # Model Device 이동
    model = model.to(device)

    # Detection Loss
    criterion = YOLO11DetectionLoss(
        num_classes=config.num_classes,
        reg_max=config.reg_max,
        strides=config.strides,
        box_gain=config.box_gain,
        cls_gain=config.cls_gain,
        dfl_gain=config.dfl_gain,
        tal_topk=config.tal_topk,
        tal_alpha=config.tal_alpha,
        tal_beta=config.tal_beta
    )

    # Detection Loss Device 이동
    criterion = criterion.to(device)

    # Optimizer
    optimizer = build_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.optimizer_eps
    )

    # Scheduler
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=config.warmup_epochs,
        min_lr_ratio=config.min_lr_ratio
    )

    # EMA
    if config.use_ema:

        ema = ModelEMA(
            model=model,
            decay=config.ema_decay,
            tau=config.ema_tau
        )

    else:

        ema = None

    # Postprocessor
    postprocessor = DetectionPostprocessor(
        num_classes=config.num_classes,
        reg_max=config.reg_max,
        strides=config.strides,
        confidence_threshold=config.confidence_threshold,
        nms_iou_threshold=config.nms_iou_threshold,
        max_detections=config.max_detections
    )

    # Detection Trainer
    detection_trainer = DetectionTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        ema=ema,
        max_grad_norm=config.max_grad_norm,
        batch_size=config.batch_size,
        nominal_batch_size=config.nominal_batch_size
    )

    # Validator
    validator = Validator(
        criterion=criterion,
        postprocessor=postprocessor,
        device=device,
        num_classes=config.num_classes,
        max_detections=config.max_detections,
        coco_evaluator=coco_evaluator
    )

    # Trainer
    trainer = Trainer(
        detection_trainer=detection_trainer,
        validator=validator,
        epochs=config.epochs,
        checkpoint_dir=config.checkpoint_dir,
        monitor=config.monitor,
        mode=config.monitor_mode
    )

    # 시작 Epoch
    start_epoch = 0

    # Best Metric
    best_metric = None

    # Checkpoint Resume
    if config.resume_path is not None:

        start_epoch, best_metric = load_checkpoint(
            checkpoint_path=config.resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device
        )

    # Training
    best_metric = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch,
        best_metric=best_metric
    )

    # Training 완료
    print(
        "Training completed."
    )

    print(
        f"Best {config.monitor}: "
        f"{best_metric:.4f}"
    )


if __name__ == "__main__":
    main()