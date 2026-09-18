# 라이브러리
from pathlib import Path

import torch


class Config:
    
    # Project 경로
    project_root = Path("/home/jblee")
    
    # Dataset 경로
    dataset_root = project_root / "dataset" / "coco"
    train_image_dir = dataset_root / "images" / "train2017"
    val_image_dir = dataset_root / "images" / "val2017"
    train_annotation_file = dataset_root / "annotations" / "instances_train2017.json"
    val_annotation_file = dataset_root / "annotations" / "instances_val2017.json"
    
    # Model
    num_classes = 80
    model_scale = "n"
    image_size = 640
    reg_max = 16
    strides = (8, 16, 32)
    
    # Dataloader
    batch_size = 16
    nominal_batch_size = 64
    num_workers = 4
    pin_memory = True
    
    # Data Augmentation
    horizontal_flip = 0.5
    
    # HSV 색상 증강
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    
    # Geometric 증강
    translate = 0.1
    scale = 0.5
    
    # Mosaic 증강
    mosaic_prob = 1.0
    close_mosaic = 10
    
    # Loss
    box_gain = 7.5
    cls_gain = 0.5
    dfl_gain = 1.5
    tal_topk = 10
    tal_alpha = 0.5
    tal_beta = 6.0
    
    # Optimizer
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 0.0005
    
    # Training
    epochs = 100
    max_grad_norm = 10.0
    
    # Scheduler
    warmup_epochs = 3.0
    warmup_momentum = 0.8
    warmup_bias_lr = 0.0
    min_lr_ratio = 0.01
    
    # EMA
    use_ema = True
    ema_decay = 0.9999
    ema_tau = 2000
    
    # Validation
    nms_max_detections = 300
    metric_max_detections = 100
    confidence_threshold = 0.001
    nms_iou_threshold = 0.7
    
    # Checkpoint
    checkpoint_dir = project_root / "ai_project" / "yolo11" / "checkpoints"
    monitor = "coco_map50_95"
    monitor_mode = "max"
    resume_path = None
    
    # Device
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    
    # Random Seed
    seed = 42