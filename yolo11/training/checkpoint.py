# 라이브러리
from pathlib import Path

import torch

def save_checkpoint(
    save_path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_metric,
    ema=None
):
    
    # 저장 경로
    save_path = Path(save_path)
    
    # 저장 폴더 생성
    save_path.parent.mkdir(
        parents=True, 
        exist_ok=True
    )
    
    # Checkpoint 구성
    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }
    
    # EMA 저장
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()
        checkpoint["ema_updates"] = ema.updates
        
    # Checkpoint 저장
    torch.save(checkpoint, save_path)
    
def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    ema=None,
    device="cpu",
    strict=True
):
    
    # Checkpoint 경로
    checkpoint_path = Path(checkpoint_path)
    
    # Checkpoint 존재 확인
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )
        
    # Checkpoint 불러오기
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )
    
    # Model 복원
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict
    )
    
    # Optimizer 복원
    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
    
    # Scheduler 복원
    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
    
    # EMA 복원
    if ema is not None and "ema_state_dict" in checkpoint:
        ema.ema.load_state_dict(
            checkpoint["ema_state_dict"],
            strict=strict
        )
        ema.updates = checkpoint.get("ema_updates", 0)
        
    # 다음 Epoch
    start_epoch = int(checkpoint["epoch"]) + 1
    
    # Best Metric
    best_metric = float(checkpoint.get("best_metric", float("inf")))
    
    return start_epoch, best_metric