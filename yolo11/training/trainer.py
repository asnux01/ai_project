# 라이브러리
from pathlib import Path

from .checkpoint import save_checkpoint


class Trainer:
    
    def __init__(
        self,
        detection_trainer,
        validator,
        epochs,
        checkpoint_dir,
        monitor="map50_95",
        mode="max"
    ):
        
        # 입력 검사
        if epochs <= 0:
            raise ValueError("epochs는 0보다 커야 합니다.")
        if mode not in ("max", "min"):
            raise ValueError("mode는 'max' 또는 'min'이어야 합니다.")
        
        # 파라미터
        self.detection_trainer = detection_trainer
        self.validator = validator
        self.epochs = epochs
        self.monitor = monitor
        self.mode = mode
        
        # 학습 객체
        self.model = detection_trainer.model
        self.optimizer = detection_trainer.optimizer
        self.scheduler = detection_trainer.scheduler
        self.ema = detection_trainer.ema
        
        # Checkpoint 저장 경로
        self.checkpoint_dir = Path(checkpoint_dir)
        self.last_path = self.checkpoint_dir / "last.pt"
        self.best_path = self.checkpoint_dir / "best.pt"
        
    def _is_better(
        self,
        current_metric,
        best_metric
    ):
        
        # 높은 값이 좋은 Metric
        if self.mode == "max":
            return current_metric > best_metric
        
        # 낮은 값이 좋은 Metric
        return current_metric < best_metric
    
    def _initial_best_metric(self):
        
        # 높은 값이 좋은 Metric
        if self.mode == "max":
            return float("-inf")
        
        # 낮은 값이 좋은 Metric
        return float("inf")
    
    def fit(
        self,
        train_loader,
        val_loader,
        start_epoch=0,
        best_metric=None
    ):
        
        # Best Metric 초기화
        if best_metric is None:
            best_metric = self._initial_best_metric()
            
        # Epoch 반복
        for epoch in range(start_epoch, self.epochs):
            
            # 학습
            train_metrics = self.detection_trainer.train_epoch(train_loader)
            
            # Validation Model 선택
            if self.ema is not None:
                validation_model = self.ema.ema
            else:
                validation_model = self.model
                
            # 검증
            val_metrics = self.validator.validate(
                model=validation_model,
                val_loader=val_loader
            )
            
            # 기준 Metric 확인
            if self.monitor not in val_metrics:
                raise KeyError(
                    f"Validation metrics에 '{self.monitor}'가 "
                    "validator에서 반환되지 않았습니다."
                )
                
            current_metric = float(val_metrics[self.monitor])
            
            # Best 여부 확인
            is_best = self._is_better(
                current_metric=current_metric,
                best_metric=best_metric
            )
            
            # Best Metric 갱신
            if is_best:
                best_metric = current_metric
                
            # Last Checkpoint 저장
            save_checkpoint(
                save_path=self.last_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_metric=best_metric,
                ema=self.ema
            )
            
            # Best Checkpoint 저장
            if is_best:
                save_checkpoint(
                    save_path=self.best_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_metric=best_metric,
                    ema=self.ema
                )
            
            # Epoch 결과 출력
            self._print_epoch_results(
                epoch=epoch,
                train_metric=train_metrics,
                val_metric=val_metrics,
                best_metric=best_metric
            )
            
        return best_metric
    
    def _print_epoch_results(
        self,
        epoch,
        train_metric,
        val_metric,
        best_metric
    ):
        
        # Epoch
        epoch_num = epoch + 1
        print(f"Epoch [{epoch_num}/{self.epochs}]")
        
        # Train 결과
        print(
            "Train | "
            f"Loss: {train_metric['loss']:.4f} | "
            f"Box: {train_metric['box_loss']:.4f} | "
            f"Cls: {train_metric['cls_loss']:.4f} | "
            f"DFL: {train_metric['dfl_loss']:.4f} | "
            f"LR: {train_metric['lr']:.6f}"
        )
        
        # Validation 결과
        print(
            "Val   | "
            f"Loss: {val_metric['loss']:.4f} | "
            f"Box: {val_metric['box_loss']:.4f} | "
            f"Cls: {val_metric['cls_loss']:.4f} | "
            f"DFL: {val_metric['dfl_loss']:.4f} | "
        )
        
        # Detection Metrics
        if "map50_95" in val_metric:
            print(
                "Metric | "
                f"mAP50-95: {val_metric['map50_95']:.4f} | "
                f"mAP50: {val_metric['map50']:.4f} | "
                f"mAP75: {val_metric['map75']:.4f} | "
                f"mAR100: {val_metric['mar100']:.4f}"
            )
        
        # Best Metric
        print(f"Best {self.monitor}: {best_metric:.4f}")