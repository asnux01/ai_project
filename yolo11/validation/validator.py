# 라이브러리
import torch

from .metrics import DetectionMetrics


class Validator:
    
    def __init__(
        self,
        criterion,
        postprocessor,
        device,
        num_classes=80,
        max_detections=100
    ):
        
        # 파라미터
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.device = torch.device(device)
        
        # Detection Metric
        self.metrics = DetectionMetrics(
            num_classes=num_classes,
            max_detections=max_detections
        )
        
        

    def _move_batch_to_device(self, batch):
    
        # Batch 저장
        device_batch = {}
    
        # Tensor를 Device로 이동
        for key, value in batch.items():
        
            if torch.is_tensor(value):
                device_batch[key] = value.to(
                    self.device, non_blocking=True
                )
            else:
                device_batch[key] = value
        
        return device_batch


    def _build_targets(self, batch, batch_size):
    
        # Ground Truth
        boxes = batch["bboxes"]
        labels = batch["cls"].reshape(-1).long()
        batch_indices = batch["batch_idx"].reshape(-1).long()
    
        # Image 단위 Target 저장
        targets = []
    
        # Batch Image 순회
        for image_index in range(batch_size):
        
            # 현재 Image의 GT 선택
            mask = batch_indices == image_index
        
            # Target 구성
            target = {
                "boxes": boxes[mask],
                "labels": labels[mask]
            }
            targets.append(target)
        
        return targets


    def validate(self, model, val_loader):
    
        # Evaluation Mode
        model.eval()
    
        # Metric 초기화
        self.metrics.reset()
    
        # Loss 누적값
        total_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0
    
        # Step 수
        num_steps = 0

        # Gradient 계산 비활성화
        with torch.no_grad():
            
            # Validation Batch 순회
            for batch in val_loader:
                
                # Batch Device 이동
                batch = self._move_batch_to_device(batch)

                # 입력 이미지
                images = batch["img"]

                # Forward
                raw_predictions = model(images)

                # Validation Loss 계산
                loss, loss_items = self.criterion(raw_predictions, batch)

                # Loss 유효성 검사
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss detected: "
                        f"{loss.item()}"
                    )

                # Prediction Postprocess
                predictions = self.postprocessor(raw_predictions)

                # Ground Truth 구성
                targets = self._build_targets(batch=batch, batch_size=images.shape[0])
                
                # Detection Metric 누적
                self.metrics.update(predictions=predictions, targets=targets)
                                
                # Loss 누적
                total_loss += loss.item()
                total_box_loss += loss_items["box_loss"].item()
                total_cls_loss += loss_items["cls_loss"].item()
                total_dfl_loss += loss_items["dfl_loss"].item()
                
                # Step 증가
                num_steps += 1
                
            # Empty DataLoader 검사
            if num_steps == 0:
                raise ValueError("val_loader에 배치가 없습니다.")

            # Detection Metric 계산
            detection_metrics = self.metrics.compute()
            
            # Validation 결과
            val_metrics = {
                "loss": total_loss / num_steps,
                "box_loss": total_box_loss / num_steps,
                "cls_loss": total_cls_loss / num_steps,
                "dfl_loss": total_dfl_loss / num_steps,
                "map50_95": detection_metrics["map50_95"],
                "map50": detection_metrics["map50"],
                "map75": detection_metrics["map75"],
                "mar100": detection_metrics["mar100"]
            }
            
        return val_metrics
    