# 라이브러리
import torch

from ultralytics.utils.tqdm import TQDM

from .metrics import DetectionMetrics


class Validator:
    
    def __init__(
        self,
        criterion,
        postprocessor,
        device,
        num_classes=80,
        max_detections=100,
        coco_evaluator=None
    ):
        
        # 파라미터
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.device = torch.device(device)
        
        # COCO 공식 평가기 저장
        self.coco_evaluator = coco_evaluator
        
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
        
        # COCO Evaluator 초기화
        if self.coco_evaluator is not None:
            self.coco_evaluator.reset()
    
        # Loss 누적값
        total_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0
    
        # Step 수
        num_steps = 0
        
        # 처리한 객체 수
        processed_instances = 0
        
        # Validation Progress Bar Header
        print(
            f"{'class':>18}"
            f"{'Images':>11}"
            f"{'Instances':>11}"
        )
        
        # Progress Bar
        progress_bar = TQDM(
            val_loader,
            total=len(val_loader),
            leave=True
        )

        # Gradient 계산 비활성화
        with torch.no_grad():
            
            # Validation Batch 순회
            for batch in progress_bar:
                
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
                
                # COCO 공식 Metric 계산을 위한 Prediction 누적
                if self.coco_evaluator is not None:
                    self.coco_evaluator.update(
                        predictions=predictions,
                        batch=batch
                    )
                    
                # Logging용 Validation Total Loss 계산
                display_loss = (
                    loss_items["box_loss"].item()
                    + loss_items["cls_loss"].item()
                    + loss_items["dfl_loss"].item()
                )
                    
                # Loss 누적
                total_loss += display_loss
                total_box_loss += loss_items["box_loss"].item()
                total_cls_loss += loss_items["cls_loss"].item()
                total_dfl_loss += loss_items["dfl_loss"].item()
                
                # Step 증가
                num_steps += 1
                
                # 처리한 객체 수 누적
                processed_instances += batch["cls"].numel()
                
                # 현재까지 처리한 Image 수
                processed_images = min(
                    num_steps * val_loader.batch_size,
                    len(val_loader.dataset)
                )
                
                # Validation Progress Bar 갱신
                progress_bar.set_description(
                    f"{'all':>18}"
                    f"{processed_images:>11}"
                    f"{processed_instances:>11}"
                )
                
            # Empty DataLoader 검사
            if num_steps == 0:
                raise ValueError("val_loader에 배치가 없습니다.")

            # Detection Metric 계산
            detection_metrics = self.metrics.compute()
            
            # COCO 공식 Metric 계산
            if self.coco_evaluator is not None:
                coco_metrics = self.coco_evaluator.compute()
            else:
                coco_metrics = None

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
            
            # COCO 공식 Metric을 Validation 결과에 추가
            if coco_metrics is not None:

                val_metrics["coco_map50_95"] = (
                    coco_metrics["map50_95"]
                )

                val_metrics["coco_map50"] = (
                    coco_metrics["map50"]
                )

                val_metrics["coco_map75"] = (
                    coco_metrics["map75"]
                )

                val_metrics["coco_mar"] = (
                    coco_metrics["mar"]
                )
                
        return val_metrics
    