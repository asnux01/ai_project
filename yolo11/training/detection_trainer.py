# 라이브러리
import torch


class DetectionTrainer:
    
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        scheduler=None,
        ema=None,
        max_grad_norm=None
    ):
        
        # 파라미터
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.ema = ema
        self.max_grad_norm = max_grad_norm
        
        # Loss 모듈 Device 설정
        self.criterion.to(self.device)
        
    def _move_to_device(self, batch):
        
        # Batch 저장
        device_batch = {}
        
        # Tensor를 Device로 이동
        for key, value in batch.items():
            
            if torch.is_tensor(value):
                device_batch[key] = value.to(
                    self.device,
                    non_blocking=True
                )
            else:
                device_batch[key] = value

        return device_batch
            
    def train_epoch(self, train_loader):
        
        # Model 학습 모드
        self.model.train()
        
        # Loss 누적값
        total_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0
        
        # Step 수
        num_steps = 0
        
        # Batch 학습
        for batch in train_loader:
            
            # Batch를 Device로 이동
            batch = self._move_to_device(batch)
            
            # 입력 이미지
            images = batch["img"]
            
            # Gradient 초기화
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward
            predictions = self.model(images)
            
            # Loss 계산
            loss, loss_items = self.criterion(predictions, batch)
            
            # Loss 유효성 검사
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected: {loss.item()}"
                )
                
            # Backward
            loss.backward()
            
            # Gradient Clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            # Model Parameter 업데이트
            self.optimizer.step()
            
            # Learning Rate 업데이트
            if self.scheduler is not None:
                self.scheduler.step()
            
            # EMA 업데이트
            if self.ema is not None:
                self.ema.update(self.model)
                
            # Loss 누적
            total_loss += loss.detach().item()
            total_box_loss += loss_items["box_loss"].item()
            total_cls_loss += loss_items["cls_loss"].item()
            total_dfl_loss += loss_items["dfl_loss"].item()
            
            # Step 수 증가
            num_steps += 1
            
        # Empty Dataloader 검사
        if num_steps == 0:
            raise ValueError("train_loader contains no batches.")
    
        # Epoch 평균 Loss
        metrics = {
            "loss": total_loss / num_steps,
            "box_loss": total_box_loss / num_steps,
            "cls_loss": total_cls_loss / num_steps,
            "dfl_loss": total_dfl_loss / num_steps,
            "lr": self.optimizer.param_groups[0]["lr"]
        }

        return metrics