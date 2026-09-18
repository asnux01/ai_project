# 라이브러리
import torch

from ultralytics.utils.tqdm import TQDM


class DetectionTrainer:
    
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        scheduler=None,
        ema=None,
        max_grad_norm=None,
        batch_size=16,
        nominal_batch_size=64
    ):
        
        # 파라미터
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.ema = ema
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.nominal_batch_size = nominal_batch_size
        self.accumulate = max(
            round(nominal_batch_size / batch_size),
            1
        )
        
        # 마지막 Optimizer Step
        self.last_optimizer_steps = -1
        
        # Gradient 초기화
        self.optimizer.zero_grad(set_to_none=True)
        
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
    
            
    def _get_gpu_memory(self):
        
        # CUDA를 사용하지 않는 경우
        if self.device.type != "cuda":
            return "0G"
        
        # 현재 Device의 예약된 GPU Memory
        memory = torch.cuda.memory_reserved(self.device) / 1e9
        
        return f"{memory:.2f}G"
    
               
    def train_epoch(
        self, 
        train_loader,
        epoch,
        epochs
    ):
        
        # Dataset에 현재 Epoch 정보 전달
        if hasattr(
            train_loader.dataset,
            "set_epoch"
        ):
            train_loader.dataset.set_epoch(
                epoch=epoch,
                total_epochs=epochs
            )
        
        # Model 학습 모드
        self.model.train()
        
        # Loss 누적값
        total_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0
        
        # Step 수
        num_steps = 0
        
        # Ultralytics 스타일 Header 출력
        print(
            "\n"
            f"{'Epoch':>11}"
            f"{'GPU_mem':>11}"
            f"{'box_loss':>11}"
            f"{'cls_loss':>11}"
            f"{'dfl_loss':>11}"
            f"{'Instances':>11}"
            f"{'Size':>11}"
        )

        # Ultralytics TQDM Progress Bar
        progress_bar = TQDM(
            train_loader,
            total=len(train_loader),
            leave=True
        )

        # Batch 학습
        for batch_index, batch in enumerate(progress_bar):
            
            # Global Step
            global_step = (
                epoch * len(train_loader)
                + batch_index
            )
            
            # Warmup
            if self.scheduler is not None:
                
                self.scheduler.prepare_batch(
                    global_step=global_step,
                    epoch=epoch
                )
                
                self.accumulate = (
                    self.scheduler.get_accumulate(
                        global_step
                    )
                )
            
            # Batch를 Device로 이동
            batch = self._move_to_device(batch)
            
            # 입력 이미지
            images = batch["img"]
            
            # Forward
            predictions = self.model(images)
            
            # Loss 계산
            loss, loss_items = self.criterion(predictions, batch)
            
            # Positive Sample 및 Target Score Debug 정보 출력
            if num_steps % 100 == 0:
                debug = self.criterion.last_debug
                
                print(
                    f"\nFG: {debug.get('foreground_count')} | "
                    f"TargetScoreSum: "
                    f"{debug.get('target_scores_sum'):.2f}"
                )
            
            # Loss 유효성 검사
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected: {loss.item()}"
                )
                
            # Backward
            loss.backward()
            
            # 현재 Batch에서 Optimizer Step을 수행할지 결정
            should_step = (
                global_step
                - self.last_optimizer_steps
                >= self.accumulate
            )
            
            # Parameter 업데이트
            if should_step:

                # Gradient Clipping
                if self.max_grad_norm is not None:
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
            
                # Optimizer Step
                self.optimizer.step()
                
                # 마지막 Step 저장
                self.last_optimizer_steps = (global_step)
            
                # Gradient 초기화
                self.optimizer.zero_grad(
                    set_to_none=True
                )
            
                # EMA 업데이트
                if self.ema is not None:
                    self.ema.update(self.model)
                
            # Logging용 Total Loss 계산
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
            
            # Step 수 증가
            num_steps += 1
            
            # 현재까지의 평균 Loss
            avg_box_loss = total_box_loss / num_steps
            avg_cls_loss = total_cls_loss / num_steps
            avg_dfl_loss = total_dfl_loss / num_steps
            
            # 현재 Batch의 객체 수
            num_instances = batch["cls"].numel()
            
            # 입력 Image 크기
            image_size = images.shape[-1]
            
            # GPU Memory 사용량
            gpu_memory = self._get_gpu_memory()
            
            # Progress Bar 설명 갱신
            progress_bar.set_description(
                f"{epoch + 1:>5}/{epochs:<5}"
                f"{gpu_memory:>11}"
                f"{avg_box_loss:>11.3f}"
                f"{avg_cls_loss:>11.3f}"
                f"{avg_dfl_loss:>11.3f}"
                f"{num_instances:>11}"
                f"{image_size:>11}"
            )
            
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