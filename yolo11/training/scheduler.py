# 라이브러리
import math


class WarmupLinearScheduler:
    
    def __init__(
        self,
        optimizer,
        epochs,
        steps_per_epoch,
        batch_size,
        nominal_batch_size=64,
        warmup_epochs=3.0,
        min_lr_ratio=0.01,
        momentum=0.9,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0
    ):
        
        # 입력 검사
        if epochs <= 0:
            raise ValueError("epochs는 0보다 커야 합니다.")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch는 0보다 커야 합니다.")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs는 0보다 크거나 같아야 합니다.")
        if not 0.0 < min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio는 0보다 크고 1보다 작거나 같아야 합니다.")
        
        # 파라미터
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.step_per_epoch = int(steps_per_epoch)
        
        self.batch_size = int(batch_size)
        self.nominal_batch_size = int(nominal_batch_size)
        
        self.min_lr_ratio = min_lr_ratio
        self.momentum = momentum
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        
        # Warmup Step
        if warmup_epochs > 0:
            self.warmup_steps = max(
                round( warmup_epochs * self.step_per_epoch),
                100
            )
        else:
            self.warmup_steps = -1
            
        # 초기 Learning Rate
        self.base_lrs = []
        
        for param_group in self.optimizer.param_groups:
            
            base_lr = param_group["lr"]
            param_group["initial_lr"] = base_lr
            self.base_lrs.append(base_lr)
        
        # 현재 Step
        self.current_step = 0
        
    
    def _lr_factor(self, epoch):
        
        # Linear LR
        return (
            max(1.0 - epoch / self.epochs, 0.0)
            * (1.0 - self.min_lr_ratio)
            + self.min_lr_ratio
        )
    
    
    def get_accumulate(self, global_step):
        
        # 목표 Accumulate
        target_accumulate = (
            self.nominal_batch_size 
            / self.batch_size
        )
        
        # Warmup
        if (
            self.warmup_steps > 0
            and global_step <= self.warmup_steps
        ):
            
            progress = global_step / self.warmup_steps
            accumulate = 1.0 + progress * (target_accumulate - 1.0)
            
            return max(1, int(round(accumulate)))
        
        return max(round(target_accumulate), 1)
    
    
    def prepare_batch(self, global_step, epoch):
        
        # 현재 Step
        self.current_step = global_step
        
        # LR 비율
        lr_factor = self._lr_factor(epoch)
        
        # Warmup
        if (
            self.warmup_steps > 0
            and global_step <= self.warmup_steps
        ):
            
            progress = global_step / self.warmup_steps
            
            for param_group, base_lr in zip(
                self.optimizer.param_groups,
                self.base_lrs
            ):
                
                # 시작 LR
                if (
                    param_group.get(
                        "param_group"
                    )
                    == "bias"
                ):
                    start_lr = self.warmup_bias_lr
                else:
                    start_lr = 0.0
                
                # 목표 LR
                target_lr = base_lr * lr_factor
                
                # LR Warmup
                param_group["lr"] = (
                    start_lr
                    + progress
                    * (target_lr - start_lr)
                )
                
                # Monmentum Warmup
                if "momentum" in param_group:
                    
                    param_group["momentum"] = (
                        self.warmup_momentum
                        + progress
                        *(self.momentum - self.warmup_momentum)
                    )
        
        else:
            
            for param_group, base_lr in zip(
                self.optimizer.param_groups,
                self.base_lrs
            ):
                
                # Learning Rate
                param_group["lr"] = base_lr * lr_factor
                
                # Monmentum
                if "momentum" in param_group:
                    param_group["momentum"] = self.momentum
    
    
    def get_last_lr(self):
        
        # 현재 Learning Rate
        return [
            param_group["lr"]
            for param_group
            in self.optimizer.param_groups
        ]
    
    
    def state_dict(self):
        
        # Scheduler 상태
        return {
            "current_step": self.current_stpe
        }
    
    
    def load_state_dict(self, state_dict):
        
        # Scheduler Step 복원
        self.current_step = int(
            state_dict["current_step"]
        )
    
    
def build_scheduler(
    optimizer,
    epochs,
    steps_per_epoch,
    batch_size,
    nominal_batch_size=64,
    warmup_epochs=3.0,
    min_lr_ratio=0.01,
    momentum=0.9,
    warmup_momentum=0.8,
    warmup_bias_lr=0.0
):
    
    # Scheduler 생성
    scheduler = WarmupLinearScheduler(
        optimizer=optimizer,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        nominal_batch_size=nominal_batch_size,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
        momentum=momentum,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr
    )
    
    return scheduler