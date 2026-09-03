# 라이브러리
import math


class WarmupCosineScheduler:
    
    def __init__(
        self,
        optimizer,
        epochs,
        steps__per_epoch,
        warmup_epcohs=3.0,
        min_lr_ratio=0.01
    ):
        
        # 입력 검사
        if epochs <= 0:
            raise ValueError("epochs는 0보다 커야 합니다.")
        if steps__per_epoch <= 0:
            raise ValueError("steps__per_epoch는 0보다 커야 합니다.")
        if warmup_epcohs < 0:
            raise ValueError("warmup_epcohs는 0보다 크거나 같아야 합니다.")
        if not 0.0 < min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio는 0보다 크고 1보다 작거나 같아야 합니다.")
        
        # 파라미터
        self.optimizer = optimizer
        self.total_steps = epochs * steps__per_epoch
        self.warmup_steps = min(
            int(round(warmup_epcohs * steps__per_epoch)),
            self.total_steps
        )
        self.min_lr_ratio = min_lr_ratio
        
        # 초기화 Learning Rate 저장
        self.base_lrs = [
            param_group["lr"]
            for param_group
            in self.optimizer.param_groups
        ]
        
        # 현재 Step
        self.current_step = 0
        
        # 첫 Step Learning Rate 설정
        self._set_lr(self._get_lr_scale(self.current_step))
        
    def _get_lr_scale(self, step):
        
        # Warmup
        if self.warmup_steps > 0 and step < self.warmup_steps:
            warmup_progress = (step + 1) / self.warmup_steps
            
            return warmup_progress
        
        # 전체 학습이 Warmup으로 끝나는 경우
        if self.total_steps <= self.warmup_steps:
            return 1.0
        
        # Cosine Decay Step 수
        decay_steps = self.total_steps - self.warmup_steps
        
        # Cosine 구간이 1 step인 경우
        if decay_steps == 1:
            return self.min_lr_ratio
        
        # Cosine 진행률
        decay_progress = (
            (step - self.warmup_steps) 
            / (decay_steps - 1)
        )
        
        decay_progress = max(
            0.0,
            min(1.0, decay_progress)
        )
        
        # Cosine 값
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        
        # Learning Rate 비율
        lr_scale = (
            self.min_lr_ratio
            + (1.0 - self.min_lr_ratio) * cosine
        )
        
        return lr_scale
    
    def _set_lr(self, lr_scale):
        
        # OPtimizer Learning Rate 변경
        for(param_group, base_lr) in zip(
            self.optimizer.param_groups,
            self.base_lrs
        ):
            param_group["lr"] = base_lr * lr_scale
    
    def step(self):
        
        # Step 증가
        self.current_step += 1
        
        # Learning Rate 계산
        lr_scale = self._get_lr_scale(self.current_step)
        
        # Learning Rate 적용
        self._set_lr(lr_scale)
        
    def get_last_lr(self):
        
        # 현재 Learning Rate 반환
        return [
            param_group["lr"]
            for param_group
            in self.optimizer.param_groups
        ]
        
    def state_dict(self):
        
        # Scheduler 상태 반환
        return {
            "current_step": self.current_step
        }
        
    def load_state_dict(self, state_dict):
        
        # Scheduler Step 복원
        self.current_step = int(state_dict["current_step"])
        
        # Learning Rate 복원
        lr_scale = self._get_lr_scale(self.current_step)
        self._set_lr(lr_scale)
        
    def bulid_scheduler(
        optimizer,
        epochs,
        steps__per_epoch,
        warmup_epcohs=3.0,
        min_lr_ratio=0.01
    ):
        # Scheduler 생성
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            epochs=epochs,
            steps__per_epoch=steps__per_epoch,
            warmup_epcohs=warmup_epcohs,
            min_lr_ratio=min_lr_ratio
        )
        
        return scheduler