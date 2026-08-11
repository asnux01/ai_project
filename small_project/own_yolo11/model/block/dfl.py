#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch
import torch.nn as nn

class DFL(nn.Module):
    
        # 초기화
        def __init__(
            self,
            reg_max
        ):
            
            # PyTorch 사용을 위해 nn.Module 초기화
            super(DFL, self).__init__()
            
            # 포워드 접근 가능 파라미터
            self.reg_max = reg_max
            
            # Integral Conv
            # 확률 분포를 기대 거리(expected distance)로 변환
            self.conv = nn.Conv2d(
                in_channels=reg_max,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )

            # Integral weights: [0, 1, 2, ..., reg_max - 1]
            projection = torch.arange(
                reg_max,
                dtype=torch.float32
            )

            projection = projection.reshape(
                1,
                reg_max,
                1,
                1
            )

            # Conv weight를 거리 구간 값으로 설정
            self.conv.weight.data.copy_(projection)

            # Projection weight는 학습 대상이 아닌 고정 값
            self.conv.requires_grad_(False)

        # 포워드
        def forward(self, x):

            # 입력 형태:
            #
            # x: [batch_size, 4 * reg_max, num_anchors]
            batch_size, _, num_anchors = x.shape
            
            # 4 방향으로 logit 분리
            # [B, 4 * reg_max, A]
            #             ↓
            # [B, 4, reg_max, A]
            x = x.reshape(
                batch_size,
                4,
                self.reg_max,
                num_anchors
            )

            # reg_max 차원을 Conv2d의 채널 차원으로 이동
            # [B, 4, reg_max, A]
            #             ↓
            # [B, reg_max, 4, A]
            x = x.transpose(1, 2)

            # 각 방향의 거리 logit을 확률 분포로 변환
            x = torch.softmax(
                x,
                dim=1
            )

            # 각 방향의 거리 기댓값을 계산
            # 0*p0 + 1*p1 + ... + (reg_max-1)*p_last
            # [B, reg_max, 4, A]
            #             ↓
            # [B, 1, 4, A]
            x = self.conv(x)

            # 불필요한 단일 채널 차원을 제거
            # [B, 1, 4, A]
            #             ↓
            # [B, 4, A]
            x = x.reshape(
                batch_size,
                4,
                num_anchors
            )

            # 반환
            return x