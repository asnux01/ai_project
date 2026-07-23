import torch
import torch.nn as nn
    
# 두 개의 특징 맵을 채널 방향으로
# 연결하는 모듈
class Concat(nn.Module):

    def __init__(
        self,
        dimension = 1
        ):
        
        # PyTorch를 사용하기 위해 nn.Module을 초기화
        super(Concat, self).__init__()

        # 연결할 차원 번호
        self.dimension = dimension
        
    def forward(self, x1, x2, x3 = None, x4 = None):
        # 반드시 전달되는 첫 번째와 두 번째 특징 맵을 저장
        feature_maps = [x1, x2]

        # 세 번째 특징 맵이 전달되었으면 추가
        if x3 is not None:
            feature_maps.append(x3)

        # 네 번째 특징 맵이 전달되었으면 추가
        if x4 is not None:
            feature_maps.append(x4)

        # 저장된 특징 맵들을 지정된 차원으로 연결
        x = torch.cat(
            feature_maps,
            dim = self.dimension
        )

        return x