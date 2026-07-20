import torch
import torch.nn as nn


# 두 개의 특징 맵을 채널 방향으로
# 연결하는 모듈이다.
class Concat(nn.Module):

    def __init__(self):

        # 부모 클래스인 nn.Module을 초기화한다.
        super(Concat, self).__init__()

    def forward(self, x1, x2):

        # PyTorch 이미지 특징 맵의 구조는 다음과 같다.
        #
        # [배치, 채널, 높이, 너비]
        #          ↑
        #        dim=1
        #
        # dim=1로 설정하면 두 특징 맵을
        # 채널 방향으로 이어 붙인다.
        #
        # 예:
        # x1 = [1, 64, 80, 80]
        # x2 = [1, 128, 80, 80]
        #
        # 결과:
        # x = [1, 192, 80, 80]
        x = torch.cat(
            (x1, x2),
            dim=1
        )

        # 연결한 특징 맵을 반환한다.
        return x