#----------------------------------------------
# 라이브러리
#----------------------------------------------

import torch

# 유효 좌표 생성
def make_anchors(
    features,
    strides,
    grid_cell_offset=0.5
):

    # 결과 보관 리스트
    anchor_points = []  # 각 anchor point를 저장할 리스트
    stride_tensors = [] # 각 anchor point에 대한 stride 값을 저장할 리스트

    # P3와 P4, P5 처리
    for index, feature in enumerate(features):

        # 특징 맵 모양 불러옴
        _, _, height, width = feature.shape

        # Tensor 정보
        device = feature.device
        dtype = feature.dtype

        # x 중심 좌표 생성
        grid_x = torch.arange(
            width,
            device=device,
            dtype=dtype
        )

        grid_x = grid_x + grid_cell_offset

        # y 중심 좌표 생성
        grid_y = torch.arange(
            height,
            device=device,
            dtype=dtype
        )

        grid_y = grid_y + grid_cell_offset

        # 모든 y와 x 좌표 조합으로 2차원 좌표 생성
        grid_y, grid_x = torch.meshgrid(
            grid_y,
            grid_x,
            indexing="ij"
        )

        # [H, W, 2]
        points = torch.stack(
            (grid_x, grid_y),
            dim=-1
        )

        # [H, W, 2] → [H × W, 2]
        points = points.reshape(
            -1,
            2
        )

        # 중심 좌표 저장
        anchor_points.append(points)

        # 현재 특징 맵에 대한 stride 값 생성
        stride_tensor = torch.full(
            size=(height * width, 1),
            fill_value=float(strides[index]),
            device=device,
            dtype=dtype
        )

        # stride 값 저장
        stride_tensors.append(stride_tensor)

    # P3, P4, P5의 좌표 후보군을 하나로 연결
    # P3: 80×80 = 6400
    # P4: 40×40 = 1600
    # P5: 20×20 = 400
    # 총합: 8400
    anchor_points = torch.cat(
        anchor_points,
        dim=0
    )

    stride_tensors = torch.cat(
        stride_tensors,
        dim=0
    )

    # anchor_points:  [8400, 2]
    # stride_tensors: [8400, 1]
    return anchor_points, stride_tensors


# 거리 값을 박스로 변환
def dist2bbox(
    distance,
    anchor_points,
    xywh=True
):

    # 네 방향 거리를 좌상단과 우하단으로 그룹 분리
    left_top, right_bottom = distance.chunk(
        chunks=2,
        dim=1
    )

    # 좌상단 좌표
    #
    # x1 = anchor_x - left
    # y1 = anchor_y - top
    x1y1 = anchor_points - left_top

    # 우하단 좌표
    #
    # x2 = anchor_x + right
    # y2 = anchor_y + bottom
    x2y2 = anchor_points + right_bottom

    # x, y, 높이, 너비로 반환
    if xywh:

        # 박스의 중심 좌표
        center = (x1y1 + x2y2) / 2

        # 박스의 높이 및 너비
        width_height = x2y2 - x1y1

        # [x, y, width, height]
        boxes = torch.cat(
            (center, width_height),
            dim=1
        )

    # x1, y1, x2, y2로 반환
    else:

        # [x1, y1, x2, y2]
        boxes = torch.cat(
            (x1y1, x2y2),
            dim=1
        )

    # 반환
    return boxes