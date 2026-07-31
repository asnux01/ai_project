# import library
import torch

# Create Grid Points
def make_anchors(
    features,
    strides,
    grid_cell_offset=0.5
):

    # Result containers
    anchor_points = []
    stride_tensors = []

    # Process P3, P4 and P5
    for index, feature in enumerate(features):

        # Feature map shape
        _, _, height, width = feature.shape

        # Tensor information
        device = feature.device
        dtype = feature.dtype

        # Grid X coordinates
        grid_x = torch.arange(
            width,
            device=device,
            dtype=dtype
        )

        grid_x = grid_x + grid_cell_offset

        # Grid Y coordinates
        grid_y = torch.arange(
            height,
            device=device,
            dtype=dtype
        )

        grid_y = grid_y + grid_cell_offset

        # Create all X and Y combinations
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

        # Save Grid points
        anchor_points.append(points)

        # Create a stride value for every Grid point
        stride_tensor = torch.full(
            size=(height * width, 1),
            fill_value=float(strides[index]),
            device=device,
            dtype=dtype
        )

        # Save strides
        stride_tensors.append(stride_tensor)

    # Connect P3, P4 and P5 Grid points
    #
    # P3: 80×80 = 6400
    # P4: 40×40 = 1600
    # P5: 20×20 = 400
    #
    # Total: 8400
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


# Convert Distances into Bounding Boxes
def dist2bbox(
    distance,
    anchor_points,
    xywh=True
):

    # Separate:
    # [left, top] and [right, bottom]
    left_top, right_bottom = distance.chunk(
        chunks=2,
        dim=1
    )

    # Top-left coordinate
    #
    # x1 = anchor_x - left
    # y1 = anchor_y - top
    x1y1 = anchor_points - left_top

    # Bottom-right coordinate
    #
    # x2 = anchor_x + right
    # y2 = anchor_y + bottom
    x2y2 = anchor_points + right_bottom

    # Return x, y, width and height
    if xywh:

        # Box center
        center = (x1y1 + x2y2) / 2

        # Box width and height
        width_height = x2y2 - x1y1

        # [x, y, width, height]
        boxes = torch.cat(
            (center, width_height),
            dim=1
        )

    # Return x1, y1, x2 and y2
    else:

        # [x1, y1, x2, y2]
        boxes = torch.cat(
            (x1y1, x2y2),
            dim=1
        )

    # return
    return boxes