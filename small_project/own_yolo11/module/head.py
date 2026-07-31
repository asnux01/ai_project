# import library
import torch
import torch.nn as nn

from ..block import BoxBranch, ClassBranch, DFL, make_anchors, dist2bbox

class Head(nn.Module):

    # initialize
    def __init__(
        self,
        num_classes,
        in_channels,
        reg_max=16,
        strides=(8, 16, 32)
    ):

        # nn.Module reset to use PyTorch
        super(Head, self).__init__()

        # parameter
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.num_levels = len(in_channels)

        # Number of Box branch output channels
        #
        # reg_max=16
        # left, top, right, bottom × 16
        #
        # 4 × 16 = 64
        self.box_channels = 4 * reg_max

        # Box branch hidden channels
        ch_bh = max(
            16,
            in_channels[0] // 4,
            4 * reg_max
        )

        # Class branch hidden channels
        ch_ch = max(
            in_channels[0],
            min(num_classes, 100)
        )

        # Create P3, P4 and P5 Box branches
        # Module container
        self.box_branches = nn.ModuleList()

        # Create one Box branch for each feature map
        for channel in in_channels:

            # Create Box branch
            box_branch = BoxBranch(
                in_channels=channel,
                hidden_channels=ch_bh,
                reg_max=reg_max
            )

            # Add Box branch to ModuleList
            self.box_branches.append(
                box_branch
            )
            
        # Create P3, P4 and P5 Class branches
        # Module container
        self.class_branches = nn.ModuleList()

        # Create one Class branch for each feature map
        for channel in in_channels:

            # Create Class branch
            class_branch = ClassBranch(
                in_channels=channel,
                hidden_channels=ch_ch,
                num_classes=num_classes
            )

            # Add Class branch to ModuleList
            self.class_branches.append(
                class_branch
            )

        # DFL Projection Block
        self.dfl = DFL(
            reg_max=reg_max
        )

    # forward
    def forward(self, features):

        # features:
        #
        # features[0] = P3
        # features[1] = P4
        # features[2] = P5

        # Batch size
        batch_size = features[0].shape[0]

        # Result containers
        box_outputs = []
        class_outputs = []

        # --------------------------------------------------
        # Run P3, P4 and P5 branches
        # --------------------------------------------------

        for index in range(self.num_levels):

            # Current feature map
            feature = features[index]

            # Current Box branch
            box_branch = self.box_branches[index]

            # Current Class branch
            class_branch = self.class_branches[index]

            # Run Box branch
            box_output = box_branch(
                feature
            )

            # Run Class branch
            class_output = class_branch(
                feature
            )

            # Save Box output
            box_outputs.append(
                box_output
            )

            # Save Class output
            class_outputs.append(
                class_output
            )

        # --------------------------------------------------
        # Reshape Box outputs
        # --------------------------------------------------

        box_reshape_outputs = []

        for box_output in box_outputs:

            # Before:
            # [B, 4 × reg_max, H, W]
            #
            # After:
            # [B, 4 × reg_max, H × W]
            box_output = box_output.reshape(
                batch_size,
                self.box_channels,
                -1
            )

            # Save reshaped Box output
            box_reshape_outputs.append(
                box_output
            )

        # Connect P3, P4 and P5 Box outputs
        #
        # P3: [B, 64, 6400]
        # P4: [B, 64, 1600]
        # P5: [B, 64,  400]
        #
        # Result: [B, 64, 8400]
        box_logits = torch.cat(
            box_reshape_outputs,
            dim=2
        )

        # --------------------------------------------------
        # Reshape Class outputs
        # --------------------------------------------------

        class_reshape_outputs = []

        for class_output in class_outputs:

            # Before:
            # [B, num_classes, H, W]
            #
            # After:
            # [B, num_classes, H × W]
            class_output = class_output.reshape(
                batch_size,
                self.num_classes,
                -1
            )

            # Save reshaped Class output
            class_reshape_outputs.append(
                class_output
            )

        # Connect P3, P4 and P5 Class outputs
        #
        # P3: [B, num_classes, 6400]
        # P4: [B, num_classes, 1600]
        # P5: [B, num_classes,  400]
        #
        # Result: [B, num_classes, 8400]
        class_logits = torch.cat(
            class_reshape_outputs,
            dim=2
        )

        # --------------------------------------------------
        # Raw outputs
        # --------------------------------------------------

        raw_outputs = {
            "box_logits": box_logits,
            "class_logits": class_logits,
            "features": features
        }

        # During training, return raw outputs
        if self.training:
            return raw_outputs

        # --------------------------------------------------
        # Inference
        # --------------------------------------------------

        # DFL Projection
        #
        # Before:
        # [B, 4 × reg_max, 8400]
        #
        # After:
        # [B, 4, 8400]
        #
        # Four distances:
        # left, top, right, bottom
        distance = self.dfl(
            box_logits
        )

        # --------------------------------------------------
        # Create Grid points
        # --------------------------------------------------

        # anchor_points:
        # [8400, 2]
        #
        # stride_tensor:
        # [8400, 1]
        anchor_points, stride_tensor = make_anchors(
            features=features,
            strides=self.strides,
            grid_cell_offset=0.5
        )

        # --------------------------------------------------
        # Change Anchor point shape
        # --------------------------------------------------

        # Before:
        # [8400, 2]
        #
        # After transpose:
        # [2, 8400]
        anchor_points = anchor_points.transpose(
            0,
            1
        )

        # Before:
        # [2, 8400]
        #
        # After:
        # [1, 2, 8400]
        anchor_points = anchor_points.unsqueeze(
            0
        )

        # --------------------------------------------------
        # Change Stride shape
        # --------------------------------------------------

        # Before:
        # [8400, 1]
        #
        # After transpose:
        # [1, 8400]
        stride_tensor = stride_tensor.transpose(
            0,
            1
        )

        # Before:
        # [1, 8400]
        #
        # After:
        # [1, 1, 8400]
        stride_tensor = stride_tensor.unsqueeze(
            0
        )

        # --------------------------------------------------
        # Convert distances into Bounding Boxes
        # --------------------------------------------------

        # Combine:
        #
        # Grid point
        #     +
        # left, top, right, bottom distances
        #
        # Result:
        # Feature-map coordinate boxes
        boxes = dist2bbox(
            distance=distance,
            anchor_points=anchor_points,
            xywh=True
        )

        # --------------------------------------------------
        # Apply Stride
        # --------------------------------------------------

        # Feature-map coordinates
        #            ↓
        # Original-image pixel coordinates
        boxes = boxes * stride_tensor

        # --------------------------------------------------
        # Calculate Class probabilities
        # --------------------------------------------------

        # Raw class logits
        #         ↓
        # Class probabilities between 0 and 1
        class_probabilities = torch.sigmoid(
            class_logits
        )

        # --------------------------------------------------
        # Combine Boxes and Class probabilities
        # --------------------------------------------------

        # boxes:
        # [B, 4, 8400]
        #
        # class_probabilities:
        # [B, num_classes, 8400]
        #
        # final_output:
        # [B, 4 + num_classes, 8400]
        final_output = torch.cat(
            (
                boxes,
                class_probabilities,
            ),
            dim=1
        )

        # NMS is performed outside DetectHead.
        return final_output, raw_outputs